#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "../include/utils.cuh"

namespace cg = cooperative_groups;

namespace mhc {

constexpr int STREAM_MIX_TC_THRESHOLD = 32;

template<int BLOCK_SIZE>
__global__ void stream_add_kernel(float* __restrict__ out, const float* __restrict__ a,
                                  const float* __restrict__ b, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_fused_sigmoid_kernel(floatX* __restrict__ out,
                                                           float* __restrict__ H_pre_activated,
                                                           const float* __restrict__ inp,
                                                           const float* __restrict__ H_pre_raw,
                                                           int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += s_H_pre[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = (floatX)sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_from_bf16_fused_sigmoid_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const floatX* __restrict__ inp,
    const float* __restrict__ H_post_raw, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int stream = remainder / C;
    int c = remainder % C;
    out[idx] = s_H_post[stream] * (float)inp[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_fused_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const float* __restrict__ x_inp,
    const floatX* __restrict__ y_norm, const float* __restrict__ H_post_raw,
    const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += s_M[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + s_H_post[i] * (float)y_norm[b * C + c];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_fused_vec4_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const float* __restrict__ x_inp,
    const floatX* __restrict__ y_norm, const float* __restrict__ H_post_raw,
    const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];
    __shared__ float s_x_buf[2][MAX_N * 256];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int vec_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (vec_idx >= B * n * C4)
        return;

    int b = vec_idx / (n * C4);
    int remainder = vec_idx % (n * C4);
    int i = remainder / C4;
    int c4 = remainder % C4;
    int c_base = c4 * 4;

    int buf_idx = 0;
    for (int j = 0; j < n; j++) {
        const float4* x_vec = reinterpret_cast<const float4*>(x_inp + b * n * C + j * C + c_base);
        float4 x = *x_vec;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 0] = x.x;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 1] = x.y;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 2] = x.z;
        s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 3] = x.w;
    }
    __syncthreads();

    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n) {
            float m_ij = s_M[i * n + j];
            result.x += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 0];
            result.y += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 1];
            result.z += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 2];
            result.w += m_ij * s_x_buf[buf_idx][j * BLOCK_SIZE * 4 + threadIdx.x * 4 + 3];
        }
    }

    const float4* y_vec = reinterpret_cast<const float4*>(
        reinterpret_cast<const __nv_bfloat16*>(y_norm) + b * C + c_base);
    __nv_bfloat16 y_bf16[4];
    *reinterpret_cast<float2*>(y_bf16) = *reinterpret_cast<const float2*>(y_vec);
    float h_i = s_H_post[i];
    result.x += h_i * __bfloat162float(y_bf16[0]);
    result.y += h_i * __bfloat162float(y_bf16[1]);
    result.z += h_i * __bfloat162float(y_bf16[2]);
    result.w += h_i * __bfloat162float(y_bf16[3]);

    float4* out_vec = reinterpret_cast<float4*>(out + b * n * C + i * C + c_base);
    *out_vec = result;
}

template<int BLOCK_SIZE>
__global__ void stream_mix_large_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                        const float* __restrict__ M, int B, int n, int C) {
    extern __shared__ float s_M_dyn[];
    for (int i = threadIdx.x; i < n * n; i += BLOCK_SIZE)
        s_M_dyn[i] = M[i];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += s_M_dyn[i * n + j] * inp[b * n * C + j * C + c];
    out[idx] = sum;
}

template<int BLOCK_SIZE>
__global__ void transpose_BnC_to_nBC_colmajor_kernel(float* __restrict__ out,
                                                     const float* __restrict__ inp, int B, int n,
                                                     int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;
    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;
    out[i * B * C + c * B + b] = inp[idx];
}

template<int BLOCK_SIZE>
__global__ void transpose_nBC_colmajor_to_BnC_kernel(float* __restrict__ out,
                                                     const float* __restrict__ inp, int B, int n,
                                                     int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;
    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;
    out[idx] = inp[i * B * C + c * B + b];
}

class StreamMixTC {
  public:
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Mdesc, Xdesc, Ydesc;
    float *X_transposed, *Y_transposed;
    int B, n, C;
    bool initialized = false;

    void init(int B_, int n_, int C_) {
        B = B_;
        n = n_;
        C = C_;
        cublasLtCreate(&handle);
        cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
        cublasLtMatrixLayoutCreate(&Mdesc, CUDA_R_32F, n, n, n);
        cublasLtMatrixLayoutCreate(&Xdesc, CUDA_R_32F, n, B * C, n);
        cublasLtMatrixLayoutCreate(&Ydesc, CUDA_R_32F, n, B * C, n);
        cudaMalloc(&X_transposed, B * n * C * sizeof(float));
        cudaMalloc(&Y_transposed, B * n * C * sizeof(float));
        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;
        cublasLtMatrixLayoutDestroy(Mdesc);
        cublasLtMatrixLayoutDestroy(Xdesc);
        cublasLtMatrixLayoutDestroy(Ydesc);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtDestroy(handle);
        cudaFree(X_transposed);
        cudaFree(Y_transposed);
        initialized = false;
    }

    void forward(float* out, const float* inp, const float* M, cudaStream_t stream = nullptr) {
        constexpr int BLOCK = 256;
        int total = B * n * C;
        int blocks = (total + BLOCK - 1) / BLOCK;
        transpose_BnC_to_nBC_colmajor_kernel<BLOCK>
            <<<blocks, BLOCK, 0, stream>>>(X_transposed, inp, B, n, C);
        float alpha = 1.0f, beta = 0.0f;
        cublasLtMatmul(handle, matmulDesc, &alpha, M, Mdesc, X_transposed, Xdesc, &beta,
                       Y_transposed, Ydesc, Y_transposed, Ydesc, nullptr, nullptr, 0, stream);
        transpose_nBC_colmajor_to_BnC_kernel<BLOCK>
            <<<blocks, BLOCK, 0, stream>>>(out, Y_transposed, B, n, C);
    }
};

inline void stream_add(float* out, const float* a, const float* b, int size,
                       cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    stream_add_kernel<BLOCK><<<(size + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(out, a, b, size);
}

inline void stream_aggregate_bf16_fused_sigmoid(floatX* out, float* H_pre_activated,
                                                const float* inp, const float* H_pre_raw, int B,
                                                int n, int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    int blocks = (B * C + BLOCK - 1) / BLOCK;
    if (n <= 8) {
#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK, 1, 1};
        config.gridDim = {(unsigned int)blocks, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchKernelEx(&config, stream_aggregate_bf16_fused_sigmoid_kernel<BLOCK, MAX_N>, out,
                           H_pre_activated, inp, H_pre_raw, B, n, C);
#else
        stream_aggregate_bf16_fused_sigmoid_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(out, H_pre_activated, inp, H_pre_raw, B, n, C);
#endif
    } else {
        fprintf(stderr, "stream_aggregate_bf16_fused_sigmoid: n > 8 not implemented\n");
    }
}

inline void stream_distribute_from_bf16_fused_sigmoid(float* out, float* H_post_activated,
                                                      const floatX* inp, const float* H_post_raw,
                                                      int B, int n, int C,
                                                      cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;
    if (n <= 8) {
        stream_distribute_from_bf16_fused_sigmoid_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(out, H_post_activated, inp, H_post_raw, B, n, C);
    } else {
        fprintf(stderr, "stream_distribute_from_bf16_fused_sigmoid: n > 8 not implemented\n");
    }
}

inline void stream_distribute_mix_add_fused(float* out, float* H_post_activated, const float* x_inp,
                                            const floatX* y_norm, const float* H_post_raw,
                                            const float* M, int B, int n, int C,
                                            cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    if (n <= 8) {
#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        if (C % 4 == 0 && C >= 64) {
            int blocks = (B * n * (C / 4) + BLOCK - 1) / BLOCK;
            config.gridDim = {(unsigned int)blocks, 1, 1};
            cudaLaunchKernelEx(&config, stream_distribute_mix_add_fused_vec4_kernel<BLOCK, MAX_N>,
                               out, H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C);
        } else {
            int blocks = (B * n * C + BLOCK - 1) / BLOCK;
            config.gridDim = {(unsigned int)blocks, 1, 1};
            cudaLaunchKernelEx(&config, stream_distribute_mix_add_fused_kernel<BLOCK, MAX_N>, out,
                               H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C);
        }
#else
        if (C % 4 == 0 && C >= 64) {
            int blocks = (B * n * (C / 4) + BLOCK - 1) / BLOCK;
            stream_distribute_mix_add_fused_vec4_kernel<BLOCK, MAX_N><<<blocks, BLOCK, 0, stream>>>(
                out, H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C);
        } else {
            int blocks = (B * n * C + BLOCK - 1) / BLOCK;
            stream_distribute_mix_add_fused_kernel<BLOCK, MAX_N><<<blocks, BLOCK, 0, stream>>>(
                out, H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C);
        }
#endif
    } else {
        fprintf(stderr, "stream_distribute_mix_add_fused: n > 8 not implemented\n");
    }
}

inline void stream_mix_large(float* out, const float* inp, const float* M, int B, int n, int C,
                             cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;
    size_t smem = n * n * sizeof(float);
    stream_mix_large_kernel<BLOCK><<<blocks, BLOCK, smem, stream>>>(out, inp, M, B, n, C);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_bf16_dynamic_kernel(floatX* __restrict__ out, const float* __restrict__ inp,
                                     const float* __restrict__ H_pre, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    const float* h = H_pre + b * n;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += h[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = (floatX)sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_dynamic_kernel(
    float* __restrict__ out, const float* __restrict__ x_inp, const floatX* __restrict__ y_norm,
    const float* __restrict__ H_post, const float* __restrict__ M, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    const float* h = H_post + b * n;
    const float* m = M + b * n * n;

    float mix_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += m[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + h[i] * (float)y_norm[b * C + c];
}

inline void stream_aggregate_bf16_dynamic(floatX* out, const float* inp, const float* H_pre, int B,
                                          int n, int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    int blocks = (B * C + BLOCK - 1) / BLOCK;
    if (n <= 8) {
#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK, 1, 1};
        config.gridDim = {(unsigned int)blocks, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchKernelEx(&config, stream_aggregate_bf16_dynamic_kernel<BLOCK, MAX_N>, out, inp,
                           H_pre, B, n, C);
#else
        stream_aggregate_bf16_dynamic_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(out, inp, H_pre, B, n, C);
#endif
    } else {
        fprintf(stderr, "stream_aggregate_bf16_dynamic: n > 8 not implemented\n");
    }
}

inline void stream_distribute_mix_add_fused_dynamic(float* out, const float* x_inp,
                                                    const floatX* y_norm, const float* H_post,
                                                    const float* M, int B, int n, int C,
                                                    cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;
    if (n <= 8) {
#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK, 1, 1};
        config.gridDim = {(unsigned int)blocks, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchKernelEx(&config, stream_distribute_mix_add_dynamic_kernel<BLOCK, MAX_N>, out,
                           x_inp, y_norm, H_post, M, B, n, C);
#else
        stream_distribute_mix_add_dynamic_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(out, x_inp, y_norm, H_post, M, B, n, C);
#endif
    } else {
        fprintf(stderr, "stream_distribute_mix_add_fused_dynamic: n > 8 not implemented\n");
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_backward_dx_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                                    const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n)
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;
    d_inp[idx] = grad[b * C + c] * s_H_pre[i];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_backward_dH_partial_kernel(float* __restrict__ partials,
                                                            const float* __restrict__ grad,
                                                            const float* __restrict__ inp, int B,
                                                            int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float s_warp_sums[MAX_N][BLOCK_SIZE / 32];

    float local_sum[MAX_N] = {0.0f};
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C;
         idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float g = grad[idx];
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                local_sum[i] += g * inp[b * n * C + i * C + c];
        }
    }

    int warp_id = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float warp_sum = cg::reduce(warp, local_sum[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_sums[i][warp_id] = warp_sum;
        }
    }
    block.sync();

    if (threadIdx.x < n) {
        float block_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 32; w++)
            block_sum += s_warp_sums[threadIdx.x][w];
        partials[blockIdx.x * n + threadIdx.x] = block_sum;
    }
}

template<int MAX_N>
__global__ void reduce_partials_kernel(float* __restrict__ out, const float* __restrict__ partials,
                                       int n, int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int i = blockIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n + i];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 32] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++)
            total += s_warp_sums[w];
        out[i] = total;
    }
}

inline void stream_aggregate_backward(float* d_inp, float* d_H_pre, const float* grad,
                                      const float* inp, const float* H_pre, int B, int n, int C,
                                      float* workspace, int workspace_num_blocks,
                                      cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;
    int blocks_dx = (B * n * C + BLOCK - 1) / BLOCK;
    stream_aggregate_backward_dx_kernel<BLOCK, MAX_N>
        <<<blocks_dx, BLOCK, 0, stream>>>(d_inp, grad, H_pre, B, n, C);
    stream_aggregate_backward_dH_partial_kernel<BLOCK, MAX_N>
        <<<workspace_num_blocks, BLOCK, 0, stream>>>(workspace, grad, inp, B, n, C);
    reduce_partials_kernel<MAX_N>
        <<<n, 128, 0, stream>>>(d_H_pre, workspace, n, workspace_num_blocks);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_dx_dy_kernel(
    float* __restrict__ d_x, float* __restrict__ d_y_norm, const float* __restrict__ grad,
    const float* __restrict__ M, const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int j = remainder / C;
    int c = remainder % C;

    float dx_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            dx_sum += s_M[i * n + j] * grad[b * n * C + i * C + c];
    }
    d_x[idx] = dx_sum;

    if (j == 0) {
        float dy_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                dy_sum += s_H[i] * grad[b * n * C + i * C + c];
        }
        d_y_norm[b * C + c] = dy_sum;
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_dx_dy_vec4_kernel(
    float* __restrict__ d_x, float* __restrict__ d_y_norm, const float* __restrict__ grad,
    const float* __restrict__ M, const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int vec_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (vec_idx >= B * n * C4)
        return;

    int b = vec_idx / (n * C4);
    int remainder = vec_idx % (n * C4);
    int j = remainder / C4;
    int c_base = (remainder % C4) * 4;

    float4 dx_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 dy_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float4 g = *reinterpret_cast<const float4*>(grad + b * n * C + i * C + c_base);
            float m_ij = s_M[i * n + j];
            dx_acc.x += m_ij * g.x;
            dx_acc.y += m_ij * g.y;
            dx_acc.z += m_ij * g.z;
            dx_acc.w += m_ij * g.w;
            if (j == 0) {
                float h_i = s_H[i];
                dy_acc.x += h_i * g.x;
                dy_acc.y += h_i * g.y;
                dy_acc.z += h_i * g.z;
                dy_acc.w += h_i * g.w;
            }
        }
    }

    *reinterpret_cast<float4*>(d_x + b * n * C + j * C + c_base) = dx_acc;
    if (j == 0)
        *reinterpret_cast<float4*>(d_y_norm + b * C + c_base) = dy_acc;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_partials_kernel(
    float* __restrict__ partials_M, float* __restrict__ partials_H, const float* __restrict__ grad,
    const float* __restrict__ x, const float* __restrict__ y_norm, int B, int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float s_warp_M[MAX_N][MAX_N][NUM_WARPS];
    __shared__ float s_warp_H[MAX_N][NUM_WARPS];

    float local_M[MAX_N][MAX_N];
    float local_H[MAX_N];
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        local_H[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < MAX_N; j++)
            local_M[i][j] = 0.0f;
    }

    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C;
         idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float y_val = y_norm[b * C + c];
#pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n) {
                float g = grad[b * n * C + i * C + c];
                local_H[i] += g * y_val;
#pragma unroll
                for (int j = 0; j < MAX_N; j++) {
                    if (j < n)
                        local_M[i][j] += g * x[b * n * C + j * C + c];
                }
            }
        }
    }

    int warp_id = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
#pragma unroll
            for (int j = 0; j < MAX_N; j++) {
                if (j < n) {
                    float ws = cg::reduce(warp, local_M[i][j], cg::plus<float>());
                    if (warp.thread_rank() == 0)
                        s_warp_M[i][j][warp_id] = ws;
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float ws = cg::reduce(warp, local_H[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_H[i][warp_id] = ws;
        }
    }
    block.sync();

    if (threadIdx.x < n * n) {
        int i = threadIdx.x / n, j = threadIdx.x % n;
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_M[i][j][w];
        partials_M[blockIdx.x * n * n + threadIdx.x] = bs;
    }
    if (threadIdx.x < n) {
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_H[threadIdx.x][w];
        partials_H[blockIdx.x * n + threadIdx.x] = bs;
    }
}

template<int MAX_N>
__global__ void reduce_partials_matrix_kernel(float* __restrict__ out,
                                              const float* __restrict__ partials, int n,
                                              int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int k = blockIdx.x;
    if (k >= n * n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n * n + k];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 32] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++)
            total += s_warp_sums[w];
        out[k] = total;
    }
}

inline void stream_distribute_mix_backward_fused(float* d_x, float* d_y_norm, float* d_M,
                                                 float* d_H_post, const float* grad, const float* x,
                                                 const float* y_norm, const float* M,
                                                 const float* H_post, int B, int n, int C,
                                                 float* workspace_M, float* workspace_H,
                                                 int workspace_num_blocks,
                                                 cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256, MAX_N = 8;

    if (C % 4 == 0 && C >= 64) {
        int blocks = (B * n * (C / 4) + BLOCK - 1) / BLOCK;
        stream_distribute_mix_backward_dx_dy_vec4_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(d_x, d_y_norm, grad, M, H_post, B, n, C);
    } else {
        int blocks = (B * n * C + BLOCK - 1) / BLOCK;
        stream_distribute_mix_backward_dx_dy_kernel<BLOCK, MAX_N>
            <<<blocks, BLOCK, 0, stream>>>(d_x, d_y_norm, grad, M, H_post, B, n, C);
    }

    stream_distribute_mix_backward_partials_kernel<BLOCK, MAX_N>
        <<<workspace_num_blocks, BLOCK, 0, stream>>>(workspace_M, workspace_H, grad, x, y_norm, B,
                                                     n, C);
    reduce_partials_matrix_kernel<MAX_N>
        <<<n * n, 128, 0, stream>>>(d_M, workspace_M, n, workspace_num_blocks);
    reduce_partials_kernel<MAX_N>
        <<<n, 128, 0, stream>>>(d_H_post, workspace_H, n, workspace_num_blocks);
}

} // namespace mhc
