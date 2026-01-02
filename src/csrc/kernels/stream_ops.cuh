#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include "../include/mhc_types.h"

namespace mhc {

constexpr int STREAM_MIX_TC_THRESHOLD = 32;

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                        const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];

    if (threadIdx.x < n) {
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * C;

    if (idx >= total)
        return;

    int b = idx / C;
    int c = idx % C;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            int src_idx = b * n * C + i * C + c;
            sum += s_H_pre[i] * inp[src_idx];
        }
    }

    out[idx] = sum;
}

template<int BLOCK_SIZE>
__global__ void
stream_aggregate_large_kernel(float* __restrict__ out, const float* __restrict__ inp,
                              const float* __restrict__ H_pre, int B, int n, int C) {
    extern __shared__ float s_H_pre_dyn[];

    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        s_H_pre_dyn[i] = H_pre[i];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * C;

    if (idx >= total)
        return;

    int b = idx / C;
    int c = idx % C;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        int src_idx = b * n * C + i * C + c;
        sum += s_H_pre_dyn[i] * inp[src_idx];
    }

    out[idx] = sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                         const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n) {
        s_H_post[threadIdx.x] = H_post[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int stream = remainder / C;
    int c = remainder % C;

    int src_idx = b * C + c;
    out[idx] = s_H_post[stream] * inp[src_idx];
}

template<int BLOCK_SIZE>
__global__ void
stream_distribute_large_kernel(float* __restrict__ out, const float* __restrict__ inp,
                               const float* __restrict__ H_post, int B, int n, int C) {
    extern __shared__ float s_H_post_dyn[];

    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        s_H_post_dyn[i] = H_post[i];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int stream = remainder / C;
    int c = remainder % C;

    int src_idx = b * C + c;
    out[idx] = s_H_post_dyn[stream] * inp[src_idx];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_mix_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                  const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];

    if (threadIdx.x < n * n) {
        s_M[threadIdx.x] = M[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n) {
            int src_idx = b * n * C + j * C + c;
            sum += s_M[i * n + j] * inp[src_idx];
        }
    }

    out[idx] = sum;
}

template<int BLOCK_SIZE>
__global__ void transpose_BnC_to_nBC_colmajor_kernel(float* __restrict__ out,
                                                     const float* __restrict__ inp, int B, int n,
                                                     int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int j = remainder / C;
    int c = remainder % C;

    int src_idx = b * n * C + j * C + c;
    int k = b * C + c;
    int dst_idx = j + k * n;
    out[dst_idx] = inp[src_idx];
}

template<int BLOCK_SIZE>
__global__ void transpose_nBC_colmajor_to_BnC_kernel(float* __restrict__ out,
                                                     const float* __restrict__ inp, int B, int n,
                                                     int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    int k = b * C + c;
    int src_idx = i + k * n;
    out[idx] = inp[src_idx];
}

template<int BLOCK_SIZE>
__global__ void stream_add_kernel(float* __restrict__ out, const float* __restrict__ a,
                                  const float* __restrict__ b, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_kernel(floatX* __restrict__ out,
                                             const float* __restrict__ inp,
                                             const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];

    if (threadIdx.x < n) {
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * C;

    if (idx >= total)
        return;

    int b = idx / C;
    int c = idx % C;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            int src_idx = b * n * C + i * C + c;
            sum += s_H_pre[i] * inp[src_idx];
        }
    }

    out[idx] = (floatX)sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_distribute_from_bf16_kernel(float* __restrict__ out, const floatX* __restrict__ inp,
                                   const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n) {
        s_H_post[threadIdx.x] = H_post[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int stream = remainder / C;
    int c = remainder % C;

    int src_idx = b * C + c;
    float val = (float)inp[src_idx];
    out[idx] = s_H_post[stream] * val;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_mix_add_kernel(float* __restrict__ out, const float* __restrict__ x_inp,
                                      const float* __restrict__ y_dist, const float* __restrict__ M,
                                      int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];

    if (threadIdx.x < n * n) {
        s_M[threadIdx.x] = M[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n) {
            int src_idx = b * n * C + j * C + c;
            mix_sum += s_M[i * n + j] * x_inp[src_idx];
        }
    }

    out[idx] = mix_sum + y_dist[idx];
}

template<int BLOCK_SIZE>
__global__ void stream_mix_add_large_kernel(float* __restrict__ out,
                                            const float* __restrict__ x_inp,
                                            const float* __restrict__ y_dist,
                                            const float* __restrict__ M, int B, int n, int C) {
    extern __shared__ float s_M_dyn[];

    for (int i = threadIdx.x; i < n * n; i += BLOCK_SIZE) {
        s_M_dyn[i] = M[i];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_sum = 0.0f;
    for (int j = 0; j < n; j++) {
        int src_idx = b * n * C + j * C + c;
        mix_sum += s_M_dyn[i * n + j] * x_inp[src_idx];
    }

    out[idx] = mix_sum + y_dist[idx];
}

inline void stream_aggregate(float* out, const float* inp, const float* H_pre, int B, int n, int C,
                             cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_aggregate_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, H_pre, B, n, C);
    } else {
        size_t smem = n * sizeof(float);
        stream_aggregate_large_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem, stream>>>(out, inp, H_pre, B, n, C);
    }
}

inline void stream_distribute(float* out, const float* inp, const float* H_post, int B, int n,
                              int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_distribute_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, H_post, B, n, C);
    } else {
        size_t smem = n * sizeof(float);
        stream_distribute_large_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem, stream>>>(out, inp, H_post, B, n, C);
    }
}

struct StreamMixTC {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t M_desc;
    cublasLtMatrixLayout_t X_desc;
    cublasLtMatrixLayout_t Y_desc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;

    float* x_transposed;
    float* y_transposed;

    int B, n, C;
    bool initialized;

    StreamMixTC()
        : initialized(false), workspace(nullptr), x_transposed(nullptr), y_transposed(nullptr) {}

    void init(int batch, int streams, int hidden, size_t ws_size = 4 * 1024 * 1024) {
        B = batch;
        n = streams;
        C = hidden;
        workspace_size = ws_size;

        CHECK_CUBLAS(cublasLtCreate(&handle));

        cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        cudaDataType_t data_type = CUDA_R_32F;

        CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, compute_type, data_type));

        cublasOperation_t trans_a = CUBLAS_OP_T;
        cublasOperation_t trans_b = CUBLAS_OP_N;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)));

        int64_t M_rows = n, M_cols = n;
        int64_t X_rows = n, X_cols = B * C;
        int64_t Y_rows = n, Y_cols = B * C;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&M_desc, data_type, M_cols, M_rows, M_cols));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&X_desc, data_type, X_rows, X_cols, X_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Y_desc, data_type, Y_rows, Y_cols, Y_rows));

        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference,
                                                          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)));

        int returned_results = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, M_desc, X_desc, Y_desc,
                                                    Y_desc, preference, 1, &heuristic,
                                                    &returned_results));

        if (returned_results == 0) {
            fprintf(stderr, "StreamMixTC: No cuBLASLt algorithm found\n");
            exit(EXIT_FAILURE);
        }

        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
        CHECK_CUDA(cudaMalloc(&x_transposed, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&y_transposed, B * n * C * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(M_desc);
        cublasLtMatrixLayoutDestroy(X_desc);
        cublasLtMatrixLayoutDestroy(Y_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtDestroy(handle);

        cudaFree(workspace);
        cudaFree(x_transposed);
        cudaFree(y_transposed);

        initialized = false;
    }

    void forward(float* out, const float* inp, const float* M, cudaStream_t stream = nullptr) {
        constexpr int BLOCK_SIZE = 256;
        int total = B * n * C;
        int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

        transpose_BnC_to_nBC_colmajor_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(x_transposed, inp, B, n, C);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasLtMatmul(handle, matmul_desc, &alpha, M, M_desc, x_transposed, X_desc,
                                    &beta, y_transposed, Y_desc, y_transposed, Y_desc,
                                    &heuristic.algo, workspace, workspace_size, stream));

        transpose_nBC_colmajor_to_BnC_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, y_transposed, B, n, C);
    }
};

template<int BLOCK_SIZE>
__global__ void stream_mix_large_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                        const float* __restrict__ M, int B, int n, int C) {
    extern __shared__ float s_M_dyn[];

    for (int i = threadIdx.x; i < n * n; i += BLOCK_SIZE) {
        s_M_dyn[i] = M[i];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        int src_idx = b * n * C + j * C + c;
        sum += s_M_dyn[i * n + j] * inp[src_idx];
    }

    out[idx] = sum;
}

inline void stream_mix(float* out, const float* inp, const float* M, int B, int n, int C,
                       cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_mix_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, M, B, n, C);
    } else {
        size_t smem = n * n * sizeof(float);
        stream_mix_large_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem, stream>>>(out, inp, M, B, n, C);
    }
}

inline void stream_add(float* out, const float* a, const float* b, int size,
                       cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    stream_add_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, a, b, size);
}

inline void stream_aggregate_bf16(floatX* out, const float* inp, const float* H_pre, int B, int n,
                                  int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_aggregate_bf16_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, H_pre, B, n, C);
    } else {
        fprintf(stderr, "stream_aggregate_bf16: n > 8 not implemented\n");
    }
}

inline void stream_distribute_from_bf16(float* out, const floatX* inp, const float* H_post, int B,
                                        int n, int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_distribute_from_bf16_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, H_post, B, n, C);
    } else {
        fprintf(stderr, "stream_distribute_from_bf16: n > 8 not implemented\n");
    }
}

inline void stream_mix_add(float* out, const float* x_inp, const float* y_dist, const float* M,
                           int B, int n, int C, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n <= 8) {
        constexpr int MAX_N = 8;
        stream_mix_add_kernel<BLOCK_SIZE, MAX_N>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, x_inp, y_dist, M, B, n, C);
    } else {
        size_t smem = n * n * sizeof(float);
        stream_mix_add_large_kernel<BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, smem, stream>>>(out, x_inp, y_dist, M, B, n, C);
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_backward_dx_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                                    const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];

    if (threadIdx.x < n) {
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    int grad_idx = b * C + c;
    d_inp[idx] = grad[grad_idx] * s_H_pre[i];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_aggregate_backward_dH_kernel(float* __restrict__ d_H_pre, const float* __restrict__ grad,
                                    const float* __restrict__ inp, int B, int n, int C) {
    int i = blockIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < B * C; idx += BLOCK_SIZE) {
        int b = idx / C;
        int c = idx % C;
        int src_idx = b * n * C + i * C + c;
        sum += grad[idx] * inp[src_idx];
    }

    __shared__ float s_partial[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 32; w++) {
            total_sum += s_partial[w];
        }
        d_H_pre[i] = total_sum;
    }
}

inline void stream_aggregate_backward(float* d_inp, float* d_H_pre, const float* grad,
                                      const float* inp, const float* H_pre, int B, int n, int C,
                                      cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int MAX_N = 8;
    stream_aggregate_backward_dx_kernel<BLOCK_SIZE, MAX_N>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, grad, H_pre, B, n, C);

    stream_aggregate_backward_dH_kernel<BLOCK_SIZE, MAX_N>
        <<<n, BLOCK_SIZE, 0, stream>>>(d_H_pre, grad, inp, B, n, C);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_distribute_backward_dx_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                                     const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n) {
        s_H_post[threadIdx.x] = H_post[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * C;

    if (idx >= total)
        return;

    int b = idx / C;
    int c = idx % C;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            int grad_idx = b * n * C + i * C + c;
            sum += grad[grad_idx] * s_H_post[i];
        }
    }

    d_inp[idx] = sum;
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void
stream_distribute_backward_dH_kernel(float* __restrict__ d_H_post, const float* __restrict__ grad,
                                     const float* __restrict__ inp, int B, int n, int C) {
    int i = blockIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < B * C; idx += BLOCK_SIZE) {
        int b = idx / C;
        int c = idx % C;
        int grad_idx = b * n * C + i * C + c;
        float inp_val = inp[b * C + c];
        sum += grad[grad_idx] * inp_val;
    }

    __shared__ float s_partial[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 32; w++) {
            total_sum += s_partial[w];
        }
        d_H_post[i] = total_sum;
    }
}

inline void stream_distribute_backward(float* d_inp, float* d_H_post, const float* grad,
                                       const float* inp, const float* H_post, int B, int n, int C,
                                       cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    int total = B * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int MAX_N = 8;
    stream_distribute_backward_dx_kernel<BLOCK_SIZE, MAX_N>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, grad, H_post, B, n, C);

    stream_distribute_backward_dH_kernel<BLOCK_SIZE, MAX_N>
        <<<n, BLOCK_SIZE, 0, stream>>>(d_H_post, grad, inp, B, n, C);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_mix_backward_dx_kernel(float* __restrict__ d_inp,
                                              const float* __restrict__ grad,
                                              const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];

    if (threadIdx.x < n * n) {
        s_M[threadIdx.x] = M[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = B * n * C;

    if (idx >= total)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int j = remainder / C;
    int c = remainder % C;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            int grad_idx = b * n * C + i * C + c;
            sum += s_M[i * n + j] * grad[grad_idx];
        }
    }

    d_inp[idx] = sum;
}

template<int BLOCK_SIZE>
__global__ void stream_mix_backward_dM_kernel(float* __restrict__ d_M,
                                              const float* __restrict__ grad,
                                              const float* __restrict__ inp, int B, int n, int C) {
    int ij = blockIdx.x;
    if (ij >= n * n)
        return;

    int i = ij / n;
    int j = ij % n;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < B * C; idx += BLOCK_SIZE) {
        int b = idx / C;
        int c = idx % C;
        int grad_idx = b * n * C + i * C + c;
        int inp_idx = b * n * C + j * C + c;
        sum += grad[grad_idx] * inp[inp_idx];
    }

    __shared__ float s_partial[BLOCK_SIZE / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        s_partial[warp_id] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 32; w++) {
            total_sum += s_partial[w];
        }
        d_M[ij] = total_sum;
    }
}

inline void stream_mix_backward(float* d_inp, float* d_M, const float* grad, const float* inp,
                                const float* M, int B, int n, int C,
                                cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    int total = B * n * C;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int MAX_N = 8;
    stream_mix_backward_dx_kernel<BLOCK_SIZE, MAX_N>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_inp, grad, M, B, n, C);

    stream_mix_backward_dM_kernel<BLOCK_SIZE>
        <<<n * n, BLOCK_SIZE, 0, stream>>>(d_M, grad, inp, B, n, C);
}

} // namespace mhc
