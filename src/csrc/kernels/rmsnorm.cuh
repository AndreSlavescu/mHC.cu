#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "../include/profiling.cuh"

namespace cg = cooperative_groups;

namespace mhc {

template<int BLOCK_SIZE, bool OUTPUT_RMS = false, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void rmsnorm_kernel(
    float* __restrict__ out, float* __restrict__ rms_out, const float* __restrict__ inp,
    const floatX* __restrict__ weight, int N, int C, float eps, int64_t* profiler_buf = nullptr,
    int max_entries = 0) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    MHC_PROFILE_INIT(idx);
    MHC_PROFILE_START(TagLoad);

    const float* x = inp + idx * C;
    float* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = x[i];
        thread_sum_sq += val * val;
    }

    MHC_PROFILE_PHASE(TagReduce);

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagStore);

    float rms_inv = s_sum_sq[0];

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = x[i];
        float w = (float)weight[i];
        o[i] = val * rms_inv * w;
    }

    MHC_PROFILE_END();
}

template<int BLOCK_SIZE, bool OUTPUT_RMS = false, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void rmsnorm_kernel_vectorized(
    float* __restrict__ out, float* __restrict__ rms_out, const float* __restrict__ inp,
    const floatX* __restrict__ weight, int N, int C, float eps, int64_t* profiler_buf = nullptr,
    int max_entries = 0) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    MHC_PROFILE_INIT(idx);
    MHC_PROFILE_START(TagLoad);

    const float* x = inp + idx * C;
    float* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    constexpr int VEC_SIZE = 4;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        float4 v = x_vec[i];
        thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    int remainder_start = C_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = x[i];
        thread_sum_sq += val * val;
    }

    MHC_PROFILE_PHASE(TagReduce);

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if constexpr (OUTPUT_RMS) {
                rms_out[idx] = rms;
            }
        }
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagStore);

    float rms_inv = s_sum_sq[0];

    // Output as fp32: read fp32 input, bf16 weight, compute in fp32, write fp32
    // Weight is bf16: load as float4 (8 bf16 = 16 bytes), process 4 elements per half
    constexpr int W_VEC_SIZE = 8;
    int C_w_vec = C / W_VEC_SIZE;
    float4* o_vec = reinterpret_cast<float4*>(o);
    using w_vec_t = float4;
    const w_vec_t* w_vec = reinterpret_cast<const w_vec_t*>(weight);

    for (int i = threadIdx.x; i < C_w_vec; i += BLOCK_SIZE) {
        // Input: 2 float4s = 8 floats
        float4 xv0 = x_vec[i * 2 + 0];
        float4 xv1 = x_vec[i * 2 + 1];
        // Weight: 1 float4 = 8 bf16 values
        w_vec_t wv = w_vec[i];
        nv_bfloat162* bf_w = reinterpret_cast<nv_bfloat162*>(&wv);

        float2 wf0 = __bfloat1622float2(bf_w[0]);
        float2 wf1 = __bfloat1622float2(bf_w[1]);
        float2 wf2 = __bfloat1622float2(bf_w[2]);
        float2 wf3 = __bfloat1622float2(bf_w[3]);

        float4 ov0 = {xv0.x * rms_inv * wf0.x, xv0.y * rms_inv * wf0.y, xv0.z * rms_inv * wf1.x,
                      xv0.w * rms_inv * wf1.y};
        float4 ov1 = {xv1.x * rms_inv * wf2.x, xv1.y * rms_inv * wf2.y, xv1.z * rms_inv * wf3.x,
                      xv1.w * rms_inv * wf3.y};
        o_vec[i * 2 + 0] = ov0;
        o_vec[i * 2 + 1] = ov1;
    }

    int vec_remainder_start = C_w_vec * W_VEC_SIZE;
    for (int i = vec_remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = x[i];
        float w = (float)weight[i];
        o[i] = val * rms_inv * w;
    }

    MHC_PROFILE_END();
}

// @bench rmsnorm
// @title: RMSNorm
// @configs: (N,C) =
// [(128,4096),(256,4096),(512,4096),(1024,4096),(2048,4096),(1024,8192),(2048,8192)]
// @in: inp float[N * C] random(-1,1), weight floatX[C] bf16(0.75,1.25)
// @out: out float[N * C]
// @scalar: eps=1e-5f
// @bandwidth: (size_t)N * C * sizeof(float) + (size_t)C * sizeof(floatX) + (size_t)N * C *
// sizeof(float)
// @profile: rmsnorm_forward_profiled N=1024 C=4096 grid=N
inline void rmsnorm_forward(float* out, const float* inp, const floatX* weight, int N, int C,
                            float eps, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, false>
            <<<grid, block, shared_mem, stream>>>(out, nullptr, inp, weight, N, C, eps);
    }
}

inline void rmsnorm_forward_with_rms(float* out, float* rms_out, const float* inp,
                                     const floatX* weight, int N, int C, float eps,
                                     cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = block;
    config.gridDim = grid;
    config.dynamicSmemBytes = shared_mem;
    config.stream = stream;

    if (C % 8 == 0 && C >= 64) {
        cudaLaunchKernelEx(&config, rmsnorm_kernel_vectorized<BLOCK_SIZE, true>, out, rms_out, inp,
                           weight, N, C, eps, (int64_t*)nullptr, 0);
    } else {
        cudaLaunchKernelEx(&config, rmsnorm_kernel<BLOCK_SIZE, true>, out, rms_out, inp, weight, N,
                           C, eps, (int64_t*)nullptr, 0);
    }
#else
    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, true>
            <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
    }
#endif
}
template<int BLOCK_SIZE, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void rmsnorm_backward_kernel(
    float* __restrict__ d_inp, float* __restrict__ d_weight, const float* __restrict__ grad,
    const float* __restrict__ inp, const floatX* __restrict__ weight, const float* __restrict__ rms,
    int N, int C, int64_t* profiler_buf = nullptr, int max_entries = 0) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    MHC_PROFILE_INIT(idx);
    MHC_PROFILE_START(TagLoad);

    const float* x = inp + idx * C;
    const float* g = grad + idx * C;
    float* dx = d_inp + idx * C;
    float r = rms[idx];
    float r_inv = 1.0f / r;

    extern __shared__ float shared[];
    float* s_reduce = shared;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = x[i];
        thread_dot += g_val * w_val * x_val;
    }

    MHC_PROFILE_PHASE(TagReduce);

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
        float block_dot = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            s_reduce[0] = block_dot;
        }
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagStore);

    float dot_sum = s_reduce[0];
    float correction = dot_sum / ((float)C * r * r);

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float g_val = g[i];
        float w_val = (float)weight[i];
        float x_val = x[i];

        dx[i] = (g_val * w_val * r_inv) - (x_val * correction * r_inv);

        atomicAdd(&d_weight[i], g_val * x_val * r_inv);
    }

    MHC_PROFILE_END();
}

// @bench rmsnorm_backward
// @title: RMSNorm Backward
// @configs: (N,C) =
// [(128,4096),(256,4096),(512,4096),(1024,4096),(2048,4096),(1024,8192),(2048,8192)]
// @in: grad float[N * C] random(-1,1,43), inp float[N * C] random(-1,1), weight floatX[C]
// bf16(0.75,1.25), rms float[N] computed
// @out: d_inp float[N * C], d_weight float[C]
// @setup: for (int i = 0; i < N; i++) {
// @setup:     float sum_sq = 0.0f;
// @setup:     for (int j = 0; j < C; j++) {
// @setup:         float v = h_inp.ptr[i * C + j];
// @setup:         sum_sq += v * v;
// @setup:     }
// @setup:     h_rms.ptr[i] = sqrtf(sum_sq / (float)C + 1e-5f);
// @setup: }
// @setup: d_rms.upload(h_rms);
// @pre-iter: d_d_weight.zero()
// @bandwidth: (size_t)N * C * sizeof(float) + (size_t)C * sizeof(floatX) + (size_t)N * C *
// sizeof(float) + (size_t)N * sizeof(float) + (size_t)N * C * sizeof(float) + (size_t)C *
// sizeof(float)
// @profile: rmsnorm_backward_profiled N=1024 C=4096 grid=N
inline void rmsnorm_backward(float* d_inp, float* d_weight, const float* grad, const float* inp,
                             const floatX* weight, const float* rms, int N, int C,
                             cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    rmsnorm_backward_kernel<BLOCK_SIZE>
        <<<N, BLOCK_SIZE, shared_mem, stream>>>(d_inp, d_weight, grad, inp, weight, rms, N, C);
}

// Profiled dispatch variants — call production kernels with DO_PROFILE=true.

inline void rmsnorm_forward_profiled(float* out, const float* inp, const floatX* weight, int N,
                                     int C, float eps, int64_t* profiler_buf, int max_entries,
                                     cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

    if (C % 8 == 0 && C >= 64) {
        rmsnorm_kernel_vectorized<BLOCK_SIZE, false, true><<<grid, block, shared_mem, stream>>>(
            out, nullptr, inp, weight, N, C, eps, profiler_buf, max_entries);
    } else {
        rmsnorm_kernel<BLOCK_SIZE, false, true><<<grid, block, shared_mem, stream>>>(
            out, nullptr, inp, weight, N, C, eps, profiler_buf, max_entries);
    }
}

inline void rmsnorm_backward_profiled(float* d_inp, float* d_weight, const float* grad,
                                      const float* inp, const floatX* weight, const float* rms,
                                      int N, int C, int64_t* profiler_buf, int max_entries,
                                      cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    rmsnorm_backward_kernel<BLOCK_SIZE, true><<<N, BLOCK_SIZE, shared_mem, stream>>>(
        d_inp, d_weight, grad, inp, weight, rms, N, C, profiler_buf, max_entries);
}

} // namespace mhc
