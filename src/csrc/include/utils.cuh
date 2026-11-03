#pragma once

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void float_to_bf16_kernel(floatX* __restrict__ out,
                                                                      const float* __restrict__ inp,
                                                                      int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (floatX)inp[idx];
    }
}

inline void float_to_bf16(floatX* out, const float* inp, int size, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}


__device__ __forceinline__ float fast_exp(float x) {
    constexpr float kExpClamp = 20.0f;
    x = fminf(x, kExpClamp);
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return __frcp_rn(1.0f + fast_exp(-x));
}

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void fused_h_activations_kernel(
    float* __restrict__ H_pre_out, float* __restrict__ H_post_out, float* __restrict__ H_res_out,
    const float* __restrict__ H_proj_concat, const float* __restrict__ rms, float alpha_pre,
    float alpha_post, float alpha_res, const float* __restrict__ b_pre,
    const float* __restrict__ b_post, const float* __restrict__ b_res, int B, int n) {
    int n_sq = n * n;
    int total_pre = B * n;
    int total_post = B * n;
    int total_res = B * n_sq;
    int stride = n + n + n_sq;

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < total_pre) {
        int b = idx / n;
        int j = idx % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + j];
        val = alpha_pre * val * r_inv + b_pre[j];
        H_pre_out[idx] = fast_sigmoid(val);
    }

    int idx2 = idx;
    if (idx2 < total_post) {
        int b = idx2 / n;
        int j = idx2 % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + j];
        val = alpha_post * val * r_inv + b_post[j];
        H_post_out[idx2] = 2.0f * fast_sigmoid(val);
    }

    int idx3 = idx;
    if (idx3 < total_res) {
        int b = idx3 / n_sq;
        int local = idx3 % n_sq;
        int i = local / n;
        int j = local % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + n + local];
        val = alpha_res * val * r_inv + b_res[i * n + j];
        H_res_out[idx3] = fast_exp(val);
    }
}

// Fused DynamicH backward pre-sinkhorn kernel.
// Computes sigmoid derivatives for d_tilde_pre/post and recomputes H_res_exp for sinkhorn backward.
// Grid: [B], Block: [32] (one warp per batch — only n+n+n²=24 elements per batch for n=4).
template<int BLOCK_SIZE, int MAX_N>
__global__ __launch_bounds__(BLOCK_SIZE, 4) void fused_h_backward_pre_kernel(
    float* __restrict__ d_tilde_pre,                  // [B, n]
    float* __restrict__ d_tilde_post,                 // [B, n]
    float* __restrict__ H_res_exp,                    // [B, n, n]
    const float* __restrict__ d_H_pre,                // [B, n]
    const float* __restrict__ d_H_post,               // [B, n]
    const float* __restrict__ H_pre_act,              // [B, n] — sigmoid values
    const float* __restrict__ H_post_act,             // [B, n] — 2*sigmoid values
    const float* __restrict__ p_concat,               // [B, n+n+n²]
    const float* __restrict__ rms_inv,                // [B]
    float alpha_res, const float* __restrict__ b_res, // [n, n]
    int B, int n) {
    int b = blockIdx.x;
    if (b >= B)
        return;

    int tid = threadIdx.x;
    int n_sq = n * n;
    int stride = n + n + n_sq;
    float r_inv = rms_inv[b];

    // Threads 0..n-1: sigmoid derivative for pre
    if (tid < n) {
        float h = H_pre_act[b * n + tid];
        float dh = d_H_pre[b * n + tid];
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        d_tilde_pre[b * n + tid] = dh * h * (1.0f - h);
    }

    // Threads n..2n-1: sigmoid derivative for post (H_post = 2*sigmoid, so derivative differs)
    if (tid >= n && tid < 2 * n) {
        int j = tid - n;
        float h = H_post_act[b * n + j]; // h = 2*sigmoid(x)
        float dh = d_H_post[b * n + j];
        // d/dx[2*sigmoid(x)] = 2*sigmoid(x)*(1-sigmoid(x)) = h*(1-h/2)
        d_tilde_post[b * n + j] = dh * h * (1.0f - h * 0.5f);
    }

    // Threads 2n..2n+n²-1: forward recompute tilde_res → H_res_exp
    if (tid >= 2 * n && tid < stride) {
        int local = tid - 2 * n;
        int ii = local / n;
        int jj = local % n;
        float p_val = p_concat[b * stride + 2 * n + local];
        float tilde = alpha_res * p_val * r_inv + b_res[ii * n + jj];
        tilde = fminf(tilde, 20.0f); // clamp
        H_res_exp[b * n_sq + local] = __expf(tilde);
    }
}

// Fused DynamicH backward post-sinkhorn kernel.
// Computes d_tilde_res, all parameter gradients, d_p_concat, and d_r.
// Grid: [B], Block: [32] (one warp per batch).
template<int BLOCK_SIZE, int MAX_N>
__global__ __launch_bounds__(BLOCK_SIZE, 4) void fused_h_backward_post_kernel(
    float* __restrict__ d_p_pre,            // [B, n]
    float* __restrict__ d_p_post,           // [B, n]
    float* __restrict__ d_p_res_flat,       // [B, n²]
    float* __restrict__ d_b_pre,            // [n] — atomicAdd target
    float* __restrict__ d_b_post,           // [n] — atomicAdd target
    float* __restrict__ d_b_res,            // [n, n] — atomicAdd target
    float* __restrict__ d_alpha_pre,        // scalar — atomicAdd target
    float* __restrict__ d_alpha_post,       // scalar — atomicAdd target
    float* __restrict__ d_alpha_res,        // scalar — atomicAdd target
    float* __restrict__ d_r,                // [B]
    const float* __restrict__ d_H_res_exp,  // [B, n, n] — from sinkhorn backward
    const float* __restrict__ H_res_exp,    // [B, n, n]
    const float* __restrict__ d_tilde_pre,  // [B, n]
    const float* __restrict__ d_tilde_post, // [B, n]
    const float* __restrict__ p_concat,     // [B, n+n+n²]
    const float* __restrict__ rms_inv,      // [B]
    float alpha_pre, float alpha_post, float alpha_res_f, int B, int n) {
    int b = blockIdx.x;
    if (b >= B)
        return;

    int tid = threadIdx.x;
    int n_sq = n * n;
    int stride = n + n + n_sq;
    float r_inv = rms_inv[b];
    float r_inv2 = r_inv * r_inv;

    // Each thread computes one element and accumulates d_r contribution
    float my_d_r = 0.0f;

    // Threads 0..n-1: process pre
    if (tid < n) {
        float dt = d_tilde_pre[b * n + tid];
        float p_val = p_concat[b * stride + tid];

        d_p_pre[b * n + tid] = dt * alpha_pre * r_inv;
        atomicAdd(&d_b_pre[tid], dt);
        atomicAdd(d_alpha_pre, dt * p_val * r_inv);
        my_d_r = -(dt * alpha_pre * p_val * r_inv2);
    }

    // Threads n..2n-1: process post
    if (tid >= n && tid < 2 * n) {
        int j = tid - n;
        float dt = d_tilde_post[b * n + j];
        float p_val = p_concat[b * stride + n + j];

        d_p_post[b * n + j] = dt * alpha_post * r_inv;
        atomicAdd(&d_b_post[j], dt);
        atomicAdd(d_alpha_post, dt * p_val * r_inv);
        my_d_r = -(dt * alpha_post * p_val * r_inv2);
    }

    // Threads 2n..2n+n²-1: process res
    if (tid >= 2 * n && tid < stride) {
        int local = tid - 2 * n;
        float d_hre = d_H_res_exp[b * n_sq + local];
        float hre = H_res_exp[b * n_sq + local];
        float dt = d_hre * hre; // d_tilde_res
        float p_val = p_concat[b * stride + 2 * n + local];

        d_p_res_flat[b * n_sq + local] = dt * alpha_res_f * r_inv;
        atomicAdd(&d_b_res[local], dt);
        atomicAdd(d_alpha_res, dt * p_val * r_inv);
        my_d_r = -(dt * alpha_res_f * p_val * r_inv2);
    }

    // Warp reduction for d_r[b]
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    float warp_d_r = cg::reduce(warp, my_d_r, cg::plus<float>());

    // For multi-warp blocks, need shared memory reduction
    if constexpr (BLOCK_SIZE > 32) {
        extern __shared__ float shared[];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int num_warps = BLOCK_SIZE / 32;

        if (lane_id == 0)
            shared[warp_id] = warp_d_r;
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
            float block_d_r = cg::reduce(warp, val, cg::plus<float>());
            if (lane_id == 0) {
                d_r[b] = block_d_r;
            }
        }
    } else {
        // Single warp — lane 0 writes directly
        if (tid == 0) {
            d_r[b] = warp_d_r;
        }
    }
}

// Fused d_r RMS correction kernel.
// Applies: d_x_flat[b*nC + c] += d_r[b] * x_flat[b*nC + c] * rms_inv[b] / nC
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void fused_rms_correction_kernel(
    float* __restrict__ d_x_flat,      // [B, nC] — modified in-place
    const float* __restrict__ d_r,     // [B]
    const float* __restrict__ x_flat,  // [B, nC]
    const float* __restrict__ rms_inv, // [B]
    int B, int nC) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * nC)
        return;
    int b = idx / nC;
    float scale = d_r[b] * rms_inv[b] / (float)nC;
    d_x_flat[idx] += scale * x_flat[idx];
}

inline void fused_h_backward_pre(float* d_tilde_pre, float* d_tilde_post, float* H_res_exp,
                                 const float* d_H_pre, const float* d_H_post,
                                 const float* H_pre_act, const float* H_post_act,
                                 const float* p_concat, const float* rms_inv, float alpha_res,
                                 const float* b_res, int B, int n, cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 32;
    int n_sq = n * n;
    int stride = n + n + n_sq;
    // Need at least stride threads per block; use 32 for warp alignment
    int block_size = ((stride + 31) / 32) * 32;
    // For typical n=4, stride=24, block_size=32. For n=8, stride=80, block_size=96.

    if (block_size <= 32) {
        fused_h_backward_pre_kernel<32, 4><<<B, 32, 0, stream>>>(
            d_tilde_pre, d_tilde_post, H_res_exp, d_H_pre, d_H_post, H_pre_act, H_post_act,
            p_concat, rms_inv, alpha_res, b_res, B, n);
    } else if (block_size <= 128) {
        fused_h_backward_pre_kernel<128, 8><<<B, 128, 0, stream>>>(
            d_tilde_pre, d_tilde_post, H_res_exp, d_H_pre, d_H_post, H_pre_act, H_post_act,
            p_concat, rms_inv, alpha_res, b_res, B, n);
    } else {
        fused_h_backward_pre_kernel<256, 16><<<B, 256, 0, stream>>>(
            d_tilde_pre, d_tilde_post, H_res_exp, d_H_pre, d_H_post, H_pre_act, H_post_act,
            p_concat, rms_inv, alpha_res, b_res, B, n);
    }
}

inline void fused_h_backward_post(float* d_p_pre, float* d_p_post, float* d_p_res_flat,
                                  float* d_b_pre, float* d_b_post, float* d_b_res,
                                  float* d_alpha_pre, float* d_alpha_post, float* d_alpha_res,
                                  float* d_r, const float* d_H_res_exp, const float* H_res_exp,
                                  const float* d_tilde_pre, const float* d_tilde_post,
                                  const float* p_concat, const float* rms_inv, float alpha_pre,
                                  float alpha_post, float alpha_res, int B, int n,
                                  cudaStream_t stream = nullptr) {
    int n_sq = n * n;
    int stride = n + n + n_sq;
    int block_size = ((stride + 31) / 32) * 32;

    if (block_size <= 32) {
        fused_h_backward_post_kernel<32, 4><<<B, 32, 0, stream>>>(
            d_p_pre, d_p_post, d_p_res_flat, d_b_pre, d_b_post, d_b_res, d_alpha_pre, d_alpha_post,
            d_alpha_res, d_r, d_H_res_exp, H_res_exp, d_tilde_pre, d_tilde_post, p_concat, rms_inv,
            alpha_pre, alpha_post, alpha_res, B, n);
    } else if (block_size <= 128) {
        size_t smem = (128 / 32) * sizeof(float);
        fused_h_backward_post_kernel<128, 8><<<B, 128, smem, stream>>>(
            d_p_pre, d_p_post, d_p_res_flat, d_b_pre, d_b_post, d_b_res, d_alpha_pre, d_alpha_post,
            d_alpha_res, d_r, d_H_res_exp, H_res_exp, d_tilde_pre, d_tilde_post, p_concat, rms_inv,
            alpha_pre, alpha_post, alpha_res, B, n);
    } else {
        size_t smem = (256 / 32) * sizeof(float);
        fused_h_backward_post_kernel<256, 16><<<B, 256, smem, stream>>>(
            d_p_pre, d_p_post, d_p_res_flat, d_b_pre, d_b_post, d_b_res, d_alpha_pre, d_alpha_post,
            d_alpha_res, d_r, d_H_res_exp, H_res_exp, d_tilde_pre, d_tilde_post, p_concat, rms_inv,
            alpha_pre, alpha_post, alpha_res, B, n);
    }
}

inline void fused_rms_correction(float* d_x_flat, const float* d_r, const float* x_flat,
                                 const float* rms_inv, int B, int nC,
                                 cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * nC + BLOCK - 1) / BLOCK;
    fused_rms_correction_kernel<BLOCK>
        <<<blocks, BLOCK, 0, stream>>>(d_x_flat, d_r, x_flat, rms_inv, B, nC);
}

inline void fused_h_activations(float* H_pre_out, float* H_post_out, float* H_res_out,
                                const float* H_proj_concat, const float* rms, float alpha_pre,
                                float alpha_post, float alpha_res, const float* b_pre,
                                const float* b_post, const float* b_res, int B, int n,
                                cudaStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int n_sq = n * n;
    int max_total = B * n_sq;
    int blocks = (max_total + BLOCK - 1) / BLOCK;

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

    cudaLaunchKernelEx(&config, fused_h_activations_kernel<BLOCK>, H_pre_out, H_post_out, H_res_out,
                       H_proj_concat, rms, alpha_pre, alpha_post, alpha_res, b_pre, b_post, b_res,
                       B, n);
#else
    fused_h_activations_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
        H_pre_out, H_post_out, H_res_out, H_proj_concat, rms, alpha_pre, alpha_post, alpha_res,
        b_pre, b_post, b_res, B, n);
#endif
}

__global__ __launch_bounds__(256, 4) void flush_l2_kernel(float* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = buf[idx] + 1.0f;
    }
}

struct L2Flusher {
    static constexpr int L2_SIZE_BYTES = 50 * 1024 * 1024;
    static constexpr int FLUSH_SIZE = L2_SIZE_BYTES / sizeof(float) * 2;
    float* buf;

    L2Flusher() : buf(nullptr) {
        cudaMalloc(&buf, FLUSH_SIZE * sizeof(float));
        cudaMemset(buf, 0, FLUSH_SIZE * sizeof(float));
    }

    ~L2Flusher() {
        if (buf)
            cudaFree(buf);
    }

    void flush() {
        int block_size = 256;
        int num_blocks = (FLUSH_SIZE + block_size - 1) / block_size;
        flush_l2_kernel<<<num_blocks, block_size>>>(buf, FLUSH_SIZE);
        cudaDeviceSynchronize();
    }
};

inline float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

inline bool check_test(float max_diff, float tolerance, const char* test_name = nullptr) {
    if (test_name) {
        printf("%s: ", test_name);
    }
    printf("max diff = %e, ", max_diff);
    if (max_diff < tolerance) {
        printf("PASSED (tol: %e)\n", tolerance);
        return true;
    } else {
        printf("FAILED (tol: %e)\n", tolerance);
        return false;
    }
}

struct BenchTimer {
    cudaEvent_t start, stop;

    BenchTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~BenchTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() { cudaEventRecord(start); }

    void record_stop() { cudaEventRecord(stop); }

    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

enum ProfilerTag {
    TagSetup = 0,
    TagLoad,
    TagCompute,
    TagReduce,
    TagStore,
    TagSync,
    TagOther,
    TagCount
};

inline const char* profiler_tag_name(ProfilerTag tag) {
    switch (tag) {
    case TagSetup:
        return "Setup";
    case TagLoad:
        return "Load";
    case TagCompute:
        return "Compute";
    case TagReduce:
        return "Reduce";
    case TagStore:
        return "Store";
    case TagSync:
        return "Sync";
    case TagOther:
        return "Other";
    default:
        return "Unknown";
    }
}

__device__ __forceinline__ int64_t globaltimer() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    // s_memrealtime exists on CDNA (gfx90a) but not RDNA (gfx1100); the portable
    // clang intrinsic lowers to the right free-running counter per arch. The
    // profiler is if-constexpr-disabled by default, so this only needs to
    // compile and emit a monotonic counter on every target.
    return (int64_t)__builtin_readcyclecounter();
#else
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
#endif
}

__device__ __forceinline__ int get_smid() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    return (int)__builtin_amdgcn_s_getreg(/*HW_REG_HW_ID, offset 0, size 32*/ 0x0000001e);
#else
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
#endif
}

struct DeviceProfiler {
    int64_t* data_ptr;
    int sm_id;
    int cnt;
    int max_entries;

    __device__ void init(int num_entries, int64_t* buffer, int block_id) {
        max_entries = num_entries;
        data_ptr = buffer + block_id * (1 + num_entries * 4);
        sm_id = get_smid();
        cnt = 0;
    }

    __device__ void start(ProfilerTag tag) {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 0] = sm_id;
        data_ptr[1 + cnt * 4 + 1] = tag;
        data_ptr[1 + cnt * 4 + 2] = globaltimer();
    }

    __device__ void stop() {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 3] = globaltimer() - data_ptr[1 + cnt * 4 + 2];
        cnt++;
    }

    __device__ void flush() { data_ptr[0] = cnt; }
};

struct ProfilerEntry {
    int sm_id;
    ProfilerTag tag;
    int64_t start_time;
    int64_t duration_ns;
};

struct HostProfiler {
    int64_t* d_buffer;
    int64_t* h_buffer;
    int num_blocks;
    int max_entries_per_block;
    size_t buffer_size;

    HostProfiler(int num_blocks_, int max_entries_per_block_)
        : num_blocks(num_blocks_), max_entries_per_block(max_entries_per_block_) {
        buffer_size = num_blocks * (1 + max_entries_per_block * 4) * sizeof(int64_t);
        cudaMalloc(&d_buffer, buffer_size);
        cudaMemset(d_buffer, 0, buffer_size);
        h_buffer = (int64_t*)malloc(buffer_size);
    }

    ~HostProfiler() {
        if (d_buffer)
            cudaFree(d_buffer);
        if (h_buffer)
            free(h_buffer);
    }

    int64_t* device_ptr() { return d_buffer; }

    void download() { cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost); }

    void print_summary() {
        download();

        int64_t tag_totals[TagCount] = {0};
        int tag_counts[TagCount] = {0};
        int64_t min_start = INT64_MAX;
        int64_t max_end = 0;

        int entry_stride = 1 + max_entries_per_block * 4;

        for (int b = 0; b < num_blocks; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            for (int e = 0; e < num_entries; e++) {
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t start = block_data[1 + e * 4 + 2];
                int64_t duration = block_data[1 + e * 4 + 3];

                if (tag < TagCount) {
                    tag_totals[tag] += duration;
                    tag_counts[tag]++;
                }

                if (start < min_start)
                    min_start = start;
                if (start + duration > max_end)
                    max_end = start + duration;
            }
        }

        int64_t wall_time = max_end - min_start;
        printf("\nProfiler Summary (%d blocks, %d max entries/block)\n", num_blocks,
               max_entries_per_block);
        printf("==================================================\n");
        printf("Total wall time: %.2f us\n\n", wall_time / 1000.0f);
        printf("%-10s %10s %10s %10s\n", "Phase", "Total(us)", "Count", "Avg(us)");
        printf("------------------------------------------\n");

        for (int t = 0; t < TagCount; t++) {
            if (tag_counts[t] > 0) {
                float total_us = tag_totals[t] / 1000.0f;
                float avg_us = total_us / tag_counts[t];
                printf("%-10s %10.2f %10d %10.2f\n", profiler_tag_name((ProfilerTag)t), total_us,
                       tag_counts[t], avg_us);
            }
        }
    }

    void print_timeline(int max_blocks = 4) {
        download();

        int entry_stride = 1 + max_entries_per_block * 4;
        int blocks_to_print = (max_blocks < num_blocks) ? max_blocks : num_blocks;

        printf("\nTimeline (first %d blocks)\n", blocks_to_print);
        printf("==========================\n");

        for (int b = 0; b < blocks_to_print; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            printf("\nBlock %d (%d events):\n", b, num_entries);

            for (int e = 0; e < num_entries; e++) {
                int sm_id = (int)block_data[1 + e * 4 + 0];
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t duration = block_data[1 + e * 4 + 3];

                printf("  SM%02d: %-10s %8.2f us\n", sm_id, profiler_tag_name((ProfilerTag)tag),
                       duration / 1000.0f);
            }
        }
    }
};

} // namespace mhc
