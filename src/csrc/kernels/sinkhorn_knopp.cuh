#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../include/profiling.cuh"

namespace cg = cooperative_groups;

namespace mhc {

template<int N_COMPILE, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_warp_optimized_kernel(
    float* __restrict__ out, const float* __restrict__ inp, int M, int N, int num_iters,
    float eps) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

    extern __shared__ float smem[];
    float* tile = smem;
    float* col_sums = smem + M * N_COMPILE;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = warp.thread_rank();

    int total_elems = M * N;
    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    block.sync();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = warp_id; r < M; r += WARPS_PER_BLOCK) {
            float val = (lane_id < N) ? tile[r * N + lane_id] : 0.0f;
            float row_sum = cg::reduce(warp, val, cg::plus<float>());

            if (lane_id < N && row_sum > eps) {
                tile[r * N + lane_id] = val * __frcp_rn(row_sum);
            }
        }
        block.sync();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        block.sync();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            tile[i] *= col_sums[c];
        }
        block.sync();
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_warp_per_row_32x32_kernel(
    float* __restrict__ out, const float* __restrict__ inp, int num_iters, float eps) {
    constexpr int N = 32;
    constexpr int WARPS = BLOCK_SIZE / 32;
    constexpr int ROWS_PER_WARP = (N + WARPS - 1) / WARPS;

    __shared__ float tile[N * (N + 1)];
    __shared__ float col_sums[N];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = warp.thread_rank();
    const int stride = N + 1;

    for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
        int r = i / N;
        int c = i % N;
        tile[r * stride + c] = inp[i];
    }
    block.sync();

    for (int iter = 0; iter < num_iters; iter++) {
#pragma unroll 4
        for (int rr = 0; rr < ROWS_PER_WARP; rr++) {
            int r = warp_id * ROWS_PER_WARP + rr;
            if (r < N) {
                float val = tile[r * stride + lane_id];
                float sum = cg::reduce(warp, val, cg::plus<float>());

                if (sum > eps) {
                    tile[r * stride + lane_id] = val * __frcp_rn(sum);
                }
            }
        }
        block.sync();

        if (threadIdx.x < N) {
            int c = threadIdx.x;
            float sum = 0.0f;
#pragma unroll 8
            for (int r = 0; r < N; r++) {
                sum += tile[r * stride + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        block.sync();

        for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
            int r = i / N;
            int c = i % N;
            tile[r * stride + c] *= col_sums[c];
        }
        block.sync();
    }

    for (int i = threadIdx.x; i < N * N; i += BLOCK_SIZE) {
        int r = i / N;
        int c = i % N;
        out[i] = tile[r * stride + c];
    }
}

template<int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE,
                             2) void sinkhorn_knopp_kernel(float* __restrict__ out,
                                                           const float* __restrict__ inp, int M,
                                                           int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + TILE_M * TILE_N;
    float* col_sums = row_sums + TILE_M;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int rows_in_tile = min(TILE_M, M - tile_row);
    int cols_in_tile = min(TILE_N, N - tile_col);

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            tile[i] = inp[global_r * N + global_c];
        } else {
            tile[i] = 0.0f;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < TILE_M; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < TILE_N; c++) {
                sum += tile[r * TILE_N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int r = i / TILE_N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < TILE_N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < TILE_M; r++) {
                sum += tile[r * TILE_N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int c = i % TILE_N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            out[global_r * N + global_c] = tile[i];
        }
    }
}

template<int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE,
                             2) void sinkhorn_knopp_pdl_kernel(float* __restrict__ out,
                                                               const float* __restrict__ inp, int M,
                                                               int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + TILE_M * TILE_N;
    float* col_sums = row_sums + TILE_M;

    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    int rows_in_tile = min(TILE_M, M - tile_row);
    int cols_in_tile = min(TILE_N, N - tile_col);

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            tile[i] = inp[global_r * N + global_c];
        } else {
            tile[i] = 0.0f;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < TILE_M; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < TILE_N; c++) {
                sum += tile[r * TILE_N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int r = i / TILE_N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < TILE_N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < TILE_M; r++) {
                sum += tile[r * TILE_N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
            int c = i % TILE_N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += BLOCK_SIZE) {
        int local_r = i / TILE_N;
        int local_c = i % TILE_N;
        int global_r = tile_row + local_r;
        int global_c = tile_col + local_c;

        if (global_r < M && global_c < N) {
            out[global_r * N + global_c] = tile[i];
        }
    }
}

template<int MAX_DIM, int BLOCK_SIZE, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_single_block_kernel(
    float* __restrict__ out, const float* __restrict__ inp, int M, int N, int num_iters, float eps,
    int64_t* profiler_buf = nullptr, int max_entries = 0) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    MHC_PROFILE_INIT(blockIdx.x);
    MHC_PROFILE_START(TagLoad);

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagCompute);

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int r = i / N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

    MHC_PROFILE_PHASE(TagStore);

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }

    MHC_PROFILE_END();
}

template<int MAX_DIM, int BLOCK_SIZE, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_single_block_fused_exp_kernel(
    float* __restrict__ out, float* __restrict__ H_res_exp, const float* __restrict__ inp, int M,
    int N, int num_iters, float eps, int64_t* profiler_buf = nullptr, int max_entries = 0) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    MHC_PROFILE_INIT(blockIdx.x);
    MHC_PROFILE_START(TagLoad);

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        float val = fast_exp(inp[i]);
        tile[i] = val;
        if (H_res_exp)
            H_res_exp[i] = val;
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagCompute);

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int r = i / N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] *= __frcp_rn(row_sum);
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] *= __frcp_rn(col_sum);
            }
        }
        __syncthreads();
    }

    MHC_PROFILE_PHASE(TagStore);

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }

    MHC_PROFILE_END();
}

// @bench sinkhorn_knopp_forward
// @group: sinkhorn_knopp
// @title: Sinkhorn-Knopp Forward
// @configs: (M,N,num_iters) =
// [(32,32,5),(32,32,10),(32,32,20),(64,64,5),(64,64,10),(64,64,20),(128,128,5),(128,128,10),(128,128,20)]
// @in: inp float[M * N] random(0.1,1.1)
// @out: out float[M * N]
// @scalar: eps=1e-8f
// @flops: 4.0 * M * N * num_iters
// @profile: sinkhorn_knopp_forward_profiled M=64 N=64 num_iters=10 grid=1 timeline=true
// timeline_blocks=1
inline void sinkhorn_knopp_forward(float* out, const float* inp, int M, int N, int num_iters,
                                   float eps, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M == 32 && N == 32) {
        sinkhorn_knopp_warp_per_row_32x32_kernel<BLOCK_SIZE>
            <<<1, BLOCK_SIZE, 0, stream>>>(out, inp, num_iters, eps);
    } else if (N <= 32 && M <= 64) {
        size_t smem_size = M * 32 * sizeof(float) + 32 * sizeof(float);
        sinkhorn_knopp_warp_optimized_kernel<32, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps, nullptr, 0);
    } else {
        constexpr int TILE_SIZE = 32;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        size_t smem_size = TILE_SIZE * TILE_SIZE * sizeof(float) + TILE_SIZE * sizeof(float) +
                           TILE_SIZE * sizeof(float);

        sinkhorn_knopp_kernel<TILE_SIZE, TILE_SIZE, BLOCK_SIZE>
            <<<grid, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    }
}

// @bench sinkhorn_knopp_forward_fused_exp
// @group: sinkhorn_knopp
// @title: Fused Exp Version (as used in MHCLayer)
// @configs: (M,N,num_iters) =
// [(32,32,5),(32,32,10),(32,32,20),(64,64,5),(64,64,10),(64,64,20),(128,128,5),(128,128,10),(128,128,20)]
// @in: inp float[M * N] random(-0.01,0.01)
// @out: out float[M * N]
// @extra-buf: H_res_exp float[M * N]
// @scalar: eps=1e-8f
// @profile: sinkhorn_knopp_forward_fused_exp_profiled M=64 N=64 num_iters=10 grid=1 timeline=true
// timeline_blocks=1
inline void sinkhorn_knopp_forward_fused_exp(float* out, float* H_res_exp, const float* inp, int M,
                                             int N, int num_iters, float eps,
                                             cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK_SIZE, 1, 1};
        config.gridDim = {1, 1, 1};
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        cudaLaunchKernelEx(&config,
                           sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>, out,
                           H_res_exp, inp, M, N, num_iters, eps, (int64_t*)nullptr, 0);
#else
        sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps);
#endif
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {BLOCK_SIZE, 1, 1};
        config.gridDim = {1, 1, 1};
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        cudaLaunchKernelEx(&config, kernel, out, H_res_exp, inp, M, N, num_iters, eps,
                           (int64_t*)nullptr, 0);
#else
        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps,
                                                     nullptr, 0);
#endif
    } else {
        fprintf(stderr, "sinkhorn_knopp_forward_fused_exp: M > 128 or N > 128 not supported\n");
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_single_block_pdl_kernel(
    float* __restrict__ out, const float* __restrict__ inp, int M, int N, int num_iters,
    float eps) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int r = i / N;
            float row_sum = row_sums[r];
            if (row_sum > eps) {
                tile[i] /= row_sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                sum += tile[r * N + c];
            }
            col_sums[c] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
            int c = i % N;
            float col_sum = col_sums[c];
            if (col_sum > eps) {
                tile[i] /= col_sum;
            }
        }
        __syncthreads();
    }

#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

template<int N_COMPILE, int MAX_ITERS, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_backward_checkpointed_kernel(
    float* __restrict__ d_inp, const float* __restrict__ grad, const float* __restrict__ M_inp,
    int N, int num_iters, float eps) {
    extern __shared__ float smem[];

    float* checkpoints = smem;
    float* d_tile = checkpoints + MAX_ITERS * N_COMPILE * N_COMPILE;
    float* row_buffer = d_tile + N_COMPILE * N_COMPILE;
    float* col_buffer = row_buffer + N_COMPILE;
    float* tile_work = col_buffer + N_COMPILE;

    int total = N * N;

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        tile_work[i] = M_inp[i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < N; c++) {
                sum += tile_work[r * N + c];
            }
            row_buffer[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            tile_work[i] *= row_buffer[r];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            checkpoints[iter * N * N + i] = tile_work[i];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < N; r++) {
                sum += tile_work[r * N + c];
            }
            col_buffer[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            tile_work[i] *= col_buffer[c];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_tile[i] = grad[i];
    }
    __syncthreads();

    for (int iter = num_iters - 1; iter >= 0; iter--) {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile_work[i] = checkpoints[iter * N * N + i];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int r = 0; r < N; r++) {
                dot += d_tile[r * N + c] * tile_work[r * N + c];
            }
            col_buffer[c] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            d_tile[i] = d_tile[i] - tile_work[i] * col_buffer[c];
        }
        __syncthreads();

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int c = 0; c < N; c++) {
                dot += d_tile[r * N + c] * tile_work[r * N + c];
            }
            row_buffer[r] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            d_tile[i] = d_tile[i] - tile_work[i] * row_buffer[r];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_inp[i] = d_tile[i];
    }
}

template<int MAX_DIM, int BLOCK_SIZE, bool DO_PROFILE = false>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void sinkhorn_knopp_backward_kernel(
    float* __restrict__ d_inp, const float* __restrict__ grad, const float* __restrict__ M_out,
    const float* __restrict__ M_inp, int N, int num_iters, float eps,
    int64_t* profiler_buf = nullptr, int max_entries = 0) {
    extern __shared__ float smem[];
    float* d_tile = smem;
    float* row_buffer = smem + MAX_DIM * MAX_DIM;
    float* col_buffer = row_buffer + MAX_DIM;
    float* tile_fwd = col_buffer + MAX_DIM;
    float* row_sums = tile_fwd + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total = N * N;

    MHC_PROFILE_INIT(blockIdx.x);
    MHC_PROFILE_START(TagLoad);

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_tile[i] = grad[i];
    }
    __syncthreads();

    MHC_PROFILE_PHASE(TagCompute);

    for (int iter = num_iters - 1; iter >= 0; iter--) {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile_fwd[i] = M_inp[i];
        }
        __syncthreads();

        for (int fwd_iter = 0; fwd_iter < iter; fwd_iter++) {
            for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
                float sum = 0.0f;
                for (int c = 0; c < N; c++) {
                    sum += tile_fwd[r * N + c];
                }
                row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int r = i / N;
                tile_fwd[i] *= row_sums[r];
            }
            __syncthreads();

            for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
                float sum = 0.0f;
                for (int r = 0; r < N; r++) {
                    sum += tile_fwd[r * N + c];
                }
                col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int c = i % N;
                tile_fwd[i] *= col_sums[c];
            }
            __syncthreads();
        }

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile_fwd[r * N + c];
            }
            row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            tile_fwd[i] *= row_sums[r];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int r = 0; r < N; r++) {
                dot += d_tile[r * N + c] * tile_fwd[r * N + c];
            }
            col_buffer[c] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % N;
            d_tile[i] = d_tile[i] - tile_fwd[i] * col_buffer[c];
        }
        __syncthreads();

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float dot = 0.0f;
            for (int c = 0; c < N; c++) {
                dot += d_tile[r * N + c] * tile_fwd[r * N + c];
            }
            row_buffer[r] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            d_tile[i] = d_tile[i] - tile_fwd[i] * row_buffer[r];
        }
        __syncthreads();
    }

    MHC_PROFILE_PHASE(TagStore);

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_inp[i] = d_tile[i];
    }

    MHC_PROFILE_END();
}

// @bench sinkhorn_knopp_backward
// @title: Sinkhorn-Knopp Backward
// @configs: (M,N,num_iters) = [(32,32,5),(32,32,10),(32,32,20),(64,64,5),(64,64,10),(64,64,20)]
// @in: M_inp float[M * N] random(0.1,1.1), grad float[M * N] random(-1,1,43)
// @out: d_inp float[M * N]
// @extra-buf: M_out float[M * N]
// @scalar: eps=1e-8f
// @flops: 8.0 * M * N * num_iters
// @bandwidth: 4 * M * N * sizeof(float)
// @setup: sinkhorn_knopp_forward(d_M_out, d_M_inp, M, N, num_iters, eps);
// @setup: CHECK_CUDA(cudaDeviceSynchronize());
// @profile: sinkhorn_knopp_backward_profiled M=64 N=64 num_iters=10 grid=1 timeline=true
// timeline_blocks=1
inline void sinkhorn_knopp_backward(float* d_inp, const float* grad, const float* M_out,
                                    const float* M_inp, int N, int num_iters, float eps,
                                    cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N <= 32 && num_iters <= 20) {
        constexpr int N_COMPILE = 32;
        constexpr int MAX_ITERS = 20;
        size_t smem_size =
            (MAX_ITERS + 3) * N_COMPILE * N_COMPILE * sizeof(float) + 2 * N_COMPILE * sizeof(float);

        auto kernel = sinkhorn_knopp_backward_checkpointed_kernel<N_COMPILE, MAX_ITERS, BLOCK_SIZE>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_inp, N, num_iters, eps);
    } else if (N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = 2 * MAX_DIM * MAX_DIM * sizeof(float) + 4 * MAX_DIM * sizeof(float);

        sinkhorn_knopp_backward_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_out, M_inp, N, num_iters, eps);
    } else {
        fprintf(stderr, "sinkhorn_knopp_backward: N > 64 not supported\n");
    }
}

// Row-normalize a 4x4 matrix stored as 4 float4 rows
__device__ __forceinline__ void sk4_row_normalize(float4& r0, float4& r1, float4& r2, float4& r3,
                                                  float eps) {
    float s0 = r0.x + r0.y + r0.z + r0.w;
    float s1 = r1.x + r1.y + r1.z + r1.w;
    float s2 = r2.x + r2.y + r2.z + r2.w;
    float s3 = r3.x + r3.y + r3.z + r3.w;
    float inv0 = (s0 > eps) ? __frcp_rn(s0) : 0.0f;
    float inv1 = (s1 > eps) ? __frcp_rn(s1) : 0.0f;
    float inv2 = (s2 > eps) ? __frcp_rn(s2) : 0.0f;
    float inv3 = (s3 > eps) ? __frcp_rn(s3) : 0.0f;
    r0.x *= inv0;
    r0.y *= inv0;
    r0.z *= inv0;
    r0.w *= inv0;
    r1.x *= inv1;
    r1.y *= inv1;
    r1.z *= inv1;
    r1.w *= inv1;
    r2.x *= inv2;
    r2.y *= inv2;
    r2.z *= inv2;
    r2.w *= inv2;
    r3.x *= inv3;
    r3.y *= inv3;
    r3.z *= inv3;
    r3.w *= inv3;
}

// Column-normalize a 4x4 matrix stored as 4 float4 rows
__device__ __forceinline__ void sk4_col_normalize(float4& r0, float4& r1, float4& r2, float4& r3,
                                                  float eps) {
    float c0 = r0.x + r1.x + r2.x + r3.x;
    float c1 = r0.y + r1.y + r2.y + r3.y;
    float c2 = r0.z + r1.z + r2.z + r3.z;
    float c3 = r0.w + r1.w + r2.w + r3.w;
    float cinv0 = (c0 > eps) ? __frcp_rn(c0) : 0.0f;
    float cinv1 = (c1 > eps) ? __frcp_rn(c1) : 0.0f;
    float cinv2 = (c2 > eps) ? __frcp_rn(c2) : 0.0f;
    float cinv3 = (c3 > eps) ? __frcp_rn(c3) : 0.0f;
    r0.x *= cinv0;
    r0.y *= cinv1;
    r0.z *= cinv2;
    r0.w *= cinv3;
    r1.x *= cinv0;
    r1.y *= cinv1;
    r1.z *= cinv2;
    r1.w *= cinv3;
    r2.x *= cinv0;
    r2.y *= cinv1;
    r2.z *= cinv2;
    r2.w *= cinv3;
    r3.x *= cinv0;
    r3.y *= cinv1;
    r3.z *= cinv2;
    r3.w *= cinv3;
}

// Full Sinkhorn forward iteration (row-norm then col-norm)
__device__ __forceinline__ void sk4_forward_step(float4& r0, float4& r1, float4& r2, float4& r3,
                                                 float eps) {
    sk4_row_normalize(r0, r1, r2, r3, eps);
    sk4_col_normalize(r0, r1, r2, r3, eps);
}

// Sinkhorn backward step through column normalization
// Given checkpoint w (after row-norm), updates d in-place
__device__ __forceinline__ void sk4_col_backward(float4& d0, float4& d1, float4& d2, float4& d3,
                                                 const float4& w0, const float4& w1,
                                                 const float4& w2, const float4& w3) {
    float col0 = d0.x * w0.x + d1.x * w1.x + d2.x * w2.x + d3.x * w3.x;
    float col1 = d0.y * w0.y + d1.y * w1.y + d2.y * w2.y + d3.y * w3.y;
    float col2 = d0.z * w0.z + d1.z * w1.z + d2.z * w2.z + d3.z * w3.z;
    float col3 = d0.w * w0.w + d1.w * w1.w + d2.w * w2.w + d3.w * w3.w;
    d0.x -= w0.x * col0;
    d0.y -= w0.y * col1;
    d0.z -= w0.z * col2;
    d0.w -= w0.w * col3;
    d1.x -= w1.x * col0;
    d1.y -= w1.y * col1;
    d1.z -= w1.z * col2;
    d1.w -= w1.w * col3;
    d2.x -= w2.x * col0;
    d2.y -= w2.y * col1;
    d2.z -= w2.z * col2;
    d2.w -= w2.w * col3;
    d3.x -= w3.x * col0;
    d3.y -= w3.y * col1;
    d3.z -= w3.z * col2;
    d3.w -= w3.w * col3;
}

// Sinkhorn backward step through row normalization
// Given checkpoint w (after row-norm), updates d in-place
__device__ __forceinline__ void sk4_row_backward(float4& d0, float4& d1, float4& d2, float4& d3,
                                                 const float4& w0, const float4& w1,
                                                 const float4& w2, const float4& w3) {
    float row0 = d0.x * w0.x + d0.y * w0.y + d0.z * w0.z + d0.w * w0.w;
    float row1 = d1.x * w1.x + d1.y * w1.y + d1.z * w1.z + d1.w * w1.w;
    float row2 = d2.x * w2.x + d2.y * w2.y + d2.z * w2.z + d2.w * w2.w;
    float row3 = d3.x * w3.x + d3.y * w3.y + d3.z * w3.z + d3.w * w3.w;
    d0.x -= w0.x * row0;
    d0.y -= w0.y * row0;
    d0.z -= w0.z * row0;
    d0.w -= w0.w * row0;
    d1.x -= w1.x * row1;
    d1.y -= w1.y * row1;
    d1.z -= w1.z * row1;
    d1.w -= w1.w * row1;
    d2.x -= w2.x * row2;
    d2.y -= w2.y * row2;
    d2.z -= w2.z * row2;
    d2.w -= w2.w * row2;
    d3.x -= w3.x * row3;
    d3.y -= w3.y * row3;
    d3.z -= w3.z * row3;
    d3.w -= w3.w * row3;
}

// Batched Sinkhorn backward for n=4, fully in registers with forward recomputation
template<int N_COMPILE>
__global__ __launch_bounds__(256, 4) void sinkhorn_knopp_backward_batched_n4_kernel(
    float* __restrict__ d_inp, const float* __restrict__ grad, const float* __restrict__ M_inp,
    int B, int num_iters, float eps) {
    static_assert(N_COMPILE == 4);

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= B)
        return;

    const float* inp_ptr = M_inp + batch_idx * 16;
    const float* grad_ptr = grad + batch_idx * 16;

    // Load original input (H_res_exp) — kept constant for recomputation
    float4 inp0 = *reinterpret_cast<const float4*>(inp_ptr);
    float4 inp1 = *reinterpret_cast<const float4*>(inp_ptr + 4);
    float4 inp2 = *reinterpret_cast<const float4*>(inp_ptr + 8);
    float4 inp3 = *reinterpret_cast<const float4*>(inp_ptr + 12);

    // Load gradient
    float4 d0 = *reinterpret_cast<const float4*>(grad_ptr);
    float4 d1 = *reinterpret_cast<const float4*>(grad_ptr + 4);
    float4 d2 = *reinterpret_cast<const float4*>(grad_ptr + 8);
    float4 d3 = *reinterpret_cast<const float4*>(grad_ptr + 12);

    // Backward iterations: for each Sinkhorn iteration (last to first),
    // recompute the forward checkpoint then apply backward step
    for (int iter = num_iters - 1; iter >= 0; iter--) {
        // Recompute forward from inp to get checkpoint at this iteration
        // Checkpoint = state after (iter) full iterations + one more row-norm
        float4 w0 = inp0, w1 = inp1, w2 = inp2, w3 = inp3;

        for (int fwd = 0; fwd < iter; fwd++) {
            sk4_forward_step(w0, w1, w2, w3, eps);
        }
        // One more row-norm (checkpoint is after row-norm, before col-norm)
        sk4_row_normalize(w0, w1, w2, w3, eps);

        // Backward through col normalization, then row normalization
        sk4_col_backward(d0, d1, d2, d3, w0, w1, w2, w3);
        sk4_row_backward(d0, d1, d2, d3, w0, w1, w2, w3);
    }

    // Write result
    float* out_ptr = d_inp + batch_idx * 16;
    *reinterpret_cast<float4*>(out_ptr) = d0;
    *reinterpret_cast<float4*>(out_ptr + 4) = d1;
    *reinterpret_cast<float4*>(out_ptr + 8) = d2;
    *reinterpret_cast<float4*>(out_ptr + 12) = d3;
}

template<int N_COMPILE>
__global__ __launch_bounds__(256, 8) void sinkhorn_knopp_batched_n4_kernel(
    float* __restrict__ out, const float* __restrict__ inp, int B, int num_iters, float eps) {
    static_assert(N_COMPILE == 4,
                  "This kernel is optimized for the case where n=4, which is the special case "
                  "highlighted in the section 4.3 introduction.");

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= B)
        return;

    const float* inp_batch = inp + batch_idx * 16;
    float* out_batch = out + batch_idx * 16;

    float4 row0 = *reinterpret_cast<const float4*>(inp_batch);
    float4 row1 = *reinterpret_cast<const float4*>(inp_batch + 4);
    float4 row2 = *reinterpret_cast<const float4*>(inp_batch + 8);
    float4 row3 = *reinterpret_cast<const float4*>(inp_batch + 12);

#pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        float s0 = row0.x + row0.y + row0.z + row0.w;
        float s1 = row1.x + row1.y + row1.z + row1.w;
        float s2 = row2.x + row2.y + row2.z + row2.w;
        float s3 = row3.x + row3.y + row3.z + row3.w;

        float inv0 = (s0 > eps) ? __frcp_rn(s0) : 0.0f;
        float inv1 = (s1 > eps) ? __frcp_rn(s1) : 0.0f;
        float inv2 = (s2 > eps) ? __frcp_rn(s2) : 0.0f;
        float inv3 = (s3 > eps) ? __frcp_rn(s3) : 0.0f;

        row0.x *= inv0;
        row0.y *= inv0;
        row0.z *= inv0;
        row0.w *= inv0;
        row1.x *= inv1;
        row1.y *= inv1;
        row1.z *= inv1;
        row1.w *= inv1;
        row2.x *= inv2;
        row2.y *= inv2;
        row2.z *= inv2;
        row2.w *= inv2;
        row3.x *= inv3;
        row3.y *= inv3;
        row3.z *= inv3;
        row3.w *= inv3;

        float c0 = row0.x + row1.x + row2.x + row3.x;
        float c1 = row0.y + row1.y + row2.y + row3.y;
        float c2 = row0.z + row1.z + row2.z + row3.z;
        float c3 = row0.w + row1.w + row2.w + row3.w;

        float cinv0 = (c0 > eps) ? __frcp_rn(c0) : 0.0f;
        float cinv1 = (c1 > eps) ? __frcp_rn(c1) : 0.0f;
        float cinv2 = (c2 > eps) ? __frcp_rn(c2) : 0.0f;
        float cinv3 = (c3 > eps) ? __frcp_rn(c3) : 0.0f;

        row0.x *= cinv0;
        row0.y *= cinv1;
        row0.z *= cinv2;
        row0.w *= cinv3;
        row1.x *= cinv0;
        row1.y *= cinv1;
        row1.z *= cinv2;
        row1.w *= cinv3;
        row2.x *= cinv0;
        row2.y *= cinv1;
        row2.z *= cinv2;
        row2.w *= cinv3;
        row3.x *= cinv0;
        row3.y *= cinv1;
        row3.z *= cinv2;
        row3.w *= cinv3;
    }

    *reinterpret_cast<float4*>(out_batch) = row0;
    *reinterpret_cast<float4*>(out_batch + 4) = row1;
    *reinterpret_cast<float4*>(out_batch + 8) = row2;
    *reinterpret_cast<float4*>(out_batch + 12) = row3;
}

template<int N_MAX, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE,
                             2) void sinkhorn_knopp_batched_kernel(float* __restrict__ out,
                                                                   const float* __restrict__ inp,
                                                                   int B, int n, int num_iters,
                                                                   float eps) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= B)
        return;

    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = tile + N_MAX * N_MAX;
    float* col_sums = row_sums + N_MAX;

    const float* inp_batch = inp + batch_idx * n * n;
    float* out_batch = out + batch_idx * n * n;

    int total = n * n;

    if (n == 4 && (total % 4) == 0) {
        int total_vec = total / 4;
        for (int i = threadIdx.x; i < total_vec; i += BLOCK_SIZE) {
            reinterpret_cast<float4*>(tile)[i] = reinterpret_cast<const float4*>(inp_batch)[i];
        }
    } else {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            tile[i] = inp_batch[i];
        }
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = threadIdx.x; r < n; r += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int c = 0; c < n; c++) {
                sum += tile[r * n + c];
            }
            row_sums[r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / n;
            tile[i] *= row_sums[r];
        }
        __syncthreads();

        for (int c = threadIdx.x; c < n; c += BLOCK_SIZE) {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = 0; r < n; r++) {
                sum += tile[r * n + c];
            }
            col_sums[c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int c = i % n;
            tile[i] *= col_sums[c];
        }
        __syncthreads();
    }

    if (n == 4 && (total % 4) == 0) {
        int total_vec = total / 4;
        for (int i = threadIdx.x; i < total_vec; i += BLOCK_SIZE) {
            reinterpret_cast<float4*>(out_batch)[i] = reinterpret_cast<float4*>(tile)[i];
        }
    } else {
        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            out_batch[i] = tile[i];
        }
    }
}

inline void sinkhorn_knopp_forward_batched(float* out, const float* inp, int B, int n,
                                           int num_iters, float eps,
                                           cudaStream_t stream = nullptr) {
    if (n == 4) {
        constexpr int THREADS_PER_BLOCK = 256;
        int num_blocks = (B + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.numAttrs = 1;
        config.attrs = attrs;
        config.blockDim = {THREADS_PER_BLOCK, 1, 1};
        config.gridDim = {(unsigned int)num_blocks, 1, 1};
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchKernelEx(&config, sinkhorn_knopp_batched_n4_kernel<4>, out, inp, B, num_iters,
                           eps);
#else
        sinkhorn_knopp_batched_n4_kernel<4>
            <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(out, inp, B, num_iters, eps);
#endif
        return;
    }

    constexpr int BLOCK_SIZE = 128;
    constexpr int N_MAX = 32;

    if (n > N_MAX) {
        for (int b = 0; b < B; b++) {
            sinkhorn_knopp_forward(out + b * n * n, inp + b * n * n, n, n, num_iters, eps, stream);
        }
        return;
    }

    size_t smem_size = N_MAX * N_MAX * sizeof(float) + 2 * N_MAX * sizeof(float);

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = {BLOCK_SIZE, 1, 1};
    config.gridDim = {(unsigned int)B, 1, 1};
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchKernelEx(&config, sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE>, out, inp, B, n,
                       num_iters, eps);
#else
    sinkhorn_knopp_batched_kernel<N_MAX, BLOCK_SIZE>
        <<<B, BLOCK_SIZE, smem_size, stream>>>(out, inp, B, n, num_iters, eps);
#endif
}

inline void sinkhorn_knopp_backward_batched(float* d_inp, const float* grad, const float* M_inp,
                                            int B, int n, int num_iters, float eps,
                                            cudaStream_t stream = nullptr) {
    if (n == 4) {
        constexpr int THREADS_PER_BLOCK = 256;
        int num_blocks = (B + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        sinkhorn_knopp_backward_batched_n4_kernel<4>
            <<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(d_inp, grad, M_inp, B, num_iters, eps);
        return;
    }

    // Fallback: launch one backward per batch element
    for (int b = 0; b < B; b++) {
        sinkhorn_knopp_backward(d_inp + b * n * n, grad + b * n * n, nullptr, M_inp + b * n * n, n,
                                num_iters, eps, stream);
    }
}

// Profiled dispatch variants — call production kernels with DO_PROFILE=true.

inline void sinkhorn_knopp_forward_profiled(float* out, const float* inp, int M, int N,
                                            int num_iters, float eps, int64_t* profiler_buf,
                                            int max_entries, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE, true>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps, profiler_buf,
                                                   max_entries);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_kernel<MAX_DIM, BLOCK_SIZE, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps, profiler_buf,
                                                     max_entries);
    }
}

inline void sinkhorn_knopp_forward_fused_exp_profiled(float* out, float* H_res_exp,
                                                      const float* inp, int M, int N, int num_iters,
                                                      float eps, int64_t* profiler_buf,
                                                      int max_entries,
                                                      cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M <= 64 && N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE, true>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps,
                                                   profiler_buf, max_entries);
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_fused_exp_kernel<MAX_DIM, BLOCK_SIZE, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, H_res_exp, inp, M, N, num_iters, eps,
                                                     profiler_buf, max_entries);
    }
}

inline void sinkhorn_knopp_backward_profiled(float* d_inp, const float* grad, const float* M_out,
                                             const float* M_inp, int N, int num_iters, float eps,
                                             int64_t* profiler_buf, int max_entries,
                                             cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = 2 * MAX_DIM * MAX_DIM * sizeof(float) + 4 * MAX_DIM * sizeof(float);

        sinkhorn_knopp_backward_kernel<MAX_DIM, BLOCK_SIZE, true>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_out, M_inp, N, num_iters, eps,
                                                   profiler_buf, max_entries);
    }
}

} // namespace mhc
