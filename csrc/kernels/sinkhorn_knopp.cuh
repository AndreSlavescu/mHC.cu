#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

template<int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_kernel(float* __restrict__ out, const float* __restrict__ inp, int M,
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
__global__ void sinkhorn_knopp_pdl_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                          int M, int N, int num_iters, float eps) {
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

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_single_block_kernel(float* __restrict__ out,
                                                   const float* __restrict__ inp, int M, int N,
                                                   int num_iters, float eps) {
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

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

inline void sinkhorn_knopp_forward(float* out, const float* inp, int M, int N, int num_iters,
                                   float eps, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (M <= 64 && N <= 64) {
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

        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    } else {
        constexpr int TILE_SIZE = 32;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        size_t smem_size = TILE_SIZE * TILE_SIZE * sizeof(float) + TILE_SIZE * sizeof(float) +
                           TILE_SIZE * sizeof(float);

        sinkhorn_knopp_kernel<TILE_SIZE, TILE_SIZE, BLOCK_SIZE>
            <<<grid, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
    }
}

template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_single_block_pdl_kernel(float* __restrict__ out,
                                                       const float* __restrict__ inp, int M, int N,
                                                       int num_iters, float eps) {
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

inline void sinkhorn_knopp_forward_pdl(float* out, const float* inp, int M, int N, int num_iters,
                                       float eps, cudaStream_t stream = nullptr) {
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
        config.gridDim = dim3(1);
        config.blockDim = dim3(BLOCK_SIZE);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;

        CHECK_CUDA(cudaLaunchKernelEx(&config,
                                      sinkhorn_knopp_single_block_pdl_kernel<MAX_DIM, BLOCK_SIZE>,
                                      out, inp, M, N, num_iters, eps));
#else
        sinkhorn_knopp_single_block_pdl_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
#endif
    } else if (M <= 128 && N <= 128) {
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_single_block_pdl_kernel<MAX_DIM, BLOCK_SIZE>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(1);
        config.blockDim = dim3(BLOCK_SIZE);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;

        CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, out, inp, M, N, num_iters, eps));
#else
        kernel<<<1, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
#endif
    } else {
        constexpr int TILE_SIZE = 32;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        size_t smem_size = TILE_SIZE * TILE_SIZE * sizeof(float) + TILE_SIZE * sizeof(float) +
                           TILE_SIZE * sizeof(float);

#ifdef MHC_ENABLE_PDL
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = dim3(BLOCK_SIZE);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;

        CHECK_CUDA(cudaLaunchKernelEx(&config,
                                      sinkhorn_knopp_pdl_kernel<TILE_SIZE, TILE_SIZE, BLOCK_SIZE>,
                                      out, inp, M, N, num_iters, eps));
#else
        sinkhorn_knopp_pdl_kernel<TILE_SIZE, TILE_SIZE, BLOCK_SIZE>
            <<<grid, BLOCK_SIZE, smem_size, stream>>>(out, inp, M, N, num_iters, eps);
#endif
    }
}
template<int MAX_DIM, int BLOCK_SIZE>
__global__ void
sinkhorn_knopp_backward_kernel(float* __restrict__ d_inp, const float* __restrict__ grad,
                               const float* __restrict__ M_out, const float* __restrict__ M_inp,
                               int N, int num_iters, float eps) {
    extern __shared__ float smem[];
    float* d_tile = smem;
    float* row_buffer = smem + MAX_DIM * MAX_DIM;
    float* col_buffer = row_buffer + MAX_DIM;
    float* tile_fwd = col_buffer + MAX_DIM;
    float* row_sums = tile_fwd + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total = N * N;

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_tile[i] = grad[i];
    }
    __syncthreads();

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
                row_sums[r] = sum;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int r = i / N;
                if (row_sums[r] > eps) {
                    tile_fwd[i] /= row_sums[r];
                }
            }
            __syncthreads();

            for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
                float sum = 0.0f;
                for (int r = 0; r < N; r++) {
                    sum += tile_fwd[r * N + c];
                }
                col_sums[c] = sum;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
                int c = i % N;
                if (col_sums[c] > eps) {
                    tile_fwd[i] /= col_sums[c];
                }
            }
            __syncthreads();
        }

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                sum += tile_fwd[r * N + c];
            }
            row_sums[r] = sum;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
            int r = i / N;
            if (row_sums[r] > eps) {
                tile_fwd[i] /= row_sums[r];
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
            col_sums[c] = 1.0f;
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
            d_tile[i] = (d_tile[i] - tile_fwd[i] * col_buffer[c]) / col_sums[c];
        }
        __syncthreads();

        for (int r = threadIdx.x; r < N; r += BLOCK_SIZE) {
            row_sums[r] = 1.0f;
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
            d_tile[i] = (d_tile[i] - tile_fwd[i] * row_buffer[r]) / row_sums[r];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < total; i += BLOCK_SIZE) {
        d_inp[i] = d_tile[i];
    }
}

inline void sinkhorn_knopp_backward(float* d_inp, const float* grad, const float* M_out,
                                    const float* M_inp, int N, int num_iters, float eps,
                                    cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N <= 64) {
        constexpr int MAX_DIM = 64;
        size_t smem_size = 2 * MAX_DIM * MAX_DIM * sizeof(float) + 4 * MAX_DIM * sizeof(float);

        sinkhorn_knopp_backward_kernel<MAX_DIM, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_size, stream>>>(d_inp, grad, M_out, M_inp, N, num_iters, eps);
    } else {
        fprintf(stderr, "sinkhorn_knopp_backward: N > 64 not supported\n");
    }
}

} // namespace mhc
