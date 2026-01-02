#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/sinkhorn_knopp.cuh"

using namespace mhc;

template<int MAX_DIM, int BLOCK_SIZE, bool DO_PROFILE>
__global__ void
sinkhorn_knopp_profiled_kernel(float* __restrict__ out, const float* __restrict__ inp, int M, int N,
                               int num_iters, float eps, int64_t* profiler_buf, int max_entries) {
    extern __shared__ float smem[];
    float* tile = smem;
    float* row_sums = smem + MAX_DIM * MAX_DIM;
    float* col_sums = row_sums + MAX_DIM;

    int total_elems = M * N;

    DeviceProfiler profiler;
    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.init(max_entries, profiler_buf, blockIdx.x);
            profiler.start(TagLoad);
        }
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        tile[i] = inp[i];
    }
    __syncthreads();

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
        }
    }

    for (int iter = 0; iter < num_iters; iter++) {
        if constexpr (DO_PROFILE) {
            if (threadIdx.x == 0 && iter == 0) {
                profiler.start(TagCompute);
            }
        }

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

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
            profiler.start(TagStore);
        }
    }

    for (int i = threadIdx.x; i < total_elems; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
            profiler.flush();
        }
    }
}

int main() {
    const int bench_runs = 100;
    const float eps = 1e-8f;
    const int max_entries = 8;

    L2Flusher flusher;

    struct Config {
        int M;
        int N;
        int iters;
    };

    Config configs[] = {
        {32, 32, 5},  {32, 32, 10},  {32, 32, 20},   {64, 64, 5},    {64, 64, 10},
        {64, 64, 20}, {128, 128, 5}, {128, 128, 10}, {128, 128, 20},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Sinkhorn-Knopp Forward Benchmark\n");
    printf("Note: Small matrices are latency-bound (single block). GFLOPS will be low.\n");
    printf("=======================================================================\n");
    printf("%6s %6s %6s %12s %10s %10s %12s\n", "M", "N", "Iters", "Time (us)", "us/iter", "GFLOPS",
           "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int M = configs[c].M;
        int N = configs[c].N;
        int num_iters = configs[c].iters;

        float* h_inp = (float*)malloc(M * N * sizeof(float));

        srand(42);
        for (int i = 0; i < M * N; i++) {
            h_inp[i] = (float)rand() / RAND_MAX + 0.1f;
        }

        float *d_inp, *d_out;
        CHECK_CUDA(cudaMalloc(&d_inp, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, M * N * sizeof(float), cudaMemcpyHostToDevice));

        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_profiled_kernel<MAX_DIM, BLOCK_SIZE, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();

            timer.record_start();
            kernel<<<1, BLOCK_SIZE, smem_size>>>(d_out, d_inp, M, N, num_iters, eps, nullptr, 0);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float time_us = avg_time_ms * 1000.0f;
        float time_per_iter_us = time_us / num_iters;

        double flops_per_iter = 4.0 * M * N;
        double total_flops = flops_per_iter * num_iters;
        float gflops = (total_flops / 1e9f) / (avg_time_ms / 1e3f);

        size_t bytes_io = 2 * M * N * sizeof(float);
        float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

        printf("%6d %6d %6d %12.2f %10.2f %10.2f %12.2f\n", M, N, num_iters, time_us,
               time_per_iter_us, gflops, bw);

        cudaFree(d_inp);
        cudaFree(d_out);
        free(h_inp);
    }

    printf("\n--- Phase Breakdown (64 x 64, 10 iterations) ---\n");
    {
        int M = 64, N = 64, num_iters = 10;

        float* h_inp = (float*)malloc(M * N * sizeof(float));
        srand(42);
        for (int i = 0; i < M * N; i++) {
            h_inp[i] = (float)rand() / RAND_MAX + 0.1f;
        }

        float *d_inp, *d_out;
        CHECK_CUDA(cudaMalloc(&d_inp, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, M * N * sizeof(float), cudaMemcpyHostToDevice));

        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_DIM = 128;
        size_t smem_size =
            MAX_DIM * MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float) + MAX_DIM * sizeof(float);

        auto kernel = sinkhorn_knopp_profiled_kernel<MAX_DIM, BLOCK_SIZE, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        HostProfiler profiler(1, max_entries);

        flusher.flush();
        kernel<<<1, BLOCK_SIZE, smem_size>>>(d_out, d_inp, M, N, num_iters, eps,
                                             profiler.device_ptr(), max_entries);
        CHECK_CUDA(cudaDeviceSynchronize());

        profiler.print_summary();
        profiler.print_timeline(1);

        cudaFree(d_inp);
        cudaFree(d_out);
        free(h_inp);
    }

    return 0;
}
