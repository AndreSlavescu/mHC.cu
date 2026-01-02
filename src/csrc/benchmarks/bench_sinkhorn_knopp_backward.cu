#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/sinkhorn_knopp.cuh"

using namespace mhc;

int main() {
    const int bench_runs = 100;
    const float eps = 1e-8f;

    L2Flusher flusher;

    struct Config {
        int M;
        int N;
        int iters;
    };

    Config configs[] = {
        {32, 32, 5}, {32, 32, 10}, {32, 32, 20}, {64, 64, 5}, {64, 64, 10}, {64, 64, 20},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Sinkhorn-Knopp Backward Benchmark\n");
    printf("=======================================================================\n");
    printf("%6s %6s %6s %12s %10s %10s %12s\n", "M", "N", "Iters", "Time (us)", "us/iter", "GFLOPS",
           "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int M = configs[c].M;
        int N = configs[c].N;
        int num_iters = configs[c].iters;

        float* h_inp = (float*)malloc(M * N * sizeof(float));
        float* h_grad = (float*)malloc(M * N * sizeof(float));

        srand(42);
        for (int i = 0; i < M * N; i++) {
            h_inp[i] = (float)rand() / RAND_MAX + 0.1f;
            h_grad[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }

        float *d_inp, *d_out, *d_grad, *d_dinp;
        CHECK_CUDA(cudaMalloc(&d_inp, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dinp, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grad, h_grad, M * N * sizeof(float), cudaMemcpyHostToDevice));

        sinkhorn_knopp_forward(d_out, d_inp, M, N, num_iters, eps);
        CHECK_CUDA(cudaDeviceSynchronize());

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();

            timer.record_start();
            sinkhorn_knopp_backward(d_dinp, d_grad, d_out, d_inp, N, num_iters, eps);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float time_us = avg_time_ms * 1000.0f;
        float time_per_iter_us = time_us / num_iters;

        double flops_per_iter = 8.0 * M * N;
        double total_flops = flops_per_iter * num_iters;
        float gflops = (total_flops / 1e9f) / (avg_time_ms / 1e3f);

        size_t bytes_io = 4 * M * N * sizeof(float);
        float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

        printf("%6d %6d %6d %12.2f %10.2f %10.2f %12.2f\n", M, N, num_iters, time_us,
               time_per_iter_us, gflops, bw);

        cudaFree(d_inp);
        cudaFree(d_out);
        cudaFree(d_grad);
        cudaFree(d_dinp);
        free(h_inp);
        free(h_grad);
    }

    return 0;
}
