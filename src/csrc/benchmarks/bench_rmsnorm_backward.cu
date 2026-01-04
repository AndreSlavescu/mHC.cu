#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/rmsnorm.cuh"

using namespace mhc;

int main() {
    const int bench_runs = 100;

    L2Flusher flusher;

    int configs[][2] = {
        {128, 4096},  {256, 4096},  {512, 4096},  {1024, 4096},
        {2048, 4096}, {1024, 8192}, {2048, 8192},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("RMSNorm Backward Benchmark\n");
    printf("====================================\n");
    printf("%8s %8s %12s %12s\n", "N", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("---------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int N = configs[c][0];
        int C = configs[c][1];

        floatX* h_inp = (floatX*)malloc(N * C * sizeof(floatX));
        floatX* h_weight = (floatX*)malloc(C * sizeof(floatX));
        float* h_grad = (float*)malloc(N * C * sizeof(float));
        float* h_rms = (float*)malloc(N * sizeof(float));

        srand(42);
        for (int i = 0; i < N * C; i++) {
            h_inp[i] = (floatX)((float)rand() / RAND_MAX * 2.0f - 1.0f);
            h_grad[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        for (int i = 0; i < C; i++) {
            h_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
        }
        for (int i = 0; i < N; i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < C; j++) {
                float v = (float)h_inp[i * C + j];
                sum_sq += v * v;
            }
            h_rms[i] = sqrtf(sum_sq / (float)C + 1e-5f);
        }

        floatX *d_inp, *d_weight;
        float *d_grad, *d_rms, *d_d_inp, *d_d_weight;

        CHECK_CUDA(cudaMalloc(&d_inp, N * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_weight, C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_grad, N * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rms, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_d_inp, N * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_d_weight, C * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, N * C * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, C * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grad, h_grad, N * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_rms, h_rms, N * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes_read = (size_t)N * C * sizeof(floatX) + (size_t)C * sizeof(floatX) +
                            (size_t)N * C * sizeof(float) + (size_t)N * sizeof(float);
        size_t bytes_write = (size_t)N * C * sizeof(float) + (size_t)C * sizeof(float);
        size_t total_bytes = bytes_read + bytes_write;

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            CHECK_CUDA(cudaMemset(d_d_weight, 0, C * sizeof(float)));

            timer.record_start();
            rmsnorm_backward(d_d_inp, d_d_weight, d_grad, d_inp, d_weight, d_rms, N, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (total_bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %12.2f %12.0f\n", N, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_inp);
        cudaFree(d_weight);
        cudaFree(d_grad);
        cudaFree(d_rms);
        cudaFree(d_d_inp);
        cudaFree(d_d_weight);
        free(h_inp);
        free(h_weight);
        free(h_grad);
        free(h_rms);
    }

    return 0;
}
