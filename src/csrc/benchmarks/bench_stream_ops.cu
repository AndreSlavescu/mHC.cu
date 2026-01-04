#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

int main() {
    const int bench_runs = 100;

    L2Flusher flusher;

    int configs[][3] = {
        {320, 4, 1280},
        {512, 4, 1920},
        {1280, 4, 2560},
        {2560, 4, 1280},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Stream Ops Benchmark\n");
    printf("==========================================\n");

    printf("stream_aggregate_bf16_fused_sigmoid\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0];
        int n = configs[c][1];
        int C = configs[c][2];

        float* h_x = (float*)malloc(B * n * C * sizeof(float));
        float* h_H = (float*)malloc(n * sizeof(float));

        srand(42);
        for (int i = 0; i < B * n * C; i++)
            h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < n; i++)
            h_H[i] = 0.0f;

        float *d_x, *d_H, *d_H_activated;
        floatX* d_out;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H_activated, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * C * sizeof(floatX)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_H, h_H, n * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes = (B * n * C + n) * sizeof(float) + B * C * sizeof(floatX);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_aggregate_bf16_fused_sigmoid(d_out, d_H_activated, d_x, d_H, B, n, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_x);
        cudaFree(d_H);
        cudaFree(d_H_activated);
        cudaFree(d_out);
        free(h_x);
        free(h_H);
    }

    printf("\nstream_distribute_mix_add_fused\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0];
        int n = configs[c][1];
        int C = configs[c][2];

        float* h_x = (float*)malloc(B * n * C * sizeof(float));
        floatX* h_y_norm = (floatX*)malloc(B * C * sizeof(floatX));
        float* h_H = (float*)malloc(n * sizeof(float));
        float* h_M = (float*)malloc(n * n * sizeof(float));

        srand(42);
        for (int i = 0; i < B * n * C; i++)
            h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < B * C; i++)
            h_y_norm[i] = (floatX)((float)rand() / RAND_MAX * 2.0f - 1.0f);
        for (int i = 0; i < n; i++)
            h_H[i] = 0.0f;
        for (int i = 0; i < n * n; i++)
            h_M[i] = 1.0f / n;

        float *d_x, *d_H, *d_H_activated, *d_M, *d_out;
        floatX* d_y_norm;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y_norm, B * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H_activated, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * n * C * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y_norm, h_y_norm, B * C * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_H, h_H, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes = (B * n * C + n * n + n) * sizeof(float) + B * C * sizeof(floatX) +
                       B * n * C * sizeof(float);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_distribute_mix_add_fused(d_out, d_H_activated, d_x, d_y_norm, d_H, d_M, B, n, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_x);
        cudaFree(d_y_norm);
        cudaFree(d_H);
        cudaFree(d_H_activated);
        cudaFree(d_M);
        cudaFree(d_out);
        free(h_x);
        free(h_y_norm);
        free(h_H);
        free(h_M);
    }

    return 0;
}
