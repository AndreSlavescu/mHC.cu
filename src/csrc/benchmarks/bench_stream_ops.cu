#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

int main() {
    const int bench_runs = 100;

    L2Flusher flusher;

    int configs[][3] = {
        {128, 4, 4096}, {256, 4, 4096}, {512, 4, 4096}, {256, 8, 4096}, {512, 8, 4096},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Stream Ops Benchmark\n");
    printf("====================================\n");

    printf("stream_aggregate (x[B,n,C] * H_pre[n] -> out[B,C])\n");
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
            h_H[i] = 1.0f / n;

        float *d_x, *d_H, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * C * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_H, h_H, n * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes = (B * n * C + n) * sizeof(float) + B * C * sizeof(float);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_aggregate(d_out, d_x, d_H, B, n, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_x);
        cudaFree(d_H);
        cudaFree(d_out);
        free(h_x);
        free(h_H);
    }

    printf("\nstream_distribute (x[B,C] * H_post[n] -> out[B,n,C])\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0];
        int n = configs[c][1];
        int C = configs[c][2];

        float* h_x = (float*)malloc(B * C * sizeof(float));
        float* h_H = (float*)malloc(n * sizeof(float));

        srand(42);
        for (int i = 0; i < B * C; i++)
            h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < n; i++)
            h_H[i] = 1.0f;

        float *d_x, *d_H, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * n * C * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, B * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_H, h_H, n * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes = (B * C + n) * sizeof(float) + B * n * C * sizeof(float);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_distribute(d_out, d_x, d_H, B, n, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_x);
        cudaFree(d_H);
        cudaFree(d_out);
        free(h_x);
        free(h_H);
    }

    printf("\nstream_mix (x[B,n,C] @ M[n,n] -> out[B,n,C])\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0];
        int n = configs[c][1];
        int C = configs[c][2];

        float* h_x = (float*)malloc(B * n * C * sizeof(float));
        float* h_M = (float*)malloc(n * n * sizeof(float));

        srand(42);
        for (int i = 0; i < B * n * C; i++)
            h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < n * n; i++)
            h_M[i] = 1.0f / n;

        float *d_x, *d_M, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * n * C * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));

        size_t bytes = (B * n * C + n * n) * sizeof(float) + B * n * C * sizeof(float);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_mix(d_out, d_x, d_M, B, n, C);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_x);
        cudaFree(d_M);
        cudaFree(d_out);
        free(h_x);
        free(h_M);
    }

    return 0;
}
