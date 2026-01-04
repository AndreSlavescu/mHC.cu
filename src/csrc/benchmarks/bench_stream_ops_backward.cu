#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

    printf("Stream Ops Backward Benchmark\n");
    printf("==============================================\n\n");

    printf("stream_aggregate_backward\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0], n = configs[c][1], C = configs[c][2];
        constexpr int BLOCK = 256;
        int workspace_num_blocks = std::min(128, (B * C + BLOCK - 1) / BLOCK);

        float *d_x, *d_H, *d_grad, *d_dx, *d_dH, *d_workspace;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dx, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dH, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_num_blocks * n * sizeof(float)));

        size_t bytes = (B * n * C + B * C + n) * sizeof(float) + (B * n * C + n) * sizeof(float);
        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_aggregate_backward(d_dx, d_dH, d_grad, d_x, d_H, B, n, C, d_workspace,
                                      workspace_num_blocks);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_ms = total_time / bench_runs;
        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_ms * 1000.0f,
               (bytes / 1e9f) / (avg_ms / 1e3f));

        cudaFree(d_x);
        cudaFree(d_H);
        cudaFree(d_grad);
        cudaFree(d_dx);
        cudaFree(d_dH);
        cudaFree(d_workspace);
    }

    printf("\nstream_distribute_mix_backward_fused\n");
    printf("%8s %8s %8s %12s %12s\n", "B", "n", "C", "Time (us)", "BW (GB/s)");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c][0], n = configs[c][1], C = configs[c][2];
        constexpr int BLOCK = 256;
        int workspace_num_blocks = std::min(128, (B * C + BLOCK - 1) / BLOCK);

        float *d_x, *d_y, *d_H, *d_M, *d_grad;
        float *d_dx, *d_dy, *d_dM, *d_dH, *d_workspace_M, *d_workspace_H;
        CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dx, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dy, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dM, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dH, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_workspace_M, workspace_num_blocks * n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_workspace_H, workspace_num_blocks * n * sizeof(float)));

        size_t bytes = (2 * B * n * C + B * C + n * n + n) * sizeof(float) +
                       (B * n * C + B * C + n * n + n) * sizeof(float);
        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            timer.record_start();
            stream_distribute_mix_backward_fused(d_dx, d_dy, d_dM, d_dH, d_grad, d_x, d_y, d_M, d_H,
                                                 B, n, C, d_workspace_M, d_workspace_H,
                                                 workspace_num_blocks);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_ms = total_time / bench_runs;
        printf("%8d %8d %8d %12.2f %12.0f\n", B, n, C, avg_ms * 1000.0f,
               (bytes / 1e9f) / (avg_ms / 1e3f));

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_H);
        cudaFree(d_M);
        cudaFree(d_grad);
        cudaFree(d_dx);
        cudaFree(d_dy);
        cudaFree(d_dM);
        cudaFree(d_dH);
        cudaFree(d_workspace_M);
        cudaFree(d_workspace_H);
    }

    return 0;
}
