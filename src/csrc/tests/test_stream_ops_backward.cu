#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

int main() {
    const int B = 16, n = 4, C = 64;
    constexpr int BLOCK = 256;
    int workspace_num_blocks = std::min(128, (B * C + BLOCK - 1) / BLOCK);

    float* h_x = (float*)malloc(B * n * C * sizeof(float));
    float* h_y = (float*)malloc(B * C * sizeof(float));
    float* h_H_pre = (float*)malloc(n * sizeof(float));
    float* h_H_post = (float*)malloc(n * sizeof(float));
    float* h_M = (float*)malloc(n * n * sizeof(float));
    float* h_grad = (float*)malloc(B * n * C * sizeof(float));
    float* h_grad_agg = (float*)malloc(B * C * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++)
        h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < B * C; i++)
        h_y[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < n; i++)
        h_H_pre[i] = 1.0f / (1.0f + expf(-((float)rand() / RAND_MAX * 2.0f - 1.0f)));
    for (int i = 0; i < n; i++)
        h_H_post[i] = 2.0f / (1.0f + expf(-((float)rand() / RAND_MAX * 2.0f - 1.0f)));
    for (int i = 0; i < n * n; i++)
        h_M[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < B * n * C; i++)
        h_grad[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < B * C; i++)
        h_grad_agg[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    float *d_x, *d_y, *d_H_pre, *d_H_post, *d_M, *d_grad, *d_grad_agg;
    float *d_dx, *d_dy, *d_dM, *d_dH, *d_dx_agg, *d_dH_agg;
    float *d_workspace_M, *d_workspace_H;

    CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_pre, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_post, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_agg, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dy, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dM, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dH, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx_agg, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dH_agg, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_workspace_M, workspace_num_blocks * n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_workspace_H, workspace_num_blocks * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, B * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_pre, h_H_pre, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_post, h_H_post, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad, h_grad, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_agg, h_grad_agg, B * C * sizeof(float), cudaMemcpyHostToDevice));

    printf("Stream Ops Backward Test\n");
    printf("========================\nB=%d, n=%d, C=%d\n\n", B, n, C);

    stream_aggregate_backward(d_dx_agg, d_dH_agg, d_grad_agg, d_x, d_H_pre, B, n, C, d_workspace_H,
                              workspace_num_blocks);

    float* h_dx_agg_gpu = (float*)malloc(B * n * C * sizeof(float));
    float* h_dH_agg_gpu = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(
        cudaMemcpy(h_dx_agg_gpu, d_dx_agg, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dH_agg_gpu, d_dH_agg, n * sizeof(float), cudaMemcpyDeviceToHost));

    float* h_dx_agg_cpu = (float*)malloc(B * n * C * sizeof(float));
    float* h_dH_agg_cpu = (float*)calloc(n, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                h_dx_agg_cpu[b * n * C + i * C + c] = h_grad_agg[b * C + c] * h_H_pre[i];
                h_dH_agg_cpu[i] += h_grad_agg[b * C + c] * h_x[b * n * C + i * C + c];
            }
        }
    }
    float dx_agg_diff = max_abs_diff(h_dx_agg_gpu, h_dx_agg_cpu, B * n * C);
    float dH_agg_diff = max_abs_diff(h_dH_agg_gpu, h_dH_agg_cpu, n);
    printf("stream_aggregate_backward d_x: max diff = %.6e %s\n", dx_agg_diff,
           dx_agg_diff < 1e-5f ? "PASSED" : "FAILED");
    printf("stream_aggregate_backward d_H: max diff = %.6e %s\n", dH_agg_diff,
           dH_agg_diff < 1e-3f ? "PASSED" : "FAILED");

    stream_distribute_mix_backward_fused(d_dx, d_dy, d_dM, d_dH, d_grad, d_x, d_y, d_M, d_H_post, B,
                                         n, C, d_workspace_M, d_workspace_H, workspace_num_blocks);

    float* h_dx_gpu = (float*)malloc(B * n * C * sizeof(float));
    float* h_dy_gpu = (float*)malloc(B * C * sizeof(float));
    float* h_dM_gpu = (float*)malloc(n * n * sizeof(float));
    float* h_dH_gpu = (float*)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_dx_gpu, d_dx, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dy_gpu, d_dy, B * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dM_gpu, d_dM, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dH_gpu, d_dH, n * sizeof(float), cudaMemcpyDeviceToHost));

    float* h_dx_cpu = (float*)calloc(B * n * C, sizeof(float));
    float* h_dy_cpu = (float*)calloc(B * C, sizeof(float));
    float* h_dM_cpu = (float*)calloc(n * n, sizeof(float));
    float* h_dH_cpu = (float*)calloc(n, sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                float g = h_grad[b * n * C + i * C + c];
                h_dy_cpu[b * C + c] += h_H_post[i] * g;
                h_dH_cpu[i] += g * h_y[b * C + c];
                for (int j = 0; j < n; j++) {
                    h_dx_cpu[b * n * C + j * C + c] += h_M[i * n + j] * g;
                    h_dM_cpu[i * n + j] += g * h_x[b * n * C + j * C + c];
                }
            }
        }
    }

    float dx_diff = max_abs_diff(h_dx_gpu, h_dx_cpu, B * n * C);
    float dy_diff = max_abs_diff(h_dy_gpu, h_dy_cpu, B * C);
    float dM_diff = max_abs_diff(h_dM_gpu, h_dM_cpu, n * n);
    float dH_diff = max_abs_diff(h_dH_gpu, h_dH_cpu, n);

    printf("stream_distribute_mix_backward d_x: max diff = %.6e %s\n", dx_diff,
           dx_diff < 1e-4f ? "PASSED" : "FAILED");
    printf("stream_distribute_mix_backward d_y: max diff = %.6e %s\n", dy_diff,
           dy_diff < 1e-4f ? "PASSED" : "FAILED");
    printf("stream_distribute_mix_backward d_M: max diff = %.6e %s\n", dM_diff,
           dM_diff < 1e-3f ? "PASSED" : "FAILED");
    printf("stream_distribute_mix_backward d_H: max diff = %.6e %s\n", dH_diff,
           dH_diff < 1e-3f ? "PASSED" : "FAILED");

    return 0;
}
