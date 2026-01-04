#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

int main() {
    const int B = 16, n = 4, C = 64;

    float* h_x = (float*)malloc(B * n * C * sizeof(float));
    float* h_y = (float*)malloc(B * C * sizeof(float));
    float* h_H_pre = (float*)malloc(n * sizeof(float));
    float* h_H_post = (float*)malloc(n * sizeof(float));
    float* h_M = (float*)malloc(n * n * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++)
        h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < B * C; i++)
        h_y[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < n; i++)
        h_H_pre[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < n; i++)
        h_H_post[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < n * n; i++)
        h_M[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    float *d_x, *d_H_pre, *d_H_post, *d_M, *d_H_pre_act, *d_H_post_act, *d_out;
    floatX *d_agg_bf16, *d_y_bf16;

    CHECK_CUDA(cudaMalloc(&d_x, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_pre, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_post, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_pre_act, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_post_act, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_agg_bf16, B * C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_y_bf16, B * C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_out, B * n * C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_pre, h_H_pre, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_post, h_H_post, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));

    float* h_y_tmp = (float*)malloc(B * C * sizeof(float));
    for (int i = 0; i < B * C; i++)
        h_y_tmp[i] = h_y[i];
    float* d_y_f32;
    CHECK_CUDA(cudaMalloc(&d_y_f32, B * C * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y_f32, h_y_tmp, B * C * sizeof(float), cudaMemcpyHostToDevice));
    float_to_bf16(d_y_bf16, d_y_f32, B * C);

    printf("Stream Ops Test\n");
    printf("=======================\nB=%d, n=%d, C=%d\n\n", B, n, C);

    stream_aggregate_bf16_fused_sigmoid(d_agg_bf16, d_H_pre_act, d_x, d_H_pre, B, n, C);
    float* h_agg_gpu = (float*)malloc(B * C * sizeof(float));
    floatX* h_agg_bf16 = (floatX*)malloc(B * C * sizeof(floatX));
    CHECK_CUDA(cudaMemcpy(h_agg_bf16, d_agg_bf16, B * C * sizeof(floatX), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B * C; i++)
        h_agg_gpu[i] = (float)h_agg_bf16[i];

    float* h_agg_cpu = (float*)malloc(B * C * sizeof(float));
    for (int bc = 0; bc < B * C; bc++) {
        int b = bc / C, c = bc % C;
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float h_act = 1.0f / (1.0f + expf(-h_H_pre[i]));
            sum += h_act * h_x[b * n * C + i * C + c];
        }
        h_agg_cpu[bc] = sum;
    }
    float agg_diff = max_abs_diff(h_agg_gpu, h_agg_cpu, B * C);
    printf("stream_aggregate_bf16_fused_sigmoid: max diff = %.6e %s\n", agg_diff,
           agg_diff < 0.01f ? "PASSED" : "FAILED");

    stream_distribute_mix_add_fused(d_out, d_H_post_act, d_x, d_y_bf16, d_H_post, d_M, B, n, C);
    float* h_out_gpu = (float*)malloc(B * n * C * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));

    float* h_out_cpu = (float*)malloc(B * n * C * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            float h_act = 2.0f / (1.0f + expf(-h_H_post[i]));
            for (int c = 0; c < C; c++) {
                float mix = 0.0f;
                for (int j = 0; j < n; j++)
                    mix += h_M[i * n + j] * h_x[b * n * C + j * C + c];
                h_out_cpu[b * n * C + i * C + c] = mix + h_act * h_y[b * C + c];
            }
        }
    }
    float out_diff = max_abs_diff(h_out_gpu, h_out_cpu, B * n * C);
    printf("stream_distribute_mix_add_fused: max diff = %.6e %s\n", out_diff,
           out_diff < 0.01f ? "PASSED" : "FAILED");

    cudaFree(d_x);
    cudaFree(d_H_pre);
    cudaFree(d_H_post);
    cudaFree(d_M);
    cudaFree(d_H_pre_act);
    cudaFree(d_H_post_act);
    cudaFree(d_agg_bf16);
    cudaFree(d_y_bf16);
    cudaFree(d_y_f32);
    cudaFree(d_out);
    free(h_x);
    free(h_y);
    free(h_H_pre);
    free(h_H_post);
    free(h_M);
    free(h_agg_gpu);
    free(h_agg_bf16);
    free(h_agg_cpu);
    free(h_out_gpu);
    free(h_out_cpu);
    free(h_y_tmp);

    return 0;
}
