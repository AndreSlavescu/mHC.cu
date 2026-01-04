#include <cmath>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/fused_rmsnorm_matmul.cuh"

using namespace mhc;

void fused_rmsnorm_matmul_forward_cpu(float* out, float* rms_out, const float* x,
                                      const float* proj_weight, int M, int N, int K, float eps) {
    for (int row = 0; row < M; row++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < K; i++) {
            sum_sq += x[row * K + i] * x[row * K + i];
        }
        float rms = sqrtf(sum_sq / (float)K + eps);
        rms_out[row] = rms;
        float rms_inv = 1.0f / rms;

        for (int o = 0; o < N; o++) {
            float acc = 0.0f;
            for (int i = 0; i < K; i++) {
                acc += x[row * K + i] * proj_weight[o * K + i];
            }
            out[row * N + o] = acc * rms_inv;
        }
    }
}

int main() {
    printf("Fused RMSNorm + Matmul Forward Test\n");
    printf("====================================\n");

    const int M = 32;
    const int N = 64;
    const int K = 128;
    const float eps = 1e-5f;

    printf("M=%d, N=%d, K=%d\n", M, N, K);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    float* h_x = new float[M * K];
    float* h_proj_weight = new float[N * K];
    float* h_out_cpu = new float[M * N];
    float* h_rms_cpu = new float[M];
    float* h_out_gpu = new float[M * N];
    float* h_rms_gpu = new float[M];

    floatX* h_x_bf16 = new floatX[M * K];
    floatX* h_proj_weight_bf16 = new floatX[N * K];

    for (int i = 0; i < M * K; i++) {
        h_x[i] = dist(gen);
        h_x_bf16[i] = (floatX)h_x[i];
    }
    for (int i = 0; i < N * K; i++) {
        h_proj_weight[i] = dist(gen);
        h_proj_weight_bf16[i] = (floatX)h_proj_weight[i];
    }

    fused_rmsnorm_matmul_forward_cpu(h_out_cpu, h_rms_cpu, h_x, h_proj_weight, M, N, K, eps);

    floatX *d_x, *d_proj_weight;
    float* d_out;

    CHECK_CUDA(cudaMalloc(&d_x, M * K * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_proj_weight, N * K * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x_bf16, M * K * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_proj_weight, h_proj_weight_bf16, N * K * sizeof(floatX),
                          cudaMemcpyHostToDevice));

    FusedRMSNormMatmul fused;
    fused.init(M, N, K, eps);
    fused.forward(d_out, d_x, d_proj_weight);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_rms_gpu, fused.get_rms_values(), M * sizeof(float), cudaMemcpyDeviceToHost));

#if DEBUG
    printf("\nSample output (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_cpu[i]);
    printf("\n");

    printf("\nSample RMS (first 5):\n");
    printf("  GPU: ");
    for (int i = 0; i < 5; i++)
        printf("%.4f ", h_rms_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 5; i++)
        printf("%.4f ", h_rms_cpu[i]);
    printf("\n");
#endif

    float out_diff = max_abs_diff(h_out_gpu, h_out_cpu, M * N);
    float rms_diff = max_abs_diff(h_rms_gpu, h_rms_cpu, M);

    float out_tol = 6e-2f;
    float rms_tol = 1e-3f;

    printf("\n");
    check_test(out_diff, out_tol, "Forward output");
    check_test(rms_diff, rms_tol, "RMS values");

    fused.destroy();
    cudaFree(d_x);
    cudaFree(d_proj_weight);
    cudaFree(d_out);

    delete[] h_x;
    delete[] h_proj_weight;
    delete[] h_out_cpu;
    delete[] h_rms_cpu;
    delete[] h_out_gpu;
    delete[] h_rms_gpu;
    delete[] h_x_bf16;
    delete[] h_proj_weight_bf16;

    return 0;
}
