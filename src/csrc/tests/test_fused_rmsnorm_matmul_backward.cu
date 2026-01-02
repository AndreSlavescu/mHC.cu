#include <cmath>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/fused_rmsnorm_matmul.cuh"

using namespace mhc;

void fused_rmsnorm_matmul_backward_cpu(float* dW, float* dx, const float* grad_output,
                                       const float* x, const float* weight, const float* rms, int M,
                                       int N, int K) {
    for (int row = 0; row < M; row++) {
        float r = rms[row];
        float r_inv = 1.0f / r;

        for (int o = 0; o < N; o++) {
            float grad_scaled = grad_output[row * N + o] * r_inv;
            for (int i = 0; i < K; i++) {
                dW[o * K + i] += grad_scaled * x[row * K + i];
            }
        }

        float K_vals[4096];
        for (int i = 0; i < K; i++) {
            K_vals[i] = 0.0f;
            for (int o = 0; o < N; o++) {
                float grad_scaled = grad_output[row * N + o] * r_inv;
                K_vals[i] += grad_scaled * weight[o * K + i];
            }
        }

        float K_dot_x = 0.0f;
        for (int i = 0; i < K; i++) {
            K_dot_x += K_vals[i] * x[row * K + i];
        }

        float correction_scale = K_dot_x / ((float)K * r * r);
        for (int i = 0; i < K; i++) {
            dx[row * K + i] = K_vals[i] - correction_scale * x[row * K + i];
        }
    }
}

int main() {
    printf("Fused RMSNorm + Matmul Backward Test\n");
    printf("====================================\n");

    const int M = 32;
    const int N = 64;
    const int K = 128;
    const float eps = 1e-5f;

    printf("M=%d, N=%d, K=%d\n", M, N, K);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    float* h_x = new float[M * K];
    float* h_weight = new float[N * K];
    float* h_grad_output = new float[M * N];
    float* h_rms = new float[M];
    float* h_dW_cpu = new float[N * K]();
    float* h_dx_cpu = new float[M * K]();
    float* h_dW_gpu = new float[N * K];
    float* h_dx_gpu = new float[M * K];

    floatX* h_x_bf16 = new floatX[M * K];
    floatX* h_weight_bf16 = new floatX[N * K];

    for (int i = 0; i < M * K; i++) {
        h_x[i] = dist(gen);
        h_x_bf16[i] = (floatX)h_x[i];
    }
    for (int i = 0; i < N * K; i++) {
        h_weight[i] = dist(gen);
        h_weight_bf16[i] = (floatX)h_weight[i];
    }
    for (int i = 0; i < M * N; i++) {
        h_grad_output[i] = dist(gen);
    }

    for (int row = 0; row < M; row++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < K; i++) {
            sum_sq += h_x[row * K + i] * h_x[row * K + i];
        }
        h_rms[row] = sqrtf(sum_sq / (float)K + eps);
    }

    fused_rmsnorm_matmul_backward_cpu(h_dW_cpu, h_dx_cpu, h_grad_output, h_x, h_weight, h_rms, M, N,
                                      K);

    floatX *d_x, *d_weight;
    float *d_grad_output, *d_dW, *d_dx, *d_rms;

    CHECK_CUDA(cudaMalloc(&d_x, M * K * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_weight, N * K * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW, N * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rms, M * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x_bf16, M * K * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight_bf16, N * K * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_grad_output, h_grad_output, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rms, h_rms, M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_dW, 0, N * K * sizeof(float)));

    FusedRMSNormMatmulBackward backward;
    backward.init(M, N, K, eps);
    backward.backward(d_dW, d_dx, d_grad_output, d_x, d_weight, d_rms);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_dW_gpu, d_dW, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dx_gpu, d_dx, M * K * sizeof(float), cudaMemcpyDeviceToHost));

#if DEBUG
    printf("\nSample dW (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_dW_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_dW_cpu[i]);
    printf("\n");

    printf("\nSample dx (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_dx_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_dx_cpu[i]);
    printf("\n");
#endif

    float dW_diff = max_abs_diff(h_dW_gpu, h_dW_cpu, N * K);
    float dx_diff = max_abs_diff(h_dx_gpu, h_dx_cpu, M * K);

    float dW_tol = 0.05f;
    float dx_tol = 0.06f;

    printf("\n");
    check_test(dW_diff, dW_tol, "dW gradient");
    check_test(dx_diff, dx_tol, "dx gradient");

    backward.destroy();
    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_grad_output);
    cudaFree(d_dW);
    cudaFree(d_dx);
    cudaFree(d_rms);

    delete[] h_x;
    delete[] h_weight;
    delete[] h_grad_output;
    delete[] h_rms;
    delete[] h_dW_cpu;
    delete[] h_dx_cpu;
    delete[] h_dW_gpu;
    delete[] h_dx_gpu;
    delete[] h_x_bf16;
    delete[] h_weight_bf16;

    return 0;
}
