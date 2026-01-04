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
#include "../kernels/rmsnorm.cuh"

using namespace mhc;

void rmsnorm_backward_cpu(float* d_inp, float* d_weight, const float* grad, const float* inp,
                          const float* weight, const float* rms, int N, int C) {
    for (int i = 0; i < C; i++) {
        d_weight[i] = 0.0f;
    }

    for (int row = 0; row < N; row++) {
        float r = rms[row];
        float r_inv = 1.0f / r;

        float dot_sum = 0.0f;
        for (int i = 0; i < C; i++) {
            dot_sum += grad[row * C + i] * weight[i] * inp[row * C + i];
        }
        float correction = dot_sum / ((float)C * r * r);

        for (int i = 0; i < C; i++) {
            d_inp[row * C + i] =
                (grad[row * C + i] * weight[i] * r_inv) - (inp[row * C + i] * correction * r_inv);
            d_weight[i] += grad[row * C + i] * inp[row * C + i] * r_inv;
        }
    }
}

int main() {
    printf("rmsnorm_backward Test\n");
    printf("=====================\n\n");

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const int N = 16;
    const int C = 64;

    printf("N=%d, C=%d\n", N, C);

    float* h_inp = new float[N * C];
    float* h_weight = new float[C];
    float* h_grad = new float[N * C];
    float* h_rms = new float[N];
    float* h_d_inp_cpu = new float[N * C];
    float* h_d_weight_cpu = new float[C];
    float* h_d_inp_gpu = new float[N * C];
    float* h_d_weight_gpu = new float[C];
    floatX* h_inp_bf16 = new floatX[N * C];
    floatX* h_weight_bf16 = new floatX[C];

    for (int i = 0; i < N * C; i++) {
        h_inp[i] = dist(gen);
        h_inp_bf16[i] = (floatX)h_inp[i];
    }
    for (int i = 0; i < C; i++) {
        h_weight[i] = 1.0f + 0.1f * dist(gen);
        h_weight_bf16[i] = (floatX)h_weight[i];
    }
    for (int i = 0; i < N * C; i++)
        h_grad[i] = dist(gen);

    for (int row = 0; row < N; row++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < C; i++) {
            sum_sq += h_inp[row * C + i] * h_inp[row * C + i];
        }
        h_rms[row] = sqrtf(sum_sq / (float)C + 1e-5f);
    }

    rmsnorm_backward_cpu(h_d_inp_cpu, h_d_weight_cpu, h_grad, h_inp, h_weight, h_rms, N, C);

    floatX *d_inp_bf16, *d_weight_bf16;
    float *d_grad, *d_rms, *d_d_inp, *d_d_weight;

    CHECK_CUDA(cudaMalloc(&d_inp_bf16, N * C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_weight_bf16, C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_grad, N * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rms, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_inp, N * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_weight, C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp_bf16, h_inp_bf16, N * C * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_weight_bf16, h_weight_bf16, C * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad, h_grad, N * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rms, h_rms, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_d_weight, 0, C * sizeof(float)));

    rmsnorm_backward(d_d_inp, d_d_weight, d_grad, d_inp_bf16, d_weight_bf16, d_rms, N, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_d_inp_gpu, d_d_inp, N * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_d_weight_gpu, d_d_weight, C * sizeof(float), cudaMemcpyDeviceToHost));

    float d_inp_diff = max_abs_diff(h_d_inp_gpu, h_d_inp_cpu, N * C);
    float d_weight_diff = max_abs_diff(h_d_weight_gpu, h_d_weight_cpu, C);

#if DEBUG
    printf("\nSample d_inp (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_inp_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_inp_cpu[i]);
    printf("\n");

    printf("\nSample d_weight (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_weight_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_weight_cpu[i]);
    printf("\n");
#endif

    printf("\n");
    check_test(d_inp_diff, 0.02f, "d_inp");
    check_test(d_weight_diff, 0.02f, "d_weight");

    cudaFree(d_inp_bf16);
    cudaFree(d_weight_bf16);
    cudaFree(d_grad);
    cudaFree(d_rms);
    cudaFree(d_d_inp);
    cudaFree(d_d_weight);
    delete[] h_inp;
    delete[] h_weight;
    delete[] h_grad;
    delete[] h_rms;
    delete[] h_d_inp_cpu;
    delete[] h_d_weight_cpu;
    delete[] h_d_inp_gpu;
    delete[] h_d_weight_gpu;
    delete[] h_inp_bf16;
    delete[] h_weight_bf16;

    return 0;
}
