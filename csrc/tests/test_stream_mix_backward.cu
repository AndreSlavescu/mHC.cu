#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

void stream_mix_backward_cpu(float* d_inp, float* d_M, const float* grad, const float* inp,
                             const float* M, int B, int n, int C) {
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < n; j++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    sum += M[i * n + j] * grad[b * n * C + i * C + c];
                }
                d_inp[b * n * C + j * C + c] = sum;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < C; c++) {
                    sum += grad[b * n * C + i * C + c] * inp[b * n * C + j * C + c];
                }
            }
            d_M[i * n + j] = sum;
        }
    }
}

int main() {
    printf("stream_mix_backward Test\n");
    printf("========================\n\n");

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const int B = 16;
    const int n = 4;
    const int C = 64;

    printf("B=%d, n=%d, C=%d\n", B, n, C);

    float* h_inp = new float[B * n * C];
    float* h_M = new float[n * n];
    float* h_grad = new float[B * n * C];
    float* h_d_inp_cpu = new float[B * n * C];
    float* h_d_M_cpu = new float[n * n];
    float* h_d_inp_gpu = new float[B * n * C];
    float* h_d_M_gpu = new float[n * n];

    for (int i = 0; i < B * n * C; i++)
        h_inp[i] = dist(gen);
    for (int i = 0; i < n * n; i++)
        h_M[i] = dist(gen);
    for (int i = 0; i < B * n * C; i++)
        h_grad[i] = dist(gen);

    stream_mix_backward_cpu(h_d_inp_cpu, h_d_M_cpu, h_grad, h_inp, h_M, B, n, C);

    float *d_inp, *d_M, *d_grad, *d_d_inp, *d_d_M;
    CHECK_CUDA(cudaMalloc(&d_inp, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_inp, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_M, n * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad, h_grad, B * n * C * sizeof(float), cudaMemcpyHostToDevice));

    stream_mix_backward(d_d_inp, d_d_M, d_grad, d_inp, d_M, B, n, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_d_inp_gpu, d_d_inp, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_d_M_gpu, d_d_M, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    float d_inp_diff = max_abs_diff(h_d_inp_gpu, h_d_inp_cpu, B * n * C);
    float d_M_diff = max_abs_diff(h_d_M_gpu, h_d_M_cpu, n * n);

    printf("\n");
    check_test(d_inp_diff, 1e-5f, "d_inp");
    check_test(d_M_diff, 1e-3f, "d_M");

    cudaFree(d_inp);
    cudaFree(d_M);
    cudaFree(d_grad);
    cudaFree(d_d_inp);
    cudaFree(d_d_M);
    delete[] h_inp;
    delete[] h_M;
    delete[] h_grad;
    delete[] h_d_inp_cpu;
    delete[] h_d_M_cpu;
    delete[] h_d_inp_gpu;
    delete[] h_d_M_gpu;

    return 0;
}
