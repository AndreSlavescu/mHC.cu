#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

void stream_aggregate_backward_cpu(float* d_inp, float* d_H_pre, const float* grad,
                                   const float* inp, const float* H_pre, int B, int n, int C) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                d_inp[b * n * C + i * C + c] = grad[b * C + c] * H_pre[i];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                sum += grad[b * C + c] * inp[b * n * C + i * C + c];
            }
        }
        d_H_pre[i] = sum;
    }
}

int main() {
    printf("stream_aggregate_backward Test\n");
    printf("==============================\n\n");

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const int B = 16;
    const int n = 4;
    const int C = 64;

    printf("B=%d, n=%d, C=%d\n", B, n, C);

    float* h_inp = new float[B * n * C];
    float* h_H_pre = new float[n];
    float* h_grad = new float[B * C];
    float* h_d_inp_cpu = new float[B * n * C];
    float* h_d_H_pre_cpu = new float[n];
    float* h_d_inp_gpu = new float[B * n * C];
    float* h_d_H_pre_gpu = new float[n];

    for (int i = 0; i < B * n * C; i++)
        h_inp[i] = dist(gen);
    for (int i = 0; i < n; i++)
        h_H_pre[i] = dist(gen);
    for (int i = 0; i < B * C; i++)
        h_grad[i] = dist(gen);

    stream_aggregate_backward_cpu(h_d_inp_cpu, h_d_H_pre_cpu, h_grad, h_inp, h_H_pre, B, n, C);

    float *d_inp, *d_H_pre, *d_grad, *d_d_inp, *d_d_H_pre;
    CHECK_CUDA(cudaMalloc(&d_inp, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_pre, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_inp, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_H_pre, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_pre, h_H_pre, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad, h_grad, B * C * sizeof(float), cudaMemcpyHostToDevice));

    stream_aggregate_backward(d_d_inp, d_d_H_pre, d_grad, d_inp, d_H_pre, B, n, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_d_inp_gpu, d_d_inp, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_d_H_pre_gpu, d_d_H_pre, n * sizeof(float), cudaMemcpyDeviceToHost));

    float d_inp_diff = max_abs_diff(h_d_inp_gpu, h_d_inp_cpu, B * n * C);
    float d_H_pre_diff = max_abs_diff(h_d_H_pre_gpu, h_d_H_pre_cpu, n);

    printf("\n");
    check_test(d_inp_diff, 1e-5f, "d_inp");
    check_test(d_H_pre_diff, 1e-4f, "d_H_pre");

    cudaFree(d_inp);
    cudaFree(d_H_pre);
    cudaFree(d_grad);
    cudaFree(d_d_inp);
    cudaFree(d_d_H_pre);
    delete[] h_inp;
    delete[] h_H_pre;
    delete[] h_grad;
    delete[] h_d_inp_cpu;
    delete[] h_d_H_pre_cpu;
    delete[] h_d_inp_gpu;
    delete[] h_d_H_pre_gpu;

    return 0;
}
