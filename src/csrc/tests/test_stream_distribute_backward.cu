#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "../include/mhc_types.h"
#include "../include/utils.h"
#include "../kernels/stream_ops.cuh"

using namespace mhc;

void stream_distribute_backward_cpu(float* d_inp, float* d_H_post, const float* grad,
                                    const float* inp, const float* H_post, int B, int n, int C) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += grad[b * n * C + i * C + c] * H_post[i];
            }
            d_inp[b * C + c] = sum;
        }
    }

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                sum += grad[b * n * C + i * C + c] * inp[b * C + c];
            }
        }
        d_H_post[i] = sum;
    }
}

int main() {
    printf("stream_distribute_backward Test\n");
    printf("================================\n\n");

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    const int B = 16;
    const int n = 4;
    const int C = 64;

    printf("B=%d, n=%d, C=%d\n", B, n, C);

    float* h_inp = new float[B * C];
    float* h_H_post = new float[n];
    float* h_grad = new float[B * n * C];
    float* h_d_inp_cpu = new float[B * C];
    float* h_d_H_post_cpu = new float[n];
    float* h_d_inp_gpu = new float[B * C];
    float* h_d_H_post_gpu = new float[n];

    for (int i = 0; i < B * C; i++)
        h_inp[i] = dist(gen);
    for (int i = 0; i < n; i++)
        h_H_post[i] = dist(gen);
    for (int i = 0; i < B * n * C; i++)
        h_grad[i] = dist(gen);

    stream_distribute_backward_cpu(h_d_inp_cpu, h_d_H_post_cpu, h_grad, h_inp, h_H_post, B, n, C);

    float *d_inp, *d_H_post, *d_grad, *d_d_inp, *d_d_H_post;
    CHECK_CUDA(cudaMalloc(&d_inp, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_post, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_inp, B * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d_H_post, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, B * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H_post, h_H_post, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad, h_grad, B * n * C * sizeof(float), cudaMemcpyHostToDevice));

    stream_distribute_backward(d_d_inp, d_d_H_post, d_grad, d_inp, d_H_post, B, n, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_d_inp_gpu, d_d_inp, B * C * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_d_H_post_gpu, d_d_H_post, n * sizeof(float), cudaMemcpyDeviceToHost));

    float d_inp_diff = max_abs_diff(h_d_inp_gpu, h_d_inp_cpu, B * C);
    float d_H_post_diff = max_abs_diff(h_d_H_post_gpu, h_d_H_post_cpu, n);

#if DEBUG
    printf("\nSample d_inp (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_inp_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_d_inp_cpu[i]);
    printf("\n");
#endif

    printf("\n");
    check_test(d_inp_diff, 1e-5f, "d_inp");
    check_test(d_H_post_diff, 1e-3f, "d_H_post");

    cudaFree(d_inp);
    cudaFree(d_H_post);
    cudaFree(d_grad);
    cudaFree(d_d_inp);
    cudaFree(d_d_H_post);
    delete[] h_inp;
    delete[] h_H_post;
    delete[] h_grad;
    delete[] h_d_inp_cpu;
    delete[] h_d_H_post_cpu;
    delete[] h_d_inp_gpu;
    delete[] h_d_H_post_gpu;

    return 0;
}
