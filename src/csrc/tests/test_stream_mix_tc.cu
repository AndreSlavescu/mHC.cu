#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "stream_ops.cuh"
#include "mhc_types.h"
#include "utils.h"

using namespace mhc;

void stream_mix_cpu_reference(float* out, const float* inp, const float* M, int B, int n, int C) {
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    int src = b * n * C + j * C + c;
                    sum += M[i * n + j] * inp[src];
                }
                out[b * n * C + i * C + c] = sum;
            }
        }
    }
}

int main() {
    const int B = 32;
    const int n = 32;
    const int C = 256;

    printf("Stream Mix TC Test\n");
    printf("===========================================\n");
    printf("Shape: [%d, %d, %d] = [B, n, C]\n\n", B, n, C);

    float* h_inp = (float*)malloc(B * n * C * sizeof(float));
    float* h_M = (float*)malloc(n * n * sizeof(float));
    float* h_out_cpu = (float*)malloc(B * n * C * sizeof(float));
    float* h_out_gpu = (float*)malloc(B * n * C * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++) {
        h_inp[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < n * n; i++) {
        h_M[i] = (float)rand() / RAND_MAX;
    }

    stream_mix_cpu_reference(h_out_cpu, h_inp, h_M, B, n, C);

    float *d_inp, *d_M, *d_out;
    CHECK_CUDA(cudaMalloc(&d_inp, B * n * C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, B * n * C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, B * n * C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M, h_M, n * n * sizeof(float), cudaMemcpyHostToDevice));

    StreamMixTC mixer;
    mixer.init(B, n, C);

    printf("Running TC stream mix...\n");
    mixer.forward(d_out, d_inp, d_M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, B * n * C * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(h_out_cpu, h_out_gpu, B * n * C);

#if DEBUG
    printf("Sample outputs (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_cpu[i]);
    printf("\n\n");
#endif

    check_test(max_diff, 5e-3f, "Stream Mix TC (n=32, TF32)");

    mixer.destroy();
    cudaFree(d_inp);
    cudaFree(d_M);
    cudaFree(d_out);
    free(h_inp);
    free(h_M);
    free(h_out_cpu);
    free(h_out_gpu);

    return 0;
}
