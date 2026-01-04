#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "rmsnorm.cuh"
#include "mhc_types.h"
#include "utils.cuh"

using namespace mhc;

void rmsnorm_cpu_reference(float* out, const float* inp, const float* weight, int N, int C,
                           float eps) {
    for (int i = 0; i < N; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = inp[i * C + j];
            sum_sq += val * val;
        }
        float rms_inv = 1.0f / sqrtf(sum_sq / (float)C + eps);
        for (int j = 0; j < C; j++) {
            out[i * C + j] = inp[i * C + j] * rms_inv * weight[j];
        }
    }
}

int main() {
    const int N = 128;
    const int C = 4096;
    const float eps = 1e-5f;

    float* h_inp = (float*)malloc(N * C * sizeof(float));
    float* h_weight = (float*)malloc(C * sizeof(float));
    float* h_out_ref = (float*)malloc(N * C * sizeof(float));
    float* h_out_gpu = (float*)malloc(N * C * sizeof(float));

    srand(42);
    for (int i = 0; i < N * C; i++) {
        h_inp[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < C; i++) {
        h_weight[i] = (float)rand() / RAND_MAX * 0.5f + 0.75f;
    }

    floatX* h_inp_bf16 = (floatX*)malloc(N * C * sizeof(floatX));
    floatX* h_weight_bf16 = (floatX*)malloc(C * sizeof(floatX));
    floatX* h_out_bf16 = (floatX*)malloc(N * C * sizeof(floatX));

    for (int i = 0; i < N * C; i++) {
        h_inp_bf16[i] = (floatX)h_inp[i];
        h_inp[i] = (float)h_inp_bf16[i];
    }
    for (int i = 0; i < C; i++) {
        h_weight_bf16[i] = (floatX)h_weight[i];
        h_weight[i] = (float)h_weight_bf16[i];
    }

    rmsnorm_cpu_reference(h_out_ref, h_inp, h_weight, N, C, eps);

    floatX *d_inp, *d_weight, *d_out;
    CHECK_CUDA(cudaMalloc(&d_inp, N * C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_weight, C * sizeof(floatX)));
    CHECK_CUDA(cudaMalloc(&d_out, N * C * sizeof(floatX)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp_bf16, N * C * sizeof(floatX), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight_bf16, C * sizeof(floatX), cudaMemcpyHostToDevice));

    rmsnorm_forward(d_out, d_inp, d_weight, N, C, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_bf16, d_out, N * C * sizeof(floatX), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N * C; i++) {
        h_out_gpu[i] = (float)h_out_bf16[i];
    }

    float max_diff = max_abs_diff(h_out_ref, h_out_gpu, N * C);

#if DEBUG
    printf("Sample outputs (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.4f ", h_out_ref[i]);
    printf("\n\n");
#endif

    check_test(max_diff, 1e-2f, "RMSNorm");

    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_out);
    free(h_inp);
    free(h_weight);
    free(h_out_ref);
    free(h_out_gpu);
    free(h_inp_bf16);
    free(h_weight_bf16);
    free(h_out_bf16);

    return 0;
}
