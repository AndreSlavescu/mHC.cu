#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "sinkhorn_knopp.cuh"
#include "mhc_types.h"
#include "utils.h"

using namespace mhc;

void sinkhorn_knopp_cpu_reference(float* out, const float* inp, int M, int N, int num_iters,
                                  float eps) {
    for (int i = 0; i < M * N; i++) {
        out[i] = inp[i];
    }

    for (int iter = 0; iter < num_iters; iter++) {
        for (int r = 0; r < M; r++) {
            float row_sum = 0.0f;
            for (int c = 0; c < N; c++) {
                row_sum += out[r * N + c];
            }
            if (row_sum > eps) {
                for (int c = 0; c < N; c++) {
                    out[r * N + c] /= row_sum;
                }
            }
        }

        for (int c = 0; c < N; c++) {
            float col_sum = 0.0f;
            for (int r = 0; r < M; r++) {
                col_sum += out[r * N + c];
            }
            if (col_sum > eps) {
                for (int r = 0; r < M; r++) {
                    out[r * N + c] /= col_sum;
                }
            }
        }
    }
}

float check_row_sums(const float* mat, int M, int N) {
    float max_err = 0.0f;
    for (int r = 0; r < M; r++) {
        float sum = 0.0f;
        for (int c = 0; c < N; c++) {
            sum += mat[r * N + c];
        }
        float err = fabsf(sum - 1.0f);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

float check_col_sums(const float* mat, int M, int N) {
    float max_err = 0.0f;
    for (int c = 0; c < N; c++) {
        float sum = 0.0f;
        for (int r = 0; r < M; r++) {
            sum += mat[r * N + c];
        }
        float err = fabsf(sum - 1.0f);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

int main() {
    const int M = 32;
    const int N = 32;
    const int num_iters = 10;
    const float eps = 1e-8f;

    float* h_inp = (float*)malloc(M * N * sizeof(float));
    float* h_out_ref = (float*)malloc(M * N * sizeof(float));
    float* h_out_gpu = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * N; i++) {
        h_inp[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    sinkhorn_knopp_cpu_reference(h_out_ref, h_inp, M, N, num_iters, eps);

    float *d_inp, *d_out;
    CHECK_CUDA(cudaMalloc(&d_inp, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, M * N * sizeof(float), cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward(d_out, d_inp, M, N, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(h_out_ref, h_out_gpu, M * N);
    float row_err = check_row_sums(h_out_gpu, M, N);
    float col_err = check_col_sums(h_out_gpu, M, N);

    printf("Sinkhorn-Knopp (M=%d, N=%d, iters=%d):\n", M, N, num_iters);
    printf("  CPU diff: %e, row err: %e, col err: %e\n", max_diff, row_err, col_err);

#if DEBUG
    printf("Sample outputs (first 10):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.6f ", h_out_gpu[i]);
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++)
        printf("%.6f ", h_out_ref[i]);
    printf("\n\n");
#endif

    const float tolerance = 1e-5f;
    const float doubly_stochastic_tol = 1e-4f;

    bool passed = (max_diff < tolerance && row_err < doubly_stochastic_tol &&
                   col_err < doubly_stochastic_tol);
    check_test(passed ? 0.0f : 1.0f, 0.5f, "32x32");

    printf("\nTesting 64x64...\n");
    const int M2 = 64, N2 = 64;
    float* h_inp2 = (float*)malloc(M2 * N2 * sizeof(float));
    float* h_out2 = (float*)malloc(M2 * N2 * sizeof(float));

    for (int i = 0; i < M2 * N2; i++) {
        h_inp2[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    float *d_inp2, *d_out2;
    CHECK_CUDA(cudaMalloc(&d_inp2, M2 * N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out2, M2 * N2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_inp2, h_inp2, M2 * N2 * sizeof(float), cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward(d_out2, d_inp2, M2, N2, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out2, d_out2, M2 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    float row_err2 = check_row_sums(h_out2, M2, N2);
    float col_err2 = check_col_sums(h_out2, M2, N2);

    printf("  row err: %e, col err: %e\n", row_err2, col_err2);
    bool passed2 = (row_err2 < doubly_stochastic_tol && col_err2 < doubly_stochastic_tol);
    check_test(passed2 ? 0.0f : 1.0f, 0.5f, "64x64");

    cudaFree(d_inp);
    cudaFree(d_out);
    cudaFree(d_inp2);
    cudaFree(d_out2);
    free(h_inp);
    free(h_out_ref);
    free(h_out_gpu);
    free(h_inp2);
    free(h_out2);

    return 0;
}
