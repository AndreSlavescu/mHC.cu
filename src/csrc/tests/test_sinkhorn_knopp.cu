#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "sinkhorn_knopp.cuh"
#include "mhc_types.h"
#include "utils.cuh"

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

void sinkhorn_knopp_fused_exp_cpu_reference(float* out, float* H_res_exp, const float* H_res_raw,
                                            int M, int N, int num_iters, float eps) {
    for (int i = 0; i < M * N; i++) {
        H_res_exp[i] = expf(H_res_raw[i]);
        out[i] = H_res_exp[i];
    }
    sinkhorn_knopp_cpu_reference(out, out, M, N, num_iters, eps);
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

    printf("\nTesting fused_exp version...\n");
    const int M3 = 4, N3 = 4;
    float* h_H_res_raw = (float*)malloc(M3 * N3 * sizeof(float));
    float* h_H_res_exp_cpu = (float*)malloc(M3 * N3 * sizeof(float));
    float* h_out_cpu = (float*)malloc(M3 * N3 * sizeof(float));
    float* h_H_res_exp_gpu = (float*)malloc(M3 * N3 * sizeof(float));
    float* h_out_gpu3 = (float*)malloc(M3 * N3 * sizeof(float));

    for (int i = 0; i < M3 * N3; i++) {
        h_H_res_raw[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    sinkhorn_knopp_fused_exp_cpu_reference(h_out_cpu, h_H_res_exp_cpu, h_H_res_raw, M3, N3,
                                           num_iters, eps);

    float *d_H_res_raw, *d_H_res_exp, *d_out3;
    CHECK_CUDA(cudaMalloc(&d_H_res_raw, M3 * N3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_res_exp, M3 * N3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out3, M3 * N3 * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(d_H_res_raw, h_H_res_raw, M3 * N3 * sizeof(float), cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward_fused_exp(d_out3, d_H_res_exp, d_H_res_raw, M3, N3, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_gpu3, d_out3, M3 * N3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_H_res_exp_gpu, d_H_res_exp, M3 * N3 * sizeof(float), cudaMemcpyDeviceToHost));

    float exp_diff = max_abs_diff(h_H_res_exp_cpu, h_H_res_exp_gpu, M3 * N3);
    float out_diff = max_abs_diff(h_out_cpu, h_out_gpu3, M3 * N3);
    float row_err3 = check_row_sums(h_out_gpu3, M3, N3);
    float col_err3 = check_col_sums(h_out_gpu3, M3, N3);

    printf("  exp diff: %e, out diff: %e, row err: %e, col err: %e\n", exp_diff, out_diff, row_err3,
           col_err3);
    bool passed3 = (exp_diff < 1e-5f && out_diff < 1e-5f && row_err3 < doubly_stochastic_tol &&
                    col_err3 < doubly_stochastic_tol);
    check_test(passed3 ? 0.0f : 1.0f, 0.5f, "fused_exp 4x4");

    cudaFree(d_H_res_raw);
    cudaFree(d_H_res_exp);
    cudaFree(d_out3);
    free(h_H_res_raw);
    free(h_H_res_exp_cpu);
    free(h_out_cpu);
    free(h_H_res_exp_gpu);
    free(h_out_gpu3);

    printf("\nTesting batched sinkhorn (n=4, B=64)...\n");
    const int B_batch = 64, n_batch = 4;
    float* h_inp_batch = (float*)malloc(B_batch * n_batch * n_batch * sizeof(float));
    float* h_out_batch_cpu = (float*)malloc(B_batch * n_batch * n_batch * sizeof(float));
    float* h_out_batch_gpu = (float*)malloc(B_batch * n_batch * n_batch * sizeof(float));

    for (int i = 0; i < B_batch * n_batch * n_batch; i++) {
        h_inp_batch[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    for (int b = 0; b < B_batch; b++) {
        sinkhorn_knopp_cpu_reference(h_out_batch_cpu + b * n_batch * n_batch,
                                     h_inp_batch + b * n_batch * n_batch, n_batch, n_batch,
                                     num_iters, eps);
    }

    float *d_inp_batch, *d_out_batch;
    CHECK_CUDA(cudaMalloc(&d_inp_batch, B_batch * n_batch * n_batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_batch, B_batch * n_batch * n_batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_inp_batch, h_inp_batch, B_batch * n_batch * n_batch * sizeof(float),
                          cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward_batched(d_out_batch, d_inp_batch, B_batch, n_batch, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_batch_gpu, d_out_batch, B_batch * n_batch * n_batch * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float batch_diff = max_abs_diff(h_out_batch_cpu, h_out_batch_gpu, B_batch * n_batch * n_batch);

    float max_row_err_batch = 0.0f, max_col_err_batch = 0.0f;
    for (int b = 0; b < B_batch; b++) {
        float re = check_row_sums(h_out_batch_gpu + b * n_batch * n_batch, n_batch, n_batch);
        float ce = check_col_sums(h_out_batch_gpu + b * n_batch * n_batch, n_batch, n_batch);
        if (re > max_row_err_batch)
            max_row_err_batch = re;
        if (ce > max_col_err_batch)
            max_col_err_batch = ce;
    }

    printf("  CPU diff: %e, row err: %e, col err: %e\n", batch_diff, max_row_err_batch,
           max_col_err_batch);
    bool passed_batch = (batch_diff < 1e-5f && max_row_err_batch < doubly_stochastic_tol &&
                         max_col_err_batch < doubly_stochastic_tol);
    check_test(passed_batch ? 0.0f : 1.0f, 0.5f, "batched 4x4 B=64");

    cudaFree(d_inp_batch);
    cudaFree(d_out_batch);
    free(h_inp_batch);
    free(h_out_batch_cpu);
    free(h_out_batch_gpu);

    printf("\nTesting batched sinkhorn (n=8, B=32)...\n");
    const int B_batch2 = 32, n_batch2 = 8;
    float* h_inp_batch2 = (float*)malloc(B_batch2 * n_batch2 * n_batch2 * sizeof(float));
    float* h_out_batch2_cpu = (float*)malloc(B_batch2 * n_batch2 * n_batch2 * sizeof(float));
    float* h_out_batch2_gpu = (float*)malloc(B_batch2 * n_batch2 * n_batch2 * sizeof(float));

    for (int i = 0; i < B_batch2 * n_batch2 * n_batch2; i++) {
        h_inp_batch2[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    for (int b = 0; b < B_batch2; b++) {
        sinkhorn_knopp_cpu_reference(h_out_batch2_cpu + b * n_batch2 * n_batch2,
                                     h_inp_batch2 + b * n_batch2 * n_batch2, n_batch2, n_batch2,
                                     num_iters, eps);
    }

    float *d_inp_batch2, *d_out_batch2;
    CHECK_CUDA(cudaMalloc(&d_inp_batch2, B_batch2 * n_batch2 * n_batch2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_batch2, B_batch2 * n_batch2 * n_batch2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_inp_batch2, h_inp_batch2,
                          B_batch2 * n_batch2 * n_batch2 * sizeof(float), cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward_batched(d_out_batch2, d_inp_batch2, B_batch2, n_batch2, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_batch2_gpu, d_out_batch2,
                          B_batch2 * n_batch2 * n_batch2 * sizeof(float), cudaMemcpyDeviceToHost));

    float batch2_diff =
        max_abs_diff(h_out_batch2_cpu, h_out_batch2_gpu, B_batch2 * n_batch2 * n_batch2);

    float max_row_err_batch2 = 0.0f, max_col_err_batch2 = 0.0f;
    for (int b = 0; b < B_batch2; b++) {
        float re = check_row_sums(h_out_batch2_gpu + b * n_batch2 * n_batch2, n_batch2, n_batch2);
        float ce = check_col_sums(h_out_batch2_gpu + b * n_batch2 * n_batch2, n_batch2, n_batch2);
        if (re > max_row_err_batch2)
            max_row_err_batch2 = re;
        if (ce > max_col_err_batch2)
            max_col_err_batch2 = ce;
    }

    printf("  CPU diff: %e, row err: %e, col err: %e\n", batch2_diff, max_row_err_batch2,
           max_col_err_batch2);
    bool passed_batch2 = (batch2_diff < 1e-5f && max_row_err_batch2 < doubly_stochastic_tol &&
                          max_col_err_batch2 < doubly_stochastic_tol);
    check_test(passed_batch2 ? 0.0f : 1.0f, 0.5f, "batched 8x8 B=32");

    cudaFree(d_inp_batch2);
    cudaFree(d_out_batch2);
    free(h_inp_batch2);
    free(h_out_batch2_cpu);
    free(h_out_batch2_gpu);

    printf("\nTesting large batch sinkhorn (n=4, B=320)...\n");
    const int B_large = 320, n_large = 4;
    float* h_inp_large = (float*)malloc(B_large * n_large * n_large * sizeof(float));
    float* h_out_large_cpu = (float*)malloc(B_large * n_large * n_large * sizeof(float));
    float* h_out_large_gpu = (float*)malloc(B_large * n_large * n_large * sizeof(float));

    for (int i = 0; i < B_large * n_large * n_large; i++) {
        h_inp_large[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    for (int b = 0; b < B_large; b++) {
        sinkhorn_knopp_cpu_reference(h_out_large_cpu + b * n_large * n_large,
                                     h_inp_large + b * n_large * n_large, n_large, n_large,
                                     num_iters, eps);
    }

    float *d_inp_large, *d_out_large;
    CHECK_CUDA(cudaMalloc(&d_inp_large, B_large * n_large * n_large * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_large, B_large * n_large * n_large * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_inp_large, h_inp_large, B_large * n_large * n_large * sizeof(float),
                          cudaMemcpyHostToDevice));

    sinkhorn_knopp_forward_batched(d_out_large, d_inp_large, B_large, n_large, num_iters, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_large_gpu, d_out_large, B_large * n_large * n_large * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float large_diff = max_abs_diff(h_out_large_cpu, h_out_large_gpu, B_large * n_large * n_large);

    float max_row_err_large = 0.0f, max_col_err_large = 0.0f;
    for (int b = 0; b < B_large; b++) {
        float re = check_row_sums(h_out_large_gpu + b * n_large * n_large, n_large, n_large);
        float ce = check_col_sums(h_out_large_gpu + b * n_large * n_large, n_large, n_large);
        if (re > max_row_err_large)
            max_row_err_large = re;
        if (ce > max_col_err_large)
            max_col_err_large = ce;
    }

    printf("  CPU diff: %e, row err: %e, col err: %e\n", large_diff, max_row_err_large,
           max_col_err_large);
    bool passed_large = (large_diff < 1e-5f && max_row_err_large < doubly_stochastic_tol &&
                         max_col_err_large < doubly_stochastic_tol);
    check_test(passed_large ? 0.0f : 1.0f, 0.5f, "batched 4x4 B=320");

    cudaFree(d_inp_large);
    cudaFree(d_out_large);
    free(h_inp_large);
    free(h_out_large_cpu);
    free(h_out_large_gpu);

    return 0;
}
