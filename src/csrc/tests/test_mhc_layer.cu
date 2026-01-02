#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#include "mhc_layer.cuh"
#include "mhc_types.h"
#include "utils.h"

using namespace mhc;

float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void rmsnorm_cpu(float* out, const float* inp, const floatX* weight, int B, int C, float eps) {
    for (int b = 0; b < B; b++) {
        float sum_sq = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = inp[b * C + c];
            sum_sq += v * v;
        }
        float rms = sqrtf(sum_sq / C + eps);
        float rms_inv = 1.0f / rms;
        for (int c = 0; c < C; c++) {
            out[b * C + c] = inp[b * C + c] * rms_inv * (float)weight[c];
        }
    }
}

void mhc_layer_cpu_reference(float* out, const float* x_expanded, const float* H_pre,
                             const float* H_post, const float* M, const floatX* rmsnorm_weight,
                             int B, int n, int C, float eps) {
    float* x_agg = (float*)malloc(B * C * sizeof(float));
    float* x_normed = (float*)malloc(B * C * sizeof(float));
    float* y_dist = (float*)malloc(B * n * C * sizeof(float));
    float* x_mixed = (float*)malloc(B * n * C * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                int src = b * n * C + i * C + c;
                sum += H_pre[i] * x_expanded[src];
            }
            x_agg[b * C + c] = sum;
        }
    }

    rmsnorm_cpu(x_normed, x_agg, rmsnorm_weight, B, C, eps);

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                int dst = b * n * C + i * C + c;
                y_dist[dst] = H_post[i] * x_normed[b * C + c];
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    int src = b * n * C + j * C + c;
                    sum += M[i * n + j] * x_expanded[src];
                }
                x_mixed[b * n * C + i * C + c] = sum;
            }
        }
    }

    for (int i = 0; i < B * n * C; i++) {
        out[i] = x_mixed[i] + y_dist[i];
    }

    free(x_agg);
    free(x_normed);
    free(y_dist);
    free(x_mixed);
}

void sinkhorn_cpu(float* M, int n, int iters) {
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < n; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n; j++) {
                row_sum += M[i * n + j];
            }
            if (row_sum > 1e-8f) {
                for (int j = 0; j < n; j++) {
                    M[i * n + j] /= row_sum;
                }
            }
        }
        for (int j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                col_sum += M[i * n + j];
            }
            if (col_sum > 1e-8f) {
                for (int i = 0; i < n; i++) {
                    M[i * n + j] /= col_sum;
                }
            }
        }
    }
}

int main() {
    const int B = 32;
    const int C = 128;
    const int n = 4;

    printf("MHC Layer Integration Test (Expanded Residual Stream)\n");
    printf("======================================================\n");
    printf("Batch: %d, Hidden: %d, Expansion: %d\n", B, C, n);
    printf("Input shape: [%d, %d, %d] = [B, n, C]\n\n", B, n, C);

    float* h_x_expanded = (float*)malloc(B * n * C * sizeof(float));
    floatX* h_rmsnorm_weight = (floatX*)malloc(C * sizeof(floatX));
    float* h_H_pre = (float*)malloc(n * sizeof(float));
    float* h_H_post = (float*)malloc(n * sizeof(float));
    float* h_H_res = (float*)malloc(n * n * sizeof(float));
    float* h_M = (float*)malloc(n * n * sizeof(float));
    float* h_out_gpu = (float*)malloc(B * n * C * sizeof(float));
    float* h_out_ref = (float*)malloc(B * n * C * sizeof(float));

    float* h_H_pre_activated = (float*)malloc(n * sizeof(float));
    float* h_H_post_activated = (float*)malloc(n * sizeof(float));

    srand(42);
    for (int i = 0; i < B * n * C; i++) {
        h_x_expanded[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < C; i++) {
        h_rmsnorm_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
    }
    for (int i = 0; i < n; i++) {
        h_H_pre[i] = 0.0f;
        h_H_post[i] = 0.0f;
        h_H_pre_activated[i] = sigmoid_cpu(h_H_pre[i]);
        h_H_post_activated[i] = 2.0f * sigmoid_cpu(h_H_post[i]);
    }
    for (int i = 0; i < n * n; i++) {
        h_H_res[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        h_M[i] = expf(h_H_res[i]);
    }

    sinkhorn_cpu(h_M, n, 20);
    mhc_layer_cpu_reference(h_out_ref, h_x_expanded, h_H_pre_activated, h_H_post_activated, h_M,
                            h_rmsnorm_weight, B, n, C, 1e-5f);

    MHCLayerConfig config;
    config.batch_size = B;
    config.hidden_dim = C;
    config.expansion_rate = n;
    config.sinkhorn_iters = 20;
    config.eps = 1e-5f;
    config.use_pdl = true;

    printf("Initializing MHC Layer...\n");
    MHCLayer layer;
    layer.init(config);

    printf("Setting weights...\n");
    layer.set_weights(h_rmsnorm_weight, h_H_pre, h_H_post, h_H_res);
    layer.sync();

    printf("Running forward pass...\n");
    layer.forward(h_x_expanded);
    layer.sync();

    CHECK_CUDA(cudaMemcpy(h_out_gpu, layer.get_output(), B * n * C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("\nForward pass completed!\n");

#if DEBUG
    printf("\nSample outputs (first 10 elements):\n");
    printf("  GPU: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_out_gpu[i]);
    }
    printf("\n  CPU: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_out_ref[i]);
    }
    printf("\n");
#endif

    float max_diff = max_abs_diff(h_out_ref, h_out_gpu, B * n * C);

    printf("\nOutput shape: [%d, %d, %d]\n", B, n, C);
    check_test(max_diff, 0.1f, "MHC Layer");

    layer.destroy();

    free(h_x_expanded);
    free(h_rmsnorm_weight);
    free(h_H_pre);
    free(h_H_post);
    free(h_H_pre_activated);
    free(h_H_post_activated);
    free(h_H_res);
    free(h_M);
    free(h_out_gpu);
    free(h_out_ref);

    return 0;
}
