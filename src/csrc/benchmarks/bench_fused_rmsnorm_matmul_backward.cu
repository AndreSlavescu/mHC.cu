#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.cuh"
#include "../kernels/fused_rmsnorm_matmul.cuh"

using namespace mhc;

int main() {
    const int bench_runs = 100;

    L2Flusher flusher;

    struct Config {
        int M;
        int N;
        int K;
    };

    Config configs[] = {
        {128, 4096, 4096},  {256, 4096, 4096},  {512, 4096, 4096},  {1024, 4096, 4096},
        {2048, 4096, 4096}, {1024, 8192, 4096}, {2048, 8192, 4096},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Fused RMSNorm + MatMul Backward Benchmark\n");
    printf("==========================================================================\n");
    printf("%8s %8s %8s %12s %12s %12s\n", "M", "N", "K", "Time (us)", "TFLOPS",
           "Bandwidth (GB/s)");
    printf("--------------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int M = configs[c].M;
        int N = configs[c].N;
        int K = configs[c].K;

        floatX* h_inp = (floatX*)malloc(M * K * sizeof(floatX));
        floatX* h_weight = (floatX*)malloc(N * K * sizeof(floatX));
        float* h_grad = (float*)malloc(M * N * sizeof(float));
        float* h_rms = (float*)malloc(M * sizeof(float));

        srand(42);
        for (int i = 0; i < M * K; i++) {
            h_inp[i] = (floatX)((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
        for (int i = 0; i < N * K; i++) {
            h_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
        }
        for (int i = 0; i < M * N; i++) {
            h_grad[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
        for (int i = 0; i < M; i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < K; j++) {
                float v = (float)h_inp[i * K + j];
                sum_sq += v * v;
            }
            h_rms[i] = sqrtf(sum_sq / (float)K + 1e-5f);
        }

        floatX *d_inp, *d_weight;
        float *d_grad, *d_rms, *d_dW, *d_dx;
        CHECK_CUDA(cudaMalloc(&d_inp, M * K * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_weight, N * K * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_grad, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rms, M * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dW, N * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dx, M * K * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, M * K * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, N * K * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_grad, h_grad, M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_rms, h_rms, M * sizeof(float), cudaMemcpyHostToDevice));

        FusedRMSNormMatmulBackward backward;
        backward.init(M, N, K);

        double flops = 4.0 * (double)M * (double)N * (double)K;

        size_t bytes_read = M * K * sizeof(floatX) + N * K * sizeof(floatX) +
                            M * N * sizeof(float) + M * sizeof(float);
        size_t bytes_write = N * K * sizeof(float) + M * K * sizeof(float);
        size_t total_bytes = bytes_read + bytes_write;

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();
            CHECK_CUDA(cudaMemset(d_dW, 0, N * K * sizeof(float)));

            timer.record_start();
            backward.backward(d_dW, d_dx, d_grad, d_inp, d_weight, d_rms);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float tflops = (flops / 1e12f) / (avg_time_ms / 1e3f);
        float bw = (total_bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.2f %12.2f\n", M, N, K, avg_time_ms * 1000.0f, tflops, bw);

        backward.destroy();
        cudaFree(d_inp);
        cudaFree(d_weight);
        cudaFree(d_grad);
        cudaFree(d_rms);
        cudaFree(d_dW);
        cudaFree(d_dx);
        free(h_inp);
        free(h_weight);
        free(h_grad);
        free(h_rms);
    }

    return 0;
}
