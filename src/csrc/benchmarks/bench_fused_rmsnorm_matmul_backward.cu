#include <cstdio>
#include <cuda_runtime.h>

#include "../include/bench_harness.cuh"
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

        HostMem<floatX> h_inp(M * K);
        HostMem<floatX> h_weight(N * K);
        HostMem<float> h_grad(M * N);
        HostMem<float> h_rms(M);

        fill_random_bf16(h_inp, M * K);
        fill_random_bf16(h_weight, N * K, 0.75f, 1.25f);
        fill_random(h_grad, M * N, -1.0f, 1.0f, 43);

        for (int i = 0; i < M; i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < K; j++) {
                float v = (float)h_inp.ptr[i * K + j];
                sum_sq += v * v;
            }
            h_rms.ptr[i] = sqrtf(sum_sq / (float)K + 1e-5f);
        }

        DeviceMem<floatX> d_inp(M * K);
        DeviceMem<floatX> d_weight(N * K);
        DeviceMem<float> d_grad(M * N);
        DeviceMem<float> d_rms(M);
        DeviceMem<float> d_dW(N * K);
        DeviceMem<float> d_dx(M * K);

        d_inp.upload(h_inp);
        d_weight.upload(h_weight);
        d_grad.upload(h_grad);
        d_rms.upload(h_rms);

        FusedRMSNormMatmulBackward backward;
        backward.init(M, N, K);

        double flops = 4.0 * (double)M * (double)N * (double)K;

        size_t bytes_read = M * K * sizeof(floatX) + N * K * sizeof(floatX) +
                            M * N * sizeof(float) + M * sizeof(float);
        size_t bytes_write = N * K * sizeof(float) + M * K * sizeof(float);
        size_t total_bytes = bytes_read + bytes_write;

        float avg_time_ms =
            bench_kernel([&]() { backward.backward(d_dW, d_dx, d_grad, d_inp, d_weight, d_rms); },
                         bench_runs, flusher, [&]() { d_dW.zero(); });

        float tflops = (flops / 1e12f) / (avg_time_ms / 1e3f);
        float bw = (total_bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %8d %12.2f %12.2f %12.2f\n", M, N, K, avg_time_ms * 1000.0f, tflops, bw);

        backward.destroy();
    }

    return 0;
}
