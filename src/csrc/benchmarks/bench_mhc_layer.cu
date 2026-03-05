#include <cstdio>
#include <cuda_runtime.h>

#include "bench_harness.cuh"
#include "mhc_layer.cuh"

using namespace mhc;

void run_static_benchmark(int B, int C, int n, int bench_runs, L2Flusher& flusher, bool use_tc) {
    HostMem<float> h_x_expanded(B * n * C);
    HostMem<floatX> h_rmsnorm_weight(C);
    HostMem<float> h_H_pre(n);
    HostMem<float> h_H_post(n);
    HostMem<float> h_H_res(n * n);

    fill_random(h_x_expanded, B * n * C);
    fill_random_bf16(h_rmsnorm_weight, C, 0.75f, 1.25f);

    for (int i = 0; i < n; i++) {
        h_H_pre.ptr[i] = 0.0f;
        h_H_post.ptr[i] = 0.0f;
    }
    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_H_res.ptr[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    DeviceMem<float> d_x_expanded(B * n * C);
    d_x_expanded.upload(h_x_expanded);

    MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = n;
    cfg.sinkhorn_iters = 20;
    cfg.eps = 1e-5f;
    cfg.use_pdl = true;
    cfg.use_dynamic_h = false;

    MHCLayer layer;
    layer.init(cfg);
    layer.use_tc_mix = use_tc;
    layer.set_weights(h_rmsnorm_weight, h_H_pre, h_H_post, h_H_res);
    layer.sync();

    layer.forward_device(d_x_expanded);
    layer.sync();

    size_t bytes_io = (size_t)B * n * C * sizeof(float) * 3;

    float avg_time_ms =
        bench_kernel([&]() { layer.forward_device(d_x_expanded); }, bench_runs, flusher);

    float throughput = B / (avg_time_ms / 1000.0f);
    float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

    printf("%6d %6d %4d %8s %6s %12.2f %14.0f %14.0f\n", B, C, n, "static",
           use_tc ? "TC" : "CUDA CORE", avg_time_ms * 1000.0f, throughput, bw);

    layer.destroy();
}

void run_dynamic_benchmark(int B, int C, int n, int bench_runs, L2Flusher& flusher) {
    int nC = n * C;
    int total_H_dim = n + n + n * n;

    HostMem<float> h_x_expanded(B * n * C);
    HostMem<floatX> h_rmsnorm_weight(C);
    HostMem<floatX> h_phi(total_H_dim * nC);
    HostMem<float> h_b_pre(n);
    HostMem<float> h_b_post(n);
    HostMem<float> h_b_res(n * n);

    fill_random(h_x_expanded, B * n * C);
    fill_random_bf16(h_rmsnorm_weight, C, 0.75f, 1.25f);
    fill_random_bf16(h_phi, total_H_dim * nC, -0.05f, 0.05f, 43);

    for (int i = 0; i < n; i++) {
        h_b_pre.ptr[i] = 0.0f;
        h_b_post.ptr[i] = 0.0f;
    }
    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_b_res.ptr[i] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    DeviceMem<float> d_x_expanded(B * n * C);
    d_x_expanded.upload(h_x_expanded);

    MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = n;
    cfg.sinkhorn_iters = 20;
    cfg.eps = 1e-5f;
    cfg.use_pdl = true;
    cfg.use_dynamic_h = true;

    MHCLayer layer;
    layer.init(cfg);

    floatX* phi_base = h_phi;
    floatX* h_phi_pre = phi_base;
    floatX* h_phi_post = phi_base + n * nC;
    floatX* h_phi_res = phi_base + 2 * n * nC;

    layer.set_weights_dynamic(h_rmsnorm_weight, h_phi_pre, h_phi_post, h_phi_res, h_b_pre, h_b_post,
                              h_b_res, 0.01f, 0.01f, 0.01f);
    layer.sync();

    layer.forward_device(d_x_expanded);
    layer.sync();

    size_t bytes_io = (size_t)B * n * C * sizeof(float) * 3;

    float avg_time_ms =
        bench_kernel([&]() { layer.forward_device(d_x_expanded); }, bench_runs, flusher);

    float throughput = B / (avg_time_ms / 1000.0f);
    float bw = (bytes_io / 1e9f) / (avg_time_ms / 1e3f);

    printf("%6d %6d %4d %8s %6s %12.2f %14.0f %14.0f\n", B, C, n, "dynamic", "CUDA",
           avg_time_ms * 1000.0f, throughput, bw);

    layer.destroy();
}

int main() {
    const int bench_runs = 50;

    L2Flusher flusher;

    struct Config {
        int B;
        int C;
        int n;
    };

    Config configs[] = {
        {64, 1280, 4},  {128, 1280, 4}, {256, 1280, 4},  {320, 1280, 4},
        {64, 1920, 4},  {128, 1920, 4}, {64, 2560, 4},   {128, 2560, 4},
        {32, 1280, 32}, {64, 1280, 32}, {128, 1280, 32},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("MHC Layer End-to-End Benchmark\n");
    printf("==========================================================\n");
    printf("Pipeline: Aggregate(H_pre) -> RMSNorm -> Distribute(H_post) + Mix(M)\n");
    printf("Static H: H coefficients shared across batch (Sinkhorn on H_res)\n");
    printf("Dynamic H: H coefficients computed per-sample via RMSNorm + MatMul + activations\n");
    printf("Input shape: [B, n, C]\n");
    printf("PDL path: %s\n\n",
#ifdef MHC_ENABLE_PDL
           "Enabled"
#else
           "Disabled"
#endif
    );

    printf("%6s %6s %4s %8s %6s %12s %14s %14s\n", "Batch", "Hidden", "n", "Mode", "Implementation",
           "Time (us)", "Samples/sec", "Bandwidth (GB/s)");
    printf(
        "--------------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int B = configs[c].B;
        int C = configs[c].C;
        int n = configs[c].n;

        run_static_benchmark(B, C, n, bench_runs, flusher, false);

        if (n >= 32) {
            run_static_benchmark(B, C, n, bench_runs, flusher, true);
        }

        run_dynamic_benchmark(B, C, n, bench_runs, flusher);

        printf("\n");
    }

    return 0;
}
