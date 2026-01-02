#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../include/mhc_types.h"
#include "../include/utils.h"

namespace cg = cooperative_groups;
using namespace mhc;

template<int BLOCK_SIZE, bool DO_PROFILE>
__global__ void rmsnorm_profiled_kernel(floatX* __restrict__ out, const floatX* __restrict__ inp,
                                        const floatX* __restrict__ weight, int N, int C, float eps,
                                        int64_t* profiler_buf, int max_entries) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    DeviceProfiler profiler;
    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.init(max_entries, profiler_buf, idx);
            profiler.start(TagLoad);
        }
    }

    const floatX* x = inp + idx * C;
    floatX* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
    }

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
            profiler.start(TagReduce);
        }
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = BLOCK_SIZE / 32;

    if (lane_id == 0) {
        s_sum_sq[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = rsqrtf(block_sum / (float)C + eps);
            s_sum_sq[0] = rms;
        }
    }
    __syncthreads();

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
            profiler.start(TagStore);
        }
    }

    float rms_inv = s_sum_sq[0];

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (floatX)(val * rms_inv * w);
    }

    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) {
            profiler.stop();
            profiler.flush();
        }
    }
}

int main() {
    const int bench_runs = 100;
    const float eps = 1e-5f;
    const int max_entries = 8;

    L2Flusher flusher;

    int configs[][2] = {
        {128, 4096},  {256, 4096},  {512, 4096},  {1024, 4096},
        {2048, 4096}, {1024, 8192}, {2048, 8192},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("RMSNorm Benchmark\n");
    printf("====================================\n");
    printf("%8s %8s %12s %12s\n", "N", "C", "Time (us)", "Bandwidth (GB/s)");
    printf("---------------------------------------------------\n");

    for (int c = 0; c < num_configs; c++) {
        int N = configs[c][0];
        int C = configs[c][1];

        floatX* h_inp = (floatX*)malloc(N * C * sizeof(floatX));
        floatX* h_weight = (floatX*)malloc(C * sizeof(floatX));

        srand(42);
        for (int i = 0; i < N * C; i++) {
            h_inp[i] = (floatX)((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
        for (int i = 0; i < C; i++) {
            h_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
        }

        floatX *d_inp, *d_weight, *d_out;
        CHECK_CUDA(cudaMalloc(&d_inp, N * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_weight, C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_out, N * C * sizeof(floatX)));

        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, N * C * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, C * sizeof(floatX), cudaMemcpyHostToDevice));

        size_t bytes_read = (size_t)N * C * sizeof(floatX) + (size_t)C * sizeof(floatX);
        size_t bytes_written = (size_t)N * C * sizeof(floatX);
        size_t total_bytes = bytes_read + bytes_written;

        constexpr int BLOCK_SIZE = 512;
        int num_warps = BLOCK_SIZE / 32;
        size_t shared_mem = num_warps * sizeof(float);

        BenchTimer timer;
        float total_time = 0.0f;

        for (int i = 0; i < bench_runs; i++) {
            flusher.flush();

            timer.record_start();
            rmsnorm_profiled_kernel<BLOCK_SIZE, false>
                <<<N, BLOCK_SIZE, shared_mem>>>(d_out, d_inp, d_weight, N, C, eps, nullptr, 0);
            timer.record_stop();
            total_time += timer.elapsed_ms();
        }

        float avg_time_ms = total_time / bench_runs;
        float bw = (total_bytes / 1e9f) / (avg_time_ms / 1e3f);

        printf("%8d %8d %12.2f %12.0f\n", N, C, avg_time_ms * 1000.0f, bw);

        cudaFree(d_inp);
        cudaFree(d_weight);
        cudaFree(d_out);
        free(h_inp);
        free(h_weight);
    }

    printf("\n--- Step Breakdown (N=1024, C=4096) ---\n");
    {
        int N = 1024, C = 4096;

        floatX* h_inp = (floatX*)malloc(N * C * sizeof(floatX));
        floatX* h_weight = (floatX*)malloc(C * sizeof(floatX));

        srand(42);
        for (int i = 0; i < N * C; i++) {
            h_inp[i] = (floatX)((float)rand() / RAND_MAX * 2.0f - 1.0f);
        }
        for (int i = 0; i < C; i++) {
            h_weight[i] = (floatX)((float)rand() / RAND_MAX * 0.5f + 0.75f);
        }

        floatX *d_inp, *d_weight, *d_out;
        CHECK_CUDA(cudaMalloc(&d_inp, N * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_weight, C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&d_out, N * C * sizeof(floatX)));

        CHECK_CUDA(cudaMemcpy(d_inp, h_inp, N * C * sizeof(floatX), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight, h_weight, C * sizeof(floatX), cudaMemcpyHostToDevice));

        constexpr int BLOCK_SIZE = 512;
        int num_warps = BLOCK_SIZE / 32;
        size_t shared_mem = num_warps * sizeof(float);

        HostProfiler profiler(N, max_entries);

        flusher.flush();
        rmsnorm_profiled_kernel<BLOCK_SIZE, true><<<N, BLOCK_SIZE, shared_mem>>>(
            d_out, d_inp, d_weight, N, C, eps, profiler.device_ptr(), max_entries);
        CHECK_CUDA(cudaDeviceSynchronize());

        profiler.print_summary();

        cudaFree(d_inp);
        cudaFree(d_weight);
        cudaFree(d_out);
        free(h_inp);
        free(h_weight);
    }

    return 0;
}
