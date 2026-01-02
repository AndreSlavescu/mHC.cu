#pragma once

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace mhc {

__global__ void flush_l2_kernel(float* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = buf[idx] + 1.0f;
    }
}

struct L2Flusher {
    static constexpr int L2_SIZE_BYTES = 50 * 1024 * 1024;
    static constexpr int FLUSH_SIZE = L2_SIZE_BYTES / sizeof(float) * 2;
    float* buf;

    L2Flusher() : buf(nullptr) {
        cudaMalloc(&buf, FLUSH_SIZE * sizeof(float));
        cudaMemset(buf, 0, FLUSH_SIZE * sizeof(float));
    }

    ~L2Flusher() {
        if (buf)
            cudaFree(buf);
    }

    void flush() {
        int block_size = 256;
        int num_blocks = (FLUSH_SIZE + block_size - 1) / block_size;
        flush_l2_kernel<<<num_blocks, block_size>>>(buf, FLUSH_SIZE);
        cudaDeviceSynchronize();
    }
};

inline float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

inline float mean_abs_diff(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum / n;
}

inline bool check_test(float max_diff, float tolerance, const char* test_name = nullptr) {
    if (test_name) {
        printf("%s: ", test_name);
    }
    printf("max diff = %e, ", max_diff);
    if (max_diff < tolerance) {
        printf("PASSED (tol: %e)\n", tolerance);
        return true;
    } else {
        printf("FAILED (tol: %e)\n", tolerance);
        return false;
    }
}

struct BenchTimer {
    cudaEvent_t start, stop;

    BenchTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~BenchTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() { cudaEventRecord(start); }

    void record_stop() { cudaEventRecord(stop); }

    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

inline void print_bandwidth(size_t total_bytes, float time_ms) {
    float bw = (total_bytes / 1e9f) / (time_ms / 1e3f);
    printf("  Time: %.2f us, Bandwidth: %.0f GB/s\n", time_ms * 1000.0f, bw);
}

enum ProfilerTag {
    TagSetup = 0,
    TagLoad,
    TagCompute,
    TagReduce,
    TagStore,
    TagSync,
    TagOther,
    TagCount
};

inline const char* profiler_tag_name(ProfilerTag tag) {
    switch (tag) {
    case TagSetup:
        return "Setup";
    case TagLoad:
        return "Load";
    case TagCompute:
        return "Compute";
    case TagReduce:
        return "Reduce";
    case TagStore:
        return "Store";
    case TagSync:
        return "Sync";
    case TagOther:
        return "Other";
    default:
        return "Unknown";
    }
}

__device__ __forceinline__ int64_t globaltimer() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}

__device__ __forceinline__ int get_smid() {
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
}

struct DeviceProfiler {
    int64_t* data_ptr;
    int sm_id;
    int cnt;
    int max_entries;

    __device__ void init(int num_entries, int64_t* buffer, int block_id) {
        max_entries = num_entries;
        data_ptr = buffer + block_id * (1 + num_entries * 4);
        sm_id = get_smid();
        cnt = 0;
    }

    __device__ void start(ProfilerTag tag) {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 0] = sm_id;
        data_ptr[1 + cnt * 4 + 1] = tag;
        data_ptr[1 + cnt * 4 + 2] = globaltimer();
    }

    __device__ void stop() {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 3] = globaltimer() - data_ptr[1 + cnt * 4 + 2];
        cnt++;
    }

    __device__ void flush() { data_ptr[0] = cnt; }
};

struct ProfilerEntry {
    int sm_id;
    ProfilerTag tag;
    int64_t start_time;
    int64_t duration_ns;
};

struct HostProfiler {
    int64_t* d_buffer;
    int64_t* h_buffer;
    int num_blocks;
    int max_entries_per_block;
    size_t buffer_size;

    HostProfiler(int num_blocks_, int max_entries_per_block_)
        : num_blocks(num_blocks_), max_entries_per_block(max_entries_per_block_) {
        buffer_size = num_blocks * (1 + max_entries_per_block * 4) * sizeof(int64_t);
        cudaMalloc(&d_buffer, buffer_size);
        cudaMemset(d_buffer, 0, buffer_size);
        h_buffer = (int64_t*)malloc(buffer_size);
    }

    ~HostProfiler() {
        if (d_buffer)
            cudaFree(d_buffer);
        if (h_buffer)
            free(h_buffer);
    }

    int64_t* device_ptr() { return d_buffer; }

    void download() { cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost); }

    void print_summary() {
        download();

        int64_t tag_totals[TagCount] = {0};
        int tag_counts[TagCount] = {0};
        int64_t min_start = INT64_MAX;
        int64_t max_end = 0;

        int entry_stride = 1 + max_entries_per_block * 4;

        for (int b = 0; b < num_blocks; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            for (int e = 0; e < num_entries; e++) {
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t start = block_data[1 + e * 4 + 2];
                int64_t duration = block_data[1 + e * 4 + 3];

                if (tag < TagCount) {
                    tag_totals[tag] += duration;
                    tag_counts[tag]++;
                }

                if (start < min_start)
                    min_start = start;
                if (start + duration > max_end)
                    max_end = start + duration;
            }
        }

        int64_t wall_time = max_end - min_start;
        printf("\nProfiler Summary (%d blocks, %d max entries/block)\n", num_blocks,
               max_entries_per_block);
        printf("==================================================\n");
        printf("Total wall time: %.2f us\n\n", wall_time / 1000.0f);
        printf("%-10s %10s %10s %10s\n", "Phase", "Total(us)", "Count", "Avg(us)");
        printf("------------------------------------------\n");

        for (int t = 0; t < TagCount; t++) {
            if (tag_counts[t] > 0) {
                float total_us = tag_totals[t] / 1000.0f;
                float avg_us = total_us / tag_counts[t];
                printf("%-10s %10.2f %10d %10.2f\n", profiler_tag_name((ProfilerTag)t), total_us,
                       tag_counts[t], avg_us);
            }
        }
    }

    void print_timeline(int max_blocks = 4) {
        download();

        int entry_stride = 1 + max_entries_per_block * 4;
        int blocks_to_print = (max_blocks < num_blocks) ? max_blocks : num_blocks;

        printf("\nTimeline (first %d blocks)\n", blocks_to_print);
        printf("==========================\n");

        for (int b = 0; b < blocks_to_print; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            printf("\nBlock %d (%d events):\n", b, num_entries);

            for (int e = 0; e < num_entries; e++) {
                int sm_id = (int)block_data[1 + e * 4 + 0];
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t duration = block_data[1 + e * 4 + 3];

                printf("  SM%02d: %-10s %8.2f us\n", sm_id, profiler_tag_name((ProfilerTag)tag),
                       duration / 1000.0f);
            }
        }
    }
};
} // namespace mhc
