#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "mhc_types.h"
#include "utils.cuh"

namespace mhc {

// RAII wrapper for device memory (cudaMalloc / cudaFree).
template<typename T> struct DeviceMem {
    T* ptr;
    size_t count;

    DeviceMem() : ptr(nullptr), count(0) {}

    explicit DeviceMem(size_t n) : ptr(nullptr), count(n) {
        CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T)));
    }

    ~DeviceMem() {
        if (ptr)
            cudaFree(ptr);
    }

    DeviceMem(const DeviceMem&) = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;

    DeviceMem(DeviceMem&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr;
        o.count = 0;
    }

    DeviceMem& operator=(DeviceMem&& o) noexcept {
        if (this != &o) {
            if (ptr)
                cudaFree(ptr);
            ptr = o.ptr;
            count = o.count;
            o.ptr = nullptr;
            o.count = 0;
        }
        return *this;
    }

    void upload(const T* host_data) {
        CHECK_CUDA(cudaMemcpy(ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void zero() { CHECK_CUDA(cudaMemset(ptr, 0, count * sizeof(T))); }

    // cppcheck-suppress unusedFunction
    size_t bytes() const { return count * sizeof(T); }

    operator T*() { return ptr; }             // cppcheck-suppress noExplicitConstructor
    operator const T*() const { return ptr; } // cppcheck-suppress noExplicitConstructor
};

// RAII wrapper for host memory (malloc / free).
template<typename T> struct HostMem {
    T* ptr;
    size_t count;

    HostMem() : ptr(nullptr), count(0) {}

    explicit HostMem(size_t n) : ptr(nullptr), count(n) { ptr = (T*)malloc(n * sizeof(T)); }

    ~HostMem() {
        if (ptr)
            free(ptr);
    }

    HostMem(const HostMem&) = delete;
    HostMem& operator=(const HostMem&) = delete;

    HostMem(HostMem&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr;
        o.count = 0;
    }

    HostMem& operator=(HostMem&& o) noexcept {
        if (this != &o) {
            if (ptr)
                free(ptr);
            ptr = o.ptr;
            count = o.count;
            o.ptr = nullptr;
            o.count = 0;
        }
        return *this;
    }

    size_t bytes() const { return count * sizeof(T); }

    operator T*() { return ptr; }             // cppcheck-suppress noExplicitConstructor
    operator const T*() const { return ptr; } // cppcheck-suppress noExplicitConstructor
};

// Run a kernel lambda `runs` times with L2 flush. Returns average time in ms.
template<typename KernelFn> float bench_kernel(KernelFn fn, int runs, L2Flusher& flusher) {
    BenchTimer timer;
    float total_time = 0.0f;

    for (int i = 0; i < runs; i++) {
        flusher.flush();
        timer.record_start();
        fn();
        timer.record_stop();
        total_time += timer.elapsed_ms();
    }
    return total_time / runs;
}

// Run with per-iteration setup (e.g. zeroing gradient buffers).
template<typename KernelFn, typename PreFn>
float bench_kernel(KernelFn fn, int runs, L2Flusher& flusher, PreFn pre) {
    BenchTimer timer;
    float total_time = 0.0f;

    for (int i = 0; i < runs; i++) {
        flusher.flush();
        pre();
        timer.record_start();
        fn();
        timer.record_stop();
        total_time += timer.elapsed_ms();
    }
    return total_time / runs;
}

// Run a profiled kernel once and print phase breakdown.
// KernelFn signature: void(int64_t* profiler_buf, int max_entries)
template<typename KernelFn>
void profile_kernel(KernelFn fn, int num_blocks, int max_entries, L2Flusher& flusher,
                    bool print_timeline = false, int timeline_blocks = 4) {
    HostProfiler profiler(num_blocks, max_entries);
    flusher.flush();
    fn(profiler.device_ptr(), max_entries);
    CHECK_CUDA(cudaDeviceSynchronize());
    profiler.print_summary();
    if (print_timeline) {
        profiler.print_timeline(timeline_blocks);
    }
}

// Fill buffer with random floats in [lo, hi].
inline void fill_random(float* buf, int n, float lo = -1.0f, float hi = 1.0f, int seed = 42) {
    srand(seed);
    float range = hi - lo;
    for (int i = 0; i < n; i++) {
        buf[i] = lo + (float)rand() / RAND_MAX * range;
    }
}

// Fill buffer with random bf16 values in [lo, hi].
inline void fill_random_bf16(floatX* buf, int n, float lo = -1.0f, float hi = 1.0f, int seed = 42) {
    srand(seed);
    float range = hi - lo;
    for (int i = 0; i < n; i++) {
        buf[i] = (floatX)(lo + (float)rand() / RAND_MAX * range);
    }
}

} // namespace mhc
