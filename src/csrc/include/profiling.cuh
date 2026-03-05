#pragma once

#include "utils.cuh"

namespace mhc {

// Phase profiling macros for production CUDA kernels.
//
// Usage: Add to __global__ kernels templated with `bool DO_PROFILE = false`.
// Kernel must accept `int64_t* profiler_buf, int max_entries` as final params.
//
// When DO_PROFILE=false, `if constexpr` eliminates all profiling code at compile
// time — zero register pressure, zero instructions, zero overhead.
//
// Example:
//   template<int BLOCK_SIZE, bool DO_PROFILE = false>
//   __global__ void my_kernel(float* out, const float* inp, int N,
//                             int64_t* profiler_buf, int max_entries) {
//       MHC_PROFILE_INIT(blockIdx.x);
//       MHC_PROFILE_START(TagLoad);
//       // ... load phase ...
//       MHC_PROFILE_PHASE(TagCompute);
//       // ... compute phase ...
//       MHC_PROFILE_PHASE(TagStore);
//       // ... store phase ...
//       MHC_PROFILE_END();
//   }

// Declare DeviceProfiler and initialize it for the given block.
#define MHC_PROFILE_INIT(block_id)                                                                 \
    DeviceProfiler _mhc_profiler;                                                                  \
    if constexpr (DO_PROFILE) {                                                                    \
        if (threadIdx.x == 0) {                                                                    \
            _mhc_profiler.init(max_entries, profiler_buf, block_id);                               \
        }                                                                                          \
    }

// Begin a new profiling phase. Use for the first phase after INIT.
#define MHC_PROFILE_START(tag)                                                                     \
    if constexpr (DO_PROFILE) {                                                                    \
        if (threadIdx.x == 0) {                                                                    \
            _mhc_profiler.start(tag);                                                              \
        }                                                                                          \
    }

// Transition from one phase to the next (stop current, start new).
#define MHC_PROFILE_PHASE(tag)                                                                     \
    if constexpr (DO_PROFILE) {                                                                    \
        if (threadIdx.x == 0) {                                                                    \
            _mhc_profiler.stop();                                                                  \
            _mhc_profiler.start(tag);                                                              \
        }                                                                                          \
    }

// Stop the final phase and flush profiling data.
#define MHC_PROFILE_END()                                                                          \
    if constexpr (DO_PROFILE) {                                                                    \
        if (threadIdx.x == 0) {                                                                    \
            _mhc_profiler.stop();                                                                  \
            _mhc_profiler.flush();                                                                 \
        }                                                                                          \
    }

} // namespace mhc
