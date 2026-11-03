cmake_minimum_required(VERSION 3.18)

option(USE_HIP "Build the kernels for ROCm/HIP instead of CUDA" OFF)

if(USE_HIP)
    project(mHC LANGUAGES CXX HIP)
else()
    project(mHC LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

add_library(mhc_kernels INTERFACE)
target_include_directories(mhc_kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

if(USE_HIP)
    # project(... LANGUAGES CXX HIP) above already honored an explicit
    # -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detected the host GPU(s), and
    # otherwise errored on a build host with no AMD GPU.
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipblaslt REQUIRED)
    # cuda_stubs/ redirects every <cuda_runtime.h>/<cublasLt.h>/... include to the
    # cuda_to_hip.h shim; it must sit ahead of any real CUDA headers on the path.
    target_include_directories(mhc_kernels INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_stubs)
    target_link_libraries(mhc_kernels INTERFACE hip::host roc::hipblas roc::hipblaslt)
    # PDL is Hopper-only and has no HIP equivalent; force it off.
    set(MHC_ENABLE_PDL OFF CACHE BOOL "Enable PDL (Hopper-only; unsupported on HIP)" FORCE)
else()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mhc_kernels INTERFACE CUDA::cudart CUDA::cublasLt)
    option(MHC_ENABLE_PDL "Enable PDL for H100/B200 (only available for SM_90 and SM_100)" ON)
endif()

function(add_mhc_executable name source)
    if(USE_HIP)
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    endif()
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mhc_kernels)
    if(USE_HIP)
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>)
    else()
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()
    if(MHC_ENABLE_PDL)
        target_compile_definitions(${name} PRIVATE MHC_ENABLE_PDL)
    endif()
endfunction()

set(TESTS
    test_rmsnorm
    test_rmsnorm_backward
    test_sinkhorn_knopp
    test_mhc_layer
    test_stream_ops
    test_stream_ops_backward
    test_fused_rmsnorm_matmul
    test_fused_rmsnorm_matmul_backward
)

set(BENCHMARKS
    bench_rmsnorm
    bench_rmsnorm_backward
    bench_sinkhorn_knopp
    bench_sinkhorn_knopp_backward
    bench_mhc_layer
    bench_stream_ops
    bench_stream_ops_backward
    bench_fused_rmsnorm_matmul
    bench_fused_rmsnorm_matmul_backward
)

foreach(test ${TESTS})
    add_mhc_executable(${test} tests/${test}.cu)
endforeach()

foreach(bench ${BENCHMARKS})
    add_mhc_executable(${bench} benchmarks/${bench}.cu)
endforeach()
