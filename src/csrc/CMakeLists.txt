cmake_minimum_required(VERSION 3.18)
project(mHC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(mhc_kernels INTERFACE)
target_include_directories(mhc_kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(mhc_kernels INTERFACE CUDA::cudart CUDA::cublasLt)

option(MHC_ENABLE_PDL "Enable PDL for H100/B200 (only available for SM_90 and SM_100)" ON)

function(add_mhc_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mhc_kernels)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
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
