#pragma once

// CUDA-to-HIP compatibility shim for the standalone CMake harness.
//
// The mHC kernels are written in CUDA spelling and shared by two build systems:
// the PyTorch CUDAExtension (where torch.utils.hipify renames cuda*/cublas*
// symbols at build time) and this standalone CMake harness compiled directly by
// hipcc. For the harness, the cuda_stubs/ headers redirect every
// <cuda_runtime.h>/<cuda_bf16.h>/<cuda_fp16.h>/<cublasLt.h>/<cublas_v2.h> and the
// <cooperative_groups/reduce.h> include to this file, which pulls in the real
// ROCm headers and maps the handful of symbols the kernels use onto their HIP
// equivalents. The torch path does not see this file; hipify handles it there.
//
// PDL (Hopper programmatic dependent launch) has no HIP equivalent and is forced
// off here; the non-PDL launch path is the functional baseline.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#ifdef MHC_ENABLE_PDL
#undef MHC_ENABLE_PDL
#endif

// --- bf16 types and intrinsics ---
#ifndef nv_bfloat16
#define nv_bfloat16 __hip_bfloat16
#endif
#ifndef nv_bfloat162
#define nv_bfloat162 __hip_bfloat162
#endif

// --- runtime types ---
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDataType_t hipDataType

// --- runtime calls ---
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
// hipFuncSetAttribute takes const void*; templated kernel pointers do not convert
// implicitly under clang overload resolution, so wrap with an explicit cast.
template<typename FuncT>
static inline hipError_t cudaFuncSetAttribute(FuncT func, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

// --- scalar data type enums ---
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F

// --- cuBLAS / cuBLASLt -> hipBLAS / hipBLASLt ---
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cublasComputeType_t hipblasComputeType_t
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
// AMD has no TF32; full fp32 accumulate is the closest correctness-safe mapping.
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT

#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtOrder_t hipblasLtOrder_t
#define CUBLASLT_ORDER_ROW HIPBLASLT_ORDER_ROW
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_MATRIX_LAYOUT_ORDER HIPBLASLT_MATRIX_LAYOUT_ORDER
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute hipblasLtMatmulDescSetAttribute
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute

// hipBLASLt does not honor HIPBLASLT_ORDER_ROW the way cuBLASLt does, so the
// kernels' row-major matmuls produce wrong results. We instead run hipBLASLt in
// its native column-major mode. The identity used: a row-major layout (r,c,ld)
// is identical to a column-major layout (c,r,ld) (the transpose view). So
// cublasLtMatrixLayoutCreate swaps rows<->cols, the ORDER_ROW set-attribute is
// dropped (left COL), and cublasLtMatmul swaps the A/B operands together with
// their TRANSA/TRANSB so that C = op(A)*op(B) still lands in row-major C. This is
// a uniform rewrite that holds for every call site (forward, both backward
// matmuls, and the dynamic-H projection).
static inline hipblasStatus_t cublasLtMatmulDescCreate(hipblasLtMatmulDesc_t* desc,
                                                       hipblasComputeType_t computeType,
                                                       hipDataType scaleType) {
    return hipblasLtMatmulDescCreate(desc, computeType, scaleType);
}
static inline hipblasStatus_t cublasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* layout,
                                                         hipDataType type, uint64_t rows,
                                                         uint64_t cols, int64_t ld) {
    return hipblasLtMatrixLayoutCreate(layout, type, cols, rows, ld);
}
static inline hipblasStatus_t
cublasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t layout,
                                 hipblasLtMatrixLayoutAttribute_t attr, const void* buf,
                                 size_t bytes) {
    if (attr == HIPBLASLT_MATRIX_LAYOUT_ORDER) {
        return HIPBLAS_STATUS_SUCCESS; // stay column-major
    }
    return hipblasLtMatrixLayoutSetAttribute(layout, attr, buf, bytes);
}
static inline hipblasStatus_t cublasLtMatmul(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, const void* alpha, const void* A,
    hipblasLtMatrixLayout_t Adesc, const void* B, hipblasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, hipblasLtMatrixLayout_t Cdesc, void* D, hipblasLtMatrixLayout_t Ddesc,
    const hipblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSize, hipStream_t stream) {
    hipblasOperation_t opA, opB;
    size_t sz = sizeof(opA);
    hipblasLtMatmulDescGetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA), &sz);
    hipblasLtMatmulDescGetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB), &sz);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));
    hipblasStatus_t st = hipblasLtMatmul(handle, desc, alpha, B, Bdesc, A, Adesc, beta, C, Cdesc, D,
                                         Ddesc, algo, workspace, workspaceSize, stream);
    // restore original orientation so a cached descriptor stays valid across calls
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    return st;
}
// The heuristic must see the same swapped operands as the matmul; intercept it too.
static inline hipblasStatus_t
cublasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                               hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
                               hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
                               hipblasLtMatmulPreference_t pref, int requestedCount,
                               hipblasLtMatmulHeuristicResult_t* results, int* returnedCount) {
    hipblasOperation_t opA, opB;
    size_t sz = sizeof(opA);
    hipblasLtMatmulDescGetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA), &sz);
    hipblasLtMatmulDescGetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB), &sz);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));
    hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
        handle, desc, Bdesc, Adesc, Cdesc, Ddesc, pref, requestedCount, results, returnedCount);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    return st;
}

// --- cooperative_groups::reduce / plus ---
// HIP's cooperative_groups ships tiled_partition/thread_block_tile and shfl_xor
// but not the CUDA cg::reduce free function. cg::plus was absent before ROCm 7.14;
// guard it so we do not redefine it on newer SDKs that ship it natively.
// The reduction walks tile.size() (32 for the logical tiles the kernels use), so it
// is correct on both wave32 and wave64 wavefronts.
#include <hip/hip_cooperative_groups.h>
namespace cooperative_groups {
// HIP_VERSION = MAJOR*10000000 + MINOR*100000 + PATCH; 7.14.x >= 71400000.
#if !defined(HIP_VERSION) || (HIP_VERSION < 71400000)
template<typename T> struct plus {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};
#endif
// CUDA cg::reduce is an all-reduce: every lane receives the full result. Use a
// butterfly (shfl_xor) over tile.size() (32 for the logical tiles here) so the
// result is identical in all lanes on both wave32 and wave64.
template<typename Group, typename T, typename Op>
__device__ __forceinline__ T reduce(const Group& g, T val, Op op) {
    for (unsigned int offset = g.size() / 2; offset > 0; offset /= 2) {
        val = op(val, g.shfl_xor(val, offset));
    }
    return val;
}
} // namespace cooperative_groups

#else // NVIDIA path: leave the CUDA spelling untouched.

#include <cuda_runtime.h>

#endif
