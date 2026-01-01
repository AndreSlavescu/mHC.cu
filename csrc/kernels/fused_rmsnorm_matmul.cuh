#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "rmsnorm.cuh"

namespace cg = cooperative_groups;

namespace mhc {

struct MatmulDescriptors {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc;
    cublasLtMatrixLayout_t B_desc;
    cublasLtMatrixLayout_t C_desc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
};

inline void init_matmul_descriptors(MatmulDescriptors& desc, int M, int N, int K,
                                    size_t workspace_size = 32 * 1024 * 1024) {
    CHECK_CUBLAS(cublasLtCreate(&desc.handle));

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t ab_type = CUDA_R_16BF;
    cudaDataType_t c_type = CUDA_R_16BF;
    cudaDataType_t scale_type = CUDA_R_32F;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&desc.matmul_desc, compute_type, scale_type));

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(desc.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(desc.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &trans_b, sizeof(trans_b)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.A_desc, ab_type, M, K, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.B_desc, ab_type, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&desc.C_desc, c_type, M, N, M));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&desc.preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(desc.preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size, sizeof(workspace_size)));

    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        desc.handle, desc.matmul_desc, desc.A_desc, desc.B_desc, desc.C_desc, desc.C_desc,
        desc.preference, 1, &desc.heuristic, &returned_results));

    if (returned_results == 0) {
        fprintf(stderr, "No cuBLASLt algorithm found\n");
        exit(EXIT_FAILURE);
    }

    desc.workspace_size = workspace_size;
    CHECK_CUDA(cudaMalloc(&desc.workspace, workspace_size));
}

inline void destroy_matmul_descriptors(MatmulDescriptors& desc) {
    cublasLtMatmulPreferenceDestroy(desc.preference);
    cublasLtMatrixLayoutDestroy(desc.A_desc);
    cublasLtMatrixLayoutDestroy(desc.B_desc);
    cublasLtMatrixLayoutDestroy(desc.C_desc);
    cublasLtMatmulDescDestroy(desc.matmul_desc);
    cublasLtDestroy(desc.handle);
    cudaFree(desc.workspace);
}

inline void matmul_forward(MatmulDescriptors& desc, floatX* out, const floatX* A, const floatX* B,
                           float alpha, float beta, cudaStream_t stream = nullptr) {
    CHECK_CUBLAS(cublasLtMatmul(desc.handle, desc.matmul_desc, &alpha, A, desc.A_desc, B,
                                desc.B_desc, &beta, out, desc.C_desc, out, desc.C_desc,
                                &desc.heuristic.algo, desc.workspace, desc.workspace_size, stream));
}

struct FusedRMSNormMatmul {
    MatmulDescriptors matmul_desc;
    floatX* norm_buffer;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmul() : norm_buffer(nullptr), rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        CHECK_CUDA(cudaMalloc(&norm_buffer, M * K * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            cudaFree(norm_buffer);
            cudaFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(floatX* out, const floatX* inp, const floatX* weight, const floatX* proj_weight,
                 cudaStream_t stream = nullptr) {
        rmsnorm_forward_with_rms(norm_buffer, rms_buffer, inp, weight, M, K, eps, stream);
        matmul_forward(matmul_desc, out, norm_buffer, proj_weight, 1.0f, 0.0f, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

template<int BLOCK_SIZE>
__global__ void rmsnorm_pdl_kernel(floatX* __restrict__ out, float* __restrict__ rms_out,
                                   const floatX* __restrict__ inp,
                                   const floatX* __restrict__ weight, int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;
    floatX* o = out + idx * C;

    extern __shared__ float shared[];
    float* s_sum_sq = shared;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        thread_sum_sq += val * val;
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
            float rms = sqrtf(block_sum / (float)C + eps);
            float rms_inv = 1.0f / rms;
            s_sum_sq[0] = rms_inv;
            if (rms_out)
                rms_out[idx] = rms;
        }
    }
    __syncthreads();

    float rms_inv = s_sum_sq[0];

#if __CUDA_ARCH__ >= 900
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = (float)x[i];
        float w = (float)weight[i];
        o[i] = (floatX)(val * rms_inv * w);
    }
}

inline void rmsnorm_forward_pdl(floatX* out, float* rms_out, const floatX* inp,
                                const floatX* weight, int N, int C, float eps,
                                cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 512;
    int num_warps = BLOCK_SIZE / 32;
    size_t shared_mem = num_warps * sizeof(float);

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);

#ifdef MHC_ENABLE_PDL
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_mem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 1;

    CHECK_CUDA(cudaLaunchKernelEx(&config, rmsnorm_pdl_kernel<BLOCK_SIZE>, out, rms_out, inp,
                                  weight, N, C, eps));
#else
    rmsnorm_pdl_kernel<BLOCK_SIZE>
        <<<grid, block, shared_mem, stream>>>(out, rms_out, inp, weight, N, C, eps);
#endif
}

struct FusedRMSNormMatmulPDL {
    MatmulDescriptors matmul_desc;
    floatX* norm_buffer;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmulPDL() : norm_buffer(nullptr), rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        CHECK_CUDA(cudaMalloc(&norm_buffer, M * K * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            cudaFree(norm_buffer);
            cudaFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(floatX* out, const floatX* inp, const floatX* weight, const floatX* proj_weight,
                 cudaStream_t stream = nullptr) {
        rmsnorm_forward_pdl(norm_buffer, rms_buffer, inp, weight, M, K, eps, stream);
        matmul_forward(matmul_desc, out, norm_buffer, proj_weight, 1.0f, 0.0f, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};
} // namespace mhc
