#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../csrc/include/mhc_types.h"
#include "../csrc/kernels/rmsnorm.cuh"
#include "../csrc/kernels/sinkhorn_knopp.cuh"
#include "../csrc/kernels/stream_ops.cuh"
#include "../csrc/kernels/fused_rmsnorm_matmul.cuh"
#include "../csrc/include/utils.cuh"

using namespace mhc;

#define CHECK_TENSOR_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_TENSOR_CUDA(x);                                                                          \
    CHECK_CONTIGUOUS(x)

struct CublasLtCache {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t A_desc = nullptr;
    cublasLtMatrixLayout_t B_n_desc = nullptr;
    cublasLtMatrixLayout_t C_n_desc = nullptr;
    cublasLtMatrixLayout_t B_nn_desc = nullptr;
    cublasLtMatrixLayout_t C_nn_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic_n;
    cublasLtMatmulHeuristicResult_t heuristic_nn;
    void* workspace = nullptr;
    size_t workspace_size = 4 * 1024 * 1024;
    int cached_B = 0, cached_n = 0, cached_nC = 0;
    bool initialized = false;

    void init(int B, int n, int nC) {
        if (initialized && cached_B == B && cached_n == n && cached_nC == nC) {
            return;
        }
        destroy();

        cached_B = B;
        cached_n = n;
        cached_nC = nC;

        CHECK_CUBLAS(cublasLtCreate(&handle));
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

        CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans_a = CUBLAS_OP_N;
        cublasOperation_t trans_b = CUBLAS_OP_T;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &trans_a, sizeof(trans_a)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &trans_b, sizeof(trans_b)));

        cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, B, nC, nC));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &row_order, sizeof(row_order)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B_n_desc, CUDA_R_16BF, n, nC, nC));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(B_n_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &row_order, sizeof(row_order)));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C_n_desc, CUDA_R_32F, B, n, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(C_n_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &row_order, sizeof(row_order)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B_nn_desc, CUDA_R_16BF, n * n, nC, nC));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(B_nn_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &row_order, sizeof(row_order)));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C_nn_desc, CUDA_R_32F, B, n * n, n * n));
        CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(C_nn_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &row_order, sizeof(row_order)));

        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
                                                          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_size, sizeof(workspace_size)));

        int returned = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_n_desc, C_n_desc,
                                                    C_n_desc, pref, 1, &heuristic_n, &returned));

        returned = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_nn_desc,
                                                    C_nn_desc, C_nn_desc, pref, 1, &heuristic_nn,
                                                    &returned));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;
        if (pref)
            cublasLtMatmulPreferenceDestroy(pref);
        if (A_desc)
            cublasLtMatrixLayoutDestroy(A_desc);
        if (B_n_desc)
            cublasLtMatrixLayoutDestroy(B_n_desc);
        if (C_n_desc)
            cublasLtMatrixLayoutDestroy(C_n_desc);
        if (B_nn_desc)
            cublasLtMatrixLayoutDestroy(B_nn_desc);
        if (C_nn_desc)
            cublasLtMatrixLayoutDestroy(C_nn_desc);
        if (matmul_desc)
            cublasLtMatmulDescDestroy(matmul_desc);
        if (handle)
            cublasLtDestroy(handle);
        if (workspace)
            cudaFree(workspace);
        pref = nullptr;
        A_desc = nullptr;
        B_n_desc = nullptr;
        C_n_desc = nullptr;
        B_nn_desc = nullptr;
        C_nn_desc = nullptr;
        matmul_desc = nullptr;
        handle = nullptr;
        workspace = nullptr;
        initialized = false;
    }

    ~CublasLtCache() { destroy(); }
};

static CublasLtCache g_cublas_cache;

torch::Tensor sinkhorn_knopp_fwd(torch::Tensor inp, int iters, float eps) {
    CHECK_INPUT(inp);
    auto out = torch::empty_like(inp);
    sinkhorn_knopp_forward(out.data_ptr<float>(), inp.data_ptr<float>(), inp.size(0), inp.size(1),
                           iters, eps);
    return out;
}

torch::Tensor sinkhorn_knopp_bwd(torch::Tensor grad, torch::Tensor out, torch::Tensor inp,
                                 int iters, float eps) {
    CHECK_INPUT(grad);
    CHECK_INPUT(out);
    CHECK_INPUT(inp);
    auto d_inp = torch::empty_like(inp);
    sinkhorn_knopp_backward(d_inp.data_ptr<float>(), grad.data_ptr<float>(), out.data_ptr<float>(),
                            inp.data_ptr<float>(), inp.size(0), iters, eps);
    return d_inp;
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_fwd(torch::Tensor inp, torch::Tensor weight,
                                                     float eps) {
    CHECK_INPUT(inp);
    CHECK_INPUT(weight);
    int B = inp.size(0), C = inp.size(1);
    auto out = torch::empty_like(inp);
    auto rms = torch::empty({B}, inp.options().dtype(torch::kFloat32));
    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(out.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(inp.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(weight.data_ptr<at::BFloat16>()), B, C, eps);
    return {out, rms};
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_bwd(torch::Tensor grad, torch::Tensor inp,
                                                     torch::Tensor weight, torch::Tensor rms) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inp);
    CHECK_INPUT(weight);
    CHECK_INPUT(rms);
    int B = inp.size(0), C = inp.size(1);
    auto grad_f32 = grad.to(torch::kFloat32).contiguous();
    auto d_inp_f32 = torch::empty({B, C}, inp.options().dtype(torch::kFloat32));
    auto d_weight = torch::zeros({C}, inp.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_inp_f32.data_ptr<float>(), d_weight.data_ptr<float>(),
                     grad_f32.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(inp.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C);
    return {d_inp_f32.to(torch::kBFloat16), d_weight};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
mhc_layer_fwd(torch::Tensor x_expanded, torch::Tensor rmsnorm_weight, torch::Tensor H_pre,
              torch::Tensor H_post, torch::Tensor H_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(H_pre);
    CHECK_INPUT(H_post);
    CHECK_INPUT(H_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);

    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto x_agg_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto rms = torch::empty({B}, x_expanded.options().dtype(torch::kFloat32));
    auto y_norm_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto M = torch::empty({n, n}, x_expanded.options().dtype(torch::kFloat32));
    auto output = torch::empty({B, n, C}, x_expanded.options().dtype(torch::kFloat32));
    auto H_pre_activated = torch::empty({n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_post_activated = torch::empty({n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_res_exp = torch::empty({n, n}, x_expanded.options().dtype(torch::kFloat32));

    stream_aggregate_bf16_fused_sigmoid(
        reinterpret_cast<floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        H_pre_activated.data_ptr<float>(), x_f32.data_ptr<float>(), H_pre.data_ptr<float>(), B, n,
        C);

    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()), B, C, eps);

    sinkhorn_knopp_forward_fused_exp(M.data_ptr<float>(), H_res_exp.data_ptr<float>(),
                                     H_res.data_ptr<float>(), n, n, sinkhorn_iters, eps);

    stream_distribute_mix_add_fused(
        output.data_ptr<float>(), H_post_activated.data_ptr<float>(), x_f32.data_ptr<float>(),
        reinterpret_cast<const floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()),
        H_post.data_ptr<float>(), M.data_ptr<float>(), B, n, C);

    return {output, rms, x_agg_bf16, H_pre_activated, H_post_activated, M, y_norm_bf16};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mhc_layer_bwd(torch::Tensor grad_output, torch::Tensor x_expanded, torch::Tensor rmsnorm_weight,
              torch::Tensor rms, torch::Tensor x_agg_bf16, torch::Tensor H_pre_activated,
              torch::Tensor H_post_activated, torch::Tensor M, torch::Tensor y_norm_bf16,
              torch::Tensor H_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(rms);
    CHECK_INPUT(x_agg_bf16);
    CHECK_INPUT(H_pre_activated);
    CHECK_INPUT(H_post_activated);
    CHECK_INPUT(M);
    CHECK_INPUT(y_norm_bf16);
    CHECK_INPUT(H_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_SIZE = 256;
    int workspace_num_blocks = std::min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto workspace_dH =
        torch::empty({workspace_num_blocks * n}, grad_output.options().dtype(torch::kFloat32));
    auto workspace_dM =
        torch::empty({workspace_num_blocks * n * n}, grad_output.options().dtype(torch::kFloat32));

    auto grad_f32 = grad_output.to(torch::kFloat32).contiguous();
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto y_norm_f32 = y_norm_bf16.to(torch::kFloat32).contiguous();

    auto d_x_mix = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_y_norm = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_M = torch::empty({n, n}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_post_activated = torch::empty({n}, grad_output.options().dtype(torch::kFloat32));

    stream_distribute_mix_backward_fused(
        d_x_mix.data_ptr<float>(), d_y_norm.data_ptr<float>(), d_M.data_ptr<float>(),
        d_H_post_activated.data_ptr<float>(), grad_f32.data_ptr<float>(), x_f32.data_ptr<float>(),
        y_norm_f32.data_ptr<float>(), M.data_ptr<float>(), H_post_activated.data_ptr<float>(), B, n,
        C, workspace_dM.data_ptr<float>(), workspace_dH.data_ptr<float>(), workspace_num_blocks,
        stream);

    auto d_H_post = d_H_post_activated * H_post_activated * (1.0f - H_post_activated / 2.0f);

    auto H_res_exp = torch::exp(H_res);
    auto d_H_res_exp = torch::empty({n, n}, grad_output.options().dtype(torch::kFloat32));
    sinkhorn_knopp_backward(d_H_res_exp.data_ptr<float>(), d_M.data_ptr<float>(),
                            M.data_ptr<float>(), H_res_exp.data_ptr<float>(), n, sinkhorn_iters,
                            eps, stream);
    auto d_H_res = d_H_res_exp * H_res_exp;

    auto d_x_agg = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_rmsnorm_weight = torch::zeros({C}, grad_output.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_x_agg.data_ptr<float>(), d_rmsnorm_weight.data_ptr<float>(),
                     d_y_norm.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C, stream);

    auto d_x_from_agg = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_pre_activated = torch::empty({n}, grad_output.options().dtype(torch::kFloat32));
    stream_aggregate_backward(d_x_from_agg.data_ptr<float>(), d_H_pre_activated.data_ptr<float>(),
                              d_x_agg.data_ptr<float>(), x_f32.data_ptr<float>(),
                              H_pre_activated.data_ptr<float>(), B, n, C,
                              workspace_dH.data_ptr<float>(), workspace_num_blocks, stream);

    auto d_H_pre = d_H_pre_activated * H_pre_activated * (1.0f - H_pre_activated);

    auto d_x_expanded = d_x_mix + d_x_from_agg;

    return {d_x_expanded, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
mhc_layer_fwd_dynamic(torch::Tensor x_expanded, torch::Tensor rmsnorm_weight, torch::Tensor phi_pre,
                      torch::Tensor phi_post, torch::Tensor phi_res, float alpha_pre,
                      float alpha_post, float alpha_res, torch::Tensor b_pre, torch::Tensor b_post,
                      torch::Tensor b_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(phi_pre);
    CHECK_INPUT(phi_post);
    CHECK_INPUT(phi_res);
    CHECK_INPUT(b_pre);
    CHECK_INPUT(b_post);
    CHECK_INPUT(b_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    int nC = n * C;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto x_flat = x_f32.reshape({B, nC});
    auto x_flat_bf16 = x_flat.to(torch::kBFloat16).contiguous();

    auto rms_h = torch::empty({B}, x_expanded.options().dtype(torch::kFloat32));
    compute_rms(rms_h.data_ptr<float>(),
                reinterpret_cast<const floatX*>(x_flat_bf16.data_ptr<at::BFloat16>()), B, nC, eps,
                stream);

    auto phi_pre_bf16 = phi_pre.to(torch::kBFloat16).contiguous();
    auto phi_post_bf16 = phi_post.to(torch::kBFloat16).contiguous();
    auto phi_res_bf16 = phi_res.to(torch::kBFloat16).contiguous();

    auto H_pre_proj = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_post_proj = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_res_proj = torch::empty({B, n * n}, x_expanded.options().dtype(torch::kFloat32));

    g_cublas_cache.init(B, n, nC);

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(g_cublas_cache.handle, g_cublas_cache.matmul_desc, &alpha,
                                x_flat_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.A_desc,
                                phi_pre_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.B_n_desc,
                                &beta, H_pre_proj.data_ptr<float>(), g_cublas_cache.C_n_desc,
                                H_pre_proj.data_ptr<float>(), g_cublas_cache.C_n_desc,
                                &g_cublas_cache.heuristic_n.algo, g_cublas_cache.workspace,
                                g_cublas_cache.workspace_size, stream));

    CHECK_CUBLAS(cublasLtMatmul(g_cublas_cache.handle, g_cublas_cache.matmul_desc, &alpha,
                                x_flat_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.A_desc,
                                phi_post_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.B_n_desc,
                                &beta, H_post_proj.data_ptr<float>(), g_cublas_cache.C_n_desc,
                                H_post_proj.data_ptr<float>(), g_cublas_cache.C_n_desc,
                                &g_cublas_cache.heuristic_n.algo, g_cublas_cache.workspace,
                                g_cublas_cache.workspace_size, stream));

    CHECK_CUBLAS(cublasLtMatmul(g_cublas_cache.handle, g_cublas_cache.matmul_desc, &alpha,
                                x_flat_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.A_desc,
                                phi_res_bf16.data_ptr<at::BFloat16>(), g_cublas_cache.B_nn_desc,
                                &beta, H_res_proj.data_ptr<float>(), g_cublas_cache.C_nn_desc,
                                H_res_proj.data_ptr<float>(), g_cublas_cache.C_nn_desc,
                                &g_cublas_cache.heuristic_nn.algo, g_cublas_cache.workspace,
                                g_cublas_cache.workspace_size, stream));

    divide_by_rms(H_pre_proj.data_ptr<float>(), rms_h.data_ptr<float>(), B, n, stream);
    divide_by_rms(H_post_proj.data_ptr<float>(), rms_h.data_ptr<float>(), B, n, stream);
    divide_by_rms(H_res_proj.data_ptr<float>(), rms_h.data_ptr<float>(), B, n * n, stream);

    auto H_pre_activated = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_post_activated = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_res_exp = torch::empty({B, n, n}, x_expanded.options().dtype(torch::kFloat32));

    apply_alpha_bias_sigmoid(H_pre_activated.data_ptr<float>(), H_pre_proj.data_ptr<float>(),
                             alpha_pre, b_pre.data_ptr<float>(), B, n, stream);

    apply_alpha_bias_2sigmoid(H_post_activated.data_ptr<float>(), H_post_proj.data_ptr<float>(),
                              alpha_post, b_post.data_ptr<float>(), B, n, stream);

    apply_alpha_bias_exp(H_res_exp.data_ptr<float>(), H_res_proj.data_ptr<float>(), alpha_res,
                         b_res.data_ptr<float>(), B, n, stream);

    auto M = torch::empty({B, n, n}, x_expanded.options().dtype(torch::kFloat32));
    sinkhorn_knopp_forward_batched(M.data_ptr<float>(), H_res_exp.data_ptr<float>(), B, n,
                                   sinkhorn_iters, eps, stream);

    auto x_agg_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto rms = torch::empty({B}, x_expanded.options().dtype(torch::kFloat32));
    auto y_norm_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto output = torch::empty({B, n, C}, x_expanded.options().dtype(torch::kFloat32));

    stream_aggregate_bf16_dynamic(reinterpret_cast<floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                                  x_f32.data_ptr<float>(), H_pre_activated.data_ptr<float>(), B, n,
                                  C, stream);

    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()), B, C, eps,
        stream);

    stream_distribute_mix_add_fused_dynamic(
        output.data_ptr<float>(), x_f32.data_ptr<float>(),
        reinterpret_cast<const floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()),
        H_post_activated.data_ptr<float>(), M.data_ptr<float>(), B, n, C, stream);

    return {output,      rms,         x_agg_bf16, H_pre_activated, H_post_activated, M,
            y_norm_bf16, x_flat_bf16, rms_h};
}

std::tuple<torch::Tensor, torch::Tensor>
mhc_layer_bwd_dynamic(torch::Tensor grad_output, torch::Tensor x_expanded,
                      torch::Tensor rmsnorm_weight, torch::Tensor rms, torch::Tensor x_agg_bf16,
                      torch::Tensor H_pre_activated, torch::Tensor H_post_activated,
                      torch::Tensor M, torch::Tensor y_norm_bf16, float eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(rms);
    CHECK_INPUT(x_agg_bf16);
    CHECK_INPUT(H_pre_activated);
    CHECK_INPUT(H_post_activated);
    CHECK_INPUT(M);
    CHECK_INPUT(y_norm_bf16);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_SIZE = 256;
    int workspace_num_blocks = std::min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto workspace_dH =
        torch::empty({workspace_num_blocks * n}, grad_output.options().dtype(torch::kFloat32));
    auto workspace_dM =
        torch::empty({workspace_num_blocks * n * n}, grad_output.options().dtype(torch::kFloat32));

    auto grad_f32 = grad_output.to(torch::kFloat32).contiguous();
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto y_norm_f32 = y_norm_bf16.to(torch::kFloat32).contiguous();

    auto d_x_mix = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_y_norm = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_M = torch::empty({B, n, n}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_post = torch::empty({B, n}, grad_output.options().dtype(torch::kFloat32));

    stream_distribute_mix_backward_fused(
        d_x_mix.data_ptr<float>(), d_y_norm.data_ptr<float>(), d_M.data_ptr<float>(),
        d_H_post.data_ptr<float>(), grad_f32.data_ptr<float>(), x_f32.data_ptr<float>(),
        y_norm_f32.data_ptr<float>(), M.data_ptr<float>(), H_post_activated.data_ptr<float>(), B, n,
        C, workspace_dM.data_ptr<float>(), workspace_dH.data_ptr<float>(), workspace_num_blocks,
        stream);

    auto d_x_agg = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_rmsnorm_weight = torch::zeros({C}, grad_output.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_x_agg.data_ptr<float>(), d_rmsnorm_weight.data_ptr<float>(),
                     d_y_norm.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C, stream);

    auto d_x_from_agg = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_pre = torch::empty({B, n}, grad_output.options().dtype(torch::kFloat32));
    stream_aggregate_backward(d_x_from_agg.data_ptr<float>(), d_H_pre.data_ptr<float>(),
                              d_x_agg.data_ptr<float>(), x_f32.data_ptr<float>(),
                              H_pre_activated.data_ptr<float>(), B, n, C,
                              workspace_dH.data_ptr<float>(), workspace_num_blocks, stream);

    auto d_x_expanded = d_x_mix + d_x_from_agg;

    return {d_x_expanded, d_rmsnorm_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_knopp_fwd", &sinkhorn_knopp_fwd);
    m.def("sinkhorn_knopp_bwd", &sinkhorn_knopp_bwd);
    m.def("rmsnorm_fwd", &rmsnorm_fwd);
    m.def("rmsnorm_bwd", &rmsnorm_bwd);
    m.def("mhc_layer_fwd", &mhc_layer_fwd);
    m.def("mhc_layer_bwd", &mhc_layer_bwd);
    m.def("mhc_layer_fwd_dynamic", &mhc_layer_fwd_dynamic);
    m.def("mhc_layer_bwd_dynamic", &mhc_layer_bwd_dynamic);
}
