#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../csrc/include/mhc_types.h"
#include "../csrc/kernels/rmsnorm.cuh"
#include "../csrc/kernels/sinkhorn_knopp.cuh"
#include "../csrc/kernels/stream_ops.cuh"
#include "../csrc/kernels/type_conversions.cuh"

using namespace mhc;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

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

torch::Tensor stream_aggregate_fwd(torch::Tensor inp, torch::Tensor H_pre) {
    CHECK_INPUT(inp);
    CHECK_INPUT(H_pre);
    int B = inp.size(0), n = inp.size(1), C = inp.size(2);
    auto out = torch::empty({B, C}, inp.options());
    stream_aggregate(out.data_ptr<float>(), inp.data_ptr<float>(), H_pre.data_ptr<float>(), B, n,
                     C);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> stream_aggregate_bwd(torch::Tensor grad, torch::Tensor inp,
                                                              torch::Tensor H_pre) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inp);
    CHECK_INPUT(H_pre);
    int B = inp.size(0), n = inp.size(1), C = inp.size(2);
    auto d_inp = torch::empty_like(inp);
    auto d_H_pre = torch::zeros_like(H_pre);
    stream_aggregate_backward(d_inp.data_ptr<float>(), d_H_pre.data_ptr<float>(),
                              grad.data_ptr<float>(), inp.data_ptr<float>(),
                              H_pre.data_ptr<float>(), B, n, C);
    return {d_inp, d_H_pre};
}

torch::Tensor stream_distribute_fwd(torch::Tensor inp, torch::Tensor H_post, int n) {
    CHECK_INPUT(inp);
    CHECK_INPUT(H_post);
    int B = inp.size(0), C = inp.size(1);
    auto out = torch::empty({B, n, C}, inp.options());
    stream_distribute(out.data_ptr<float>(), inp.data_ptr<float>(), H_post.data_ptr<float>(), B, n,
                      C);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
stream_distribute_bwd(torch::Tensor grad, torch::Tensor inp, torch::Tensor H_post) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inp);
    CHECK_INPUT(H_post);
    int B = grad.size(0), n = grad.size(1), C = grad.size(2);
    auto d_inp = torch::empty({B, C}, inp.options());
    auto d_H_post = torch::zeros_like(H_post);
    stream_distribute_backward(d_inp.data_ptr<float>(), d_H_post.data_ptr<float>(),
                               grad.data_ptr<float>(), inp.data_ptr<float>(),
                               H_post.data_ptr<float>(), B, n, C);
    return {d_inp, d_H_post};
}

torch::Tensor stream_mix_fwd(torch::Tensor inp, torch::Tensor M) {
    CHECK_INPUT(inp);
    CHECK_INPUT(M);
    int B = inp.size(0), n = inp.size(1), C = inp.size(2);
    auto out = torch::empty_like(inp);
    stream_mix(out.data_ptr<float>(), inp.data_ptr<float>(), M.data_ptr<float>(), B, n, C);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> stream_mix_bwd(torch::Tensor grad, torch::Tensor inp,
                                                        torch::Tensor M) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inp);
    CHECK_INPUT(M);
    int B = inp.size(0), n = inp.size(1), C = inp.size(2);
    auto d_inp = torch::empty_like(inp);
    auto d_M = torch::zeros_like(M);
    stream_mix_backward(d_inp.data_ptr<float>(), d_M.data_ptr<float>(), grad.data_ptr<float>(),
                        inp.data_ptr<float>(), M.data_ptr<float>(), B, n, C);
    return {d_inp, d_M};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    auto y_dist = torch::empty({B, n, C}, x_expanded.options().dtype(torch::kFloat32));
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

    stream_distribute_from_bf16_fused_sigmoid(
        y_dist.data_ptr<float>(), H_post_activated.data_ptr<float>(),
        reinterpret_cast<const floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()),
        H_post.data_ptr<float>(), B, n, C);

    sinkhorn_knopp_forward_fused_exp(M.data_ptr<float>(), H_res_exp.data_ptr<float>(),
                                     H_res.data_ptr<float>(), n, n, sinkhorn_iters, eps);

    stream_mix_add(output.data_ptr<float>(), x_f32.data_ptr<float>(), y_dist.data_ptr<float>(),
                   M.data_ptr<float>(), B, n, C);

    return {output, rms, x_agg_bf16, H_pre_activated, H_post_activated, H_res_exp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_knopp_fwd", &sinkhorn_knopp_fwd);
    m.def("sinkhorn_knopp_bwd", &sinkhorn_knopp_bwd);
    m.def("rmsnorm_fwd", &rmsnorm_fwd);
    m.def("rmsnorm_bwd", &rmsnorm_bwd);
    m.def("stream_aggregate_fwd", &stream_aggregate_fwd);
    m.def("stream_aggregate_bwd", &stream_aggregate_bwd);
    m.def("stream_distribute_fwd", &stream_distribute_fwd);
    m.def("stream_distribute_bwd", &stream_distribute_bwd);
    m.def("stream_mix_fwd", &stream_mix_fwd);
    m.def("stream_mix_bwd", &stream_mix_bwd);
    m.def("mhc_layer_fwd", &mhc_layer_fwd);
}
