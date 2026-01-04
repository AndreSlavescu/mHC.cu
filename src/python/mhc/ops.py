import torch
from torch.autograd import Function

try:
    import mhc_cuda
except ImportError:
    raise ImportError(
        "mhc_cuda not found. Please install the CUDA extension by running:\n"
        "pip install -e ."
    )


class SinkhornKnoppFunction(Function):
    @staticmethod
    def forward(ctx, inp, num_iters, eps):
        out = mhc_cuda.sinkhorn_knopp_fwd(inp.contiguous(), num_iters, eps)
        ctx.save_for_backward(out, inp)
        ctx.num_iters = num_iters
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, inp = ctx.saved_tensors
        d_inp = mhc_cuda.sinkhorn_knopp_bwd(
            grad_output.contiguous(), out, inp, ctx.num_iters, ctx.eps
        )
        return d_inp, None, None


class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, eps):
        out, rms = mhc_cuda.rmsnorm_fwd(inp.contiguous(), weight.contiguous(), eps)
        ctx.save_for_backward(inp, weight, rms)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, rms = ctx.saved_tensors
        d_inp, d_weight = mhc_cuda.rmsnorm_bwd(
            grad_output.contiguous(), inp, weight, rms
        )
        return d_inp, d_weight, None


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    return SinkhornKnoppFunction.apply(inp.float(), num_iters, eps)


def rmsnorm(inp, weight, eps=1e-5):
    return RMSNormFunction.apply(inp.bfloat16(), weight.bfloat16(), eps)


class MHCLayerFunction(Function):
    @staticmethod
    def forward(
        ctx, x_expanded, rmsnorm_weight, H_pre, H_post, H_res, sinkhorn_iters, eps
    ):
        (
            output,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
        ) = mhc_cuda.mhc_layer_fwd(
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            H_pre.contiguous(),
            H_post.contiguous(),
            H_res.contiguous(),
            sinkhorn_iters,
            eps,
        )

        ctx.save_for_backward(
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            H_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            H_res,
        ) = ctx.saved_tensors

        d_x, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res = mhc_cuda.mhc_layer_bwd(
            grad_output.contiguous(),
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            rms.contiguous(),
            x_agg_bf16.contiguous(),
            H_pre_activated.contiguous(),
            H_post_activated.contiguous(),
            M.contiguous(),
            y_norm_bf16.contiguous(),
            H_res.contiguous(),
            ctx.sinkhorn_iters,
            ctx.eps,
        )

        return d_x, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res, None, None


def mhc_layer_fused(
    x_expanded, rmsnorm_weight, H_pre, H_post, H_res, sinkhorn_iters=20, eps=1e-5
):
    return MHCLayerFunction.apply(
        x_expanded.float(),
        rmsnorm_weight.bfloat16(),
        H_pre.float(),
        H_post.float(),
        H_res.float(),
        sinkhorn_iters,
        eps,
    )


class MHCLayerDynamicFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x_expanded,
        rmsnorm_weight,
        phi_pre,
        phi_post,
        phi_res,
        alpha_pre,
        alpha_post,
        alpha_res,
        b_pre,
        b_post,
        b_res,
        sinkhorn_iters,
        eps,
    ):
        (
            output,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            x_flat_bf16,
            rms_h,
        ) = mhc_cuda.mhc_layer_fwd_dynamic(
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            phi_pre.contiguous(),
            phi_post.contiguous(),
            phi_res.contiguous(),
            alpha_pre,
            alpha_post,
            alpha_res,
            b_pre.contiguous(),
            b_post.contiguous(),
            b_res.contiguous(),
            sinkhorn_iters,
            eps,
        )

        ctx.save_for_backward(
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
        )
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
        ) = ctx.saved_tensors

        d_x, d_rmsnorm_weight = mhc_cuda.mhc_layer_bwd_dynamic(
            grad_output.contiguous(),
            x_expanded.contiguous(),
            rmsnorm_weight.contiguous(),
            rms.contiguous(),
            x_agg_bf16.contiguous(),
            H_pre_activated.contiguous(),
            H_post_activated.contiguous(),
            M.contiguous(),
            y_norm_bf16.contiguous(),
            ctx.eps,
        )

        return (
            d_x,
            d_rmsnorm_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mhc_layer_fused_dynamic(
    x_expanded,
    rmsnorm_weight,
    phi_pre,
    phi_post,
    phi_res,
    alpha_pre,
    alpha_post,
    alpha_res,
    b_pre,
    b_post,
    b_res,
    sinkhorn_iters=20,
    eps=1e-5,
):
    return MHCLayerDynamicFunction.apply(
        x_expanded.float(),
        rmsnorm_weight.bfloat16(),
        phi_pre.float(),
        phi_post.float(),
        phi_res.float(),
        alpha_pre.detach().item() if hasattr(alpha_pre, "detach") else float(alpha_pre),
        (
            alpha_post.detach().item()
            if hasattr(alpha_post, "detach")
            else float(alpha_post)
        ),
        alpha_res.detach().item() if hasattr(alpha_res, "detach") else float(alpha_res),
        b_pre.float(),
        b_post.float(),
        b_res.float(),
        sinkhorn_iters,
        eps,
    )
