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


class StreamAggregateFunction(Function):
    @staticmethod
    def forward(ctx, inp, H_pre):
        out = mhc_cuda.stream_aggregate_fwd(inp.contiguous(), H_pre.contiguous())
        ctx.save_for_backward(inp, H_pre)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, H_pre = ctx.saved_tensors
        d_inp, d_H_pre = mhc_cuda.stream_aggregate_bwd(
            grad_output.contiguous(), inp, H_pre
        )
        return d_inp, d_H_pre


class StreamDistributeFunction(Function):
    @staticmethod
    def forward(ctx, inp, H_post, n):
        out = mhc_cuda.stream_distribute_fwd(inp.contiguous(), H_post.contiguous(), n)
        ctx.save_for_backward(inp, H_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, H_post = ctx.saved_tensors
        d_inp, d_H_post = mhc_cuda.stream_distribute_bwd(
            grad_output.contiguous(), inp, H_post
        )
        return d_inp, d_H_post, None


class StreamMixFunction(Function):
    @staticmethod
    def forward(ctx, inp, M):
        out = mhc_cuda.stream_mix_fwd(inp.contiguous(), M.contiguous())
        ctx.save_for_backward(inp, M)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, M = ctx.saved_tensors
        d_inp, d_M = mhc_cuda.stream_mix_bwd(grad_output.contiguous(), inp, M)
        return d_inp, d_M


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    return SinkhornKnoppFunction.apply(inp.float(), num_iters, eps)


def rmsnorm(inp, weight, eps=1e-5):
    return RMSNormFunction.apply(inp.bfloat16(), weight.bfloat16(), eps)


def stream_aggregate(inp, H_pre):
    return StreamAggregateFunction.apply(inp.float(), H_pre.float())


def stream_distribute(inp, H_post):
    n = H_post.size(0)
    return StreamDistributeFunction.apply(inp.float(), H_post.float(), n)


def stream_mix(inp, M):
    return StreamMixFunction.apply(inp.float(), M.float())


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
            H_res_exp,
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
            H_res_exp,
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
            H_res_exp,
        ) = ctx.saved_tensors

        B, n, C = x_expanded.shape
        x_f32 = x_expanded.float().contiguous()
        grad_f32 = grad_output.float().contiguous()

        M = mhc_cuda.sinkhorn_knopp_fwd(H_res_exp, ctx.sinkhorn_iters, ctx.eps)

        d_x_mixed, d_M = mhc_cuda.stream_mix_bwd(grad_f32, x_f32, M)

        d_H_res_exp = mhc_cuda.sinkhorn_knopp_bwd(
            d_M, M, H_res_exp, ctx.sinkhorn_iters, ctx.eps
        )
        d_H_res = d_H_res_exp * H_res_exp

        y_norm_bf16 = mhc_cuda.rmsnorm_fwd(x_agg_bf16, rmsnorm_weight, ctx.eps)[0]
        y_norm_f32 = y_norm_bf16.float()

        d_y_norm, d_H_post_activated = mhc_cuda.stream_distribute_bwd(
            grad_f32, y_norm_f32, H_post_activated
        )
        d_H_post = (
            d_H_post_activated * H_post_activated * (1.0 - H_post_activated / 2.0)
        )

        d_x_agg, d_rmsnorm_weight = mhc_cuda.rmsnorm_bwd(
            d_y_norm.bfloat16(), x_agg_bf16, rmsnorm_weight, rms
        )
        d_x_agg_f32 = d_x_agg.float()

        d_x_from_agg, d_H_pre_activated = mhc_cuda.stream_aggregate_bwd(
            d_x_agg_f32, x_f32, H_pre_activated
        )
        d_H_pre = d_H_pre_activated * H_pre_activated * (1.0 - H_pre_activated)

        d_x_expanded = d_x_mixed + d_x_from_agg

        return d_x_expanded, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res, None, None


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
