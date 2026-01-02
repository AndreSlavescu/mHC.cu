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
