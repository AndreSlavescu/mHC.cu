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


def mhc_layer_fused_inference(
    x_expanded, rmsnorm_weight, H_pre, H_post, H_res, sinkhorn_iters=20, eps=1e-5
):
    # inference focused forward pass -> no backward support for maximum speed
    return mhc_cuda.mhc_layer_fwd_inference(
        x_expanded.float().contiguous(),
        rmsnorm_weight.bfloat16().contiguous(),
        H_pre.float().contiguous(),
        H_post.float().contiguous(),
        H_res.float().contiguous(),
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
        n = phi_pre.size(0)
        phi_res_view = phi_res.view(n * n, -1)
        nC = phi_pre.size(1)
        out_dim = n + n + n * n
        phi_concat = torch.empty(
            (out_dim, nC),
            device=phi_pre.device,
            dtype=torch.bfloat16,
        )
        # Avoid a large float32 concat temporary to reduce peak memory.
        phi_concat[:n].copy_(phi_pre)
        phi_concat[n : 2 * n].copy_(phi_post)
        phi_concat[2 * n :].copy_(phi_res_view)
        alpha_pre_val = (
            float(alpha_pre.item()) if torch.is_tensor(alpha_pre) else float(alpha_pre)
        )
        alpha_post_val = (
            float(alpha_post.item())
            if torch.is_tensor(alpha_post)
            else float(alpha_post)
        )
        alpha_res_val = (
            float(alpha_res.item()) if torch.is_tensor(alpha_res) else float(alpha_res)
        )

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
            phi_concat,
            alpha_pre_val,
            alpha_post_val,
            alpha_res_val,
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
            x_flat_bf16,
            rms_h,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            alpha_pre,
            alpha_post,
            alpha_res,
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
            x_flat_bf16,
            rms_h,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            alpha_pre,
            alpha_post,
            alpha_res,
        ) = ctx.saved_tensors

        d_x, d_rmsnorm_weight, d_H_pre, d_H_post, d_M = mhc_cuda.mhc_layer_bwd_dynamic(
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

        with torch.no_grad():
            B, n, C = x_expanded.shape
            nC = n * C

            x_flat = x_flat_bf16.float()
            rms_h_f32 = rms_h.float()
            rms_inv = 1.0 / rms_h_f32
            rms_inv2 = rms_inv * rms_inv

            phi_pre_f32 = phi_pre.float()
            phi_post_f32 = phi_post.float()
            phi_res_f32 = phi_res.float()

            p_pre = x_flat @ phi_pre_f32.t()
            p_post = x_flat @ phi_post_f32.t()
            p_res_flat = x_flat @ phi_res_f32.t()
            p_res = p_res_flat.view(B, n, n)

            H_pre_act = H_pre_activated.float()
            H_post_act = H_post_activated.float()

            d_tilde_pre = d_H_pre * H_pre_act * (1.0 - H_pre_act)
            d_tilde_post = d_H_post * H_post_act * (1.0 - H_post_act / 2.0)

            alpha_pre_f32 = alpha_pre.float()
            alpha_post_f32 = alpha_post.float()
            alpha_res_f32 = alpha_res.float()
            b_res_f32 = b_res.float()

            tilde_res = alpha_res_f32 * p_res * rms_inv.view(B, 1, 1) + b_res_f32
            tilde_res = torch.clamp(tilde_res, max=20.0)
            H_res_exp = torch.exp(tilde_res)

            d_H_res_exp = torch.empty_like(H_res_exp)
            for b in range(B):
                d_H_res_exp[b] = mhc_cuda.sinkhorn_knopp_bwd(
                    d_M[b].contiguous(),
                    M[b].contiguous(),
                    H_res_exp[b].contiguous(),
                    ctx.sinkhorn_iters,
                    ctx.eps,
                )
            d_tilde_res = d_H_res_exp * H_res_exp

            d_b_pre = d_tilde_pre.sum(dim=0)
            d_b_post = d_tilde_post.sum(dim=0)
            d_b_res = d_tilde_res.sum(dim=0)

            d_alpha_pre = (d_tilde_pre * (p_pre * rms_inv[:, None])).sum()
            d_alpha_post = (d_tilde_post * (p_post * rms_inv[:, None])).sum()
            d_alpha_res = (d_tilde_res * (p_res * rms_inv.view(B, 1, 1))).sum()

            d_p_pre = d_tilde_pre * (alpha_pre_f32 * rms_inv[:, None])
            d_p_post = d_tilde_post * (alpha_post_f32 * rms_inv[:, None])
            d_p_res = d_tilde_res * (alpha_res_f32 * rms_inv.view(B, 1, 1))
            d_p_res_flat = d_p_res.reshape(B, n * n)

            d_phi_pre = d_p_pre.t() @ x_flat
            d_phi_post = d_p_post.t() @ x_flat
            d_phi_res = d_p_res_flat.t() @ x_flat

            d_x_flat = d_p_pre @ phi_pre_f32
            d_x_flat += d_p_post @ phi_post_f32
            d_x_flat += d_p_res_flat @ phi_res_f32

            d_r = -(d_tilde_pre * (alpha_pre_f32 * p_pre) * rms_inv2[:, None]).sum(
                dim=1
            )
            d_r -= (d_tilde_post * (alpha_post_f32 * p_post) * rms_inv2[:, None]).sum(
                dim=1
            )
            d_r -= (d_tilde_res * (alpha_res_f32 * p_res) * rms_inv2.view(B, 1, 1)).sum(
                dim=(1, 2)
            )
            d_x_flat += d_r[:, None] * x_flat * (rms_inv[:, None] / float(nC))

            d_x = d_x + d_x_flat.view(B, n, C)

            d_b_pre = d_b_pre.to(b_pre.dtype)
            d_b_post = d_b_post.to(b_post.dtype)
            d_b_res = d_b_res.to(b_res.dtype)
            d_alpha_pre = d_alpha_pre.to(alpha_pre.dtype)
            d_alpha_post = d_alpha_post.to(alpha_post.dtype)
            d_alpha_res = d_alpha_res.to(alpha_res.dtype)
            d_phi_pre = d_phi_pre.to(phi_pre.dtype)
            d_phi_post = d_phi_post.to(phi_post.dtype)
            d_phi_res = d_phi_res.to(phi_res.dtype)

        return (
            d_x,
            d_rmsnorm_weight,
            d_phi_pre,
            d_phi_post,
            d_phi_res,
            d_alpha_pre,
            d_alpha_post,
            d_alpha_res,
            d_b_pre,
            d_b_post,
            d_b_res,
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
    if not torch.is_tensor(alpha_pre):
        alpha_pre = torch.tensor(alpha_pre, device=phi_pre.device, dtype=phi_pre.dtype)
    if not torch.is_tensor(alpha_post):
        alpha_post = torch.tensor(
            alpha_post, device=phi_pre.device, dtype=phi_pre.dtype
        )
    if not torch.is_tensor(alpha_res):
        alpha_res = torch.tensor(alpha_res, device=phi_pre.device, dtype=phi_pre.dtype)
    return MHCLayerDynamicFunction.apply(
        x_expanded.float(),
        rmsnorm_weight.bfloat16(),
        phi_pre.float(),
        phi_post.float(),
        phi_res.float(),
        alpha_pre,
        alpha_post,
        alpha_res,
        b_pre.float(),
        b_post.float(),
        b_res.float(),
        sinkhorn_iters,
        eps,
    )


def mhc_layer_fused_dynamic_inference(
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
    n = phi_pre.size(0)
    phi_res_view = phi_res.view(n * n, -1)
    nC = phi_pre.size(1)
    out_dim = n + n + n * n
    phi_concat = torch.empty(
        (out_dim, nC),
        device=phi_pre.device,
        dtype=torch.bfloat16,
    )
    phi_concat[:n].copy_(phi_pre)
    phi_concat[n : 2 * n].copy_(phi_post)
    phi_concat[2 * n :].copy_(phi_res_view)
    alpha_pre_val = (
        float(alpha_pre.item()) if torch.is_tensor(alpha_pre) else float(alpha_pre)
    )
    alpha_post_val = (
        float(alpha_post.item()) if torch.is_tensor(alpha_post) else float(alpha_post)
    )
    alpha_res_val = (
        float(alpha_res.item()) if torch.is_tensor(alpha_res) else float(alpha_res)
    )
    x_f32 = x_expanded.float().contiguous()
    H_pre_activated, H_post_activated, M = mhc_cuda.mhc_dynamic_h_fwd(
        x_f32,
        phi_concat,
        alpha_pre_val,
        alpha_post_val,
        alpha_res_val,
        b_pre.float().contiguous(),
        b_post.float().contiguous(),
        b_res.float().contiguous(),
        sinkhorn_iters,
        eps,
    )
    x_agg_bf16 = mhc_cuda.mhc_dynamic_aggregate_fwd(x_f32, H_pre_activated)
    y_norm_bf16, _ = mhc_cuda.rmsnorm_fwd(
        x_agg_bf16.contiguous(),
        rmsnorm_weight.bfloat16().contiguous(),
        eps,
    )
    output = mhc_cuda.mhc_dynamic_mix_fwd(
        x_f32,
        y_norm_bf16,
        H_post_activated,
        M,
    )
    return output
