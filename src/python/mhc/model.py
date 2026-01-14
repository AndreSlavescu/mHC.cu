from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

try:
    import mhc_cuda
except ImportError:  # pragma: no cover - optional fused path
    mhc_cuda = None

try:
    from . import ops as mhc_ops
except ImportError:  # pragma: no cover - optional fused path
    mhc_ops = None


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    ffn_dim: int
    max_seq_len: int = 4096
    expansion_rate: int = 4
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01
    rmsnorm_eps: float = 1e-5
    sinkhorn_eps: float = 1e-6
    rope_dim: int = 64
    rope_theta: float = 10000.0
    dropout: float = 0.0
    sdp_kernel: str = "flash"
    recompute_ratio: float = 0.0
    recompute_mhc: bool = False
    use_dynamic_h: bool = True
    use_fused_mhc: bool = True


H_RES_EXP_CLAMP = 20.0


_SDP_KERNEL_SETTINGS = {
    "auto": (True, True, True),
    "flash": (True, False, False),
    "mem-efficient": (False, True, False),
    "math": (False, False, True),
}


def _sdp_kernel_context(kernel: str, device: torch.device):
    if kernel == "auto" or device.type != "cuda":
        return contextlib.nullcontext()
    if kernel not in _SDP_KERNEL_SETTINGS:
        raise ValueError(f"Unsupported sdp kernel: {kernel}")
    enable_flash, enable_mem_efficient, enable_math = _SDP_KERNEL_SETTINGS[kernel]
    attention_mod = getattr(torch.nn, "attention", None)
    sdp_kernel = getattr(attention_mod, "sdpa_kernel", None) if attention_mod else None
    if sdp_kernel is not None:
        sdp_backend = getattr(attention_mod, "SDPBackend", None)
        if sdp_backend is not None:
            backend_map = {
                "flash": getattr(sdp_backend, "FLASH_ATTENTION", None),
                "mem-efficient": getattr(sdp_backend, "EFFICIENT_ATTENTION", None)
                or getattr(sdp_backend, "MEM_EFFICIENT", None),
                "math": getattr(sdp_backend, "MATH", None),
            }
            backend = backend_map[kernel]
            if backend is not None:
                for candidate in (backend, [backend], {backend}):
                    try:
                        return sdp_kernel(candidate)
                    except TypeError:
                        continue
        try:
            return sdp_kernel(
                enable_flash=enable_flash,
                enable_mem_efficient=enable_mem_efficient,
                enable_math=enable_math,
            )
        except TypeError:
            return contextlib.nullcontext()
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        )
    return contextlib.nullcontext()


def _checkpoint_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    try:
        return checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
    except TypeError:
        try:
            return checkpoint(block, x, use_reentrant=False)
        except TypeError:
            return checkpoint(block, x)


def _checkpoint_fn(fn, *args):
    try:
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
    except TypeError:
        try:
            return checkpoint(fn, *args, use_reentrant=False)
        except TypeError:
            return checkpoint(fn, *args)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class DynamicHFusedFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x_expanded: torch.Tensor,
        phi_pre: torch.Tensor,
        phi_post: torch.Tensor,
        phi_res: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        b_pre: torch.Tensor,
        b_post: torch.Tensor,
        b_res: torch.Tensor,
        sinkhorn_iters: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mhc_cuda is None:
            raise RuntimeError("mhc_cuda is required for fused mHC")
        n = phi_pre.size(0)
        phi_concat = (
            torch.cat([phi_pre, phi_post, phi_res.view(n * n, -1)], dim=0)
            .bfloat16()
            .contiguous()
        )
        alpha_pre_val = float(alpha_pre.item())
        alpha_post_val = float(alpha_post.item())
        alpha_res_val = float(alpha_res.item())
        b_pre_f32 = b_pre.float().contiguous()
        b_post_f32 = b_post.float().contiguous()
        b_res_f32 = b_res.float().contiguous()
        H_pre_activated, H_post_activated, M = mhc_cuda.mhc_dynamic_h_fwd(
            x_expanded.contiguous(),
            phi_concat,
            alpha_pre_val,
            alpha_post_val,
            alpha_res_val,
            b_pre_f32,
            b_post_f32,
            b_res_f32,
            sinkhorn_iters,
            eps,
        )
        ctx.save_for_backward(
            x_expanded,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            H_pre_activated,
            H_post_activated,
            M,
            alpha_pre,
            alpha_post,
            alpha_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        return H_pre_activated, H_post_activated, M

    @staticmethod
    def backward(ctx, d_H_pre: torch.Tensor, d_H_post: torch.Tensor, d_M: torch.Tensor):
        (
            x_expanded,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
            H_pre_activated,
            H_post_activated,
            M,
            alpha_pre,
            alpha_post,
            alpha_res,
        ) = ctx.saved_tensors

        B, n, C = x_expanded.shape
        nC = n * C
        x_flat = x_expanded.reshape(B, nC).float()
        rms_h = x_flat.pow(2).mean(dim=-1, keepdim=True).add(ctx.eps).sqrt()
        rms_inv = 1.0 / rms_h
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
        tilde_res = torch.clamp(tilde_res, max=H_RES_EXP_CLAMP)
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

        d_alpha_pre = (d_tilde_pre * (p_pre * rms_inv)).sum()
        d_alpha_post = (d_tilde_post * (p_post * rms_inv)).sum()
        d_alpha_res = (d_tilde_res * (p_res * rms_inv.view(B, 1, 1))).sum()

        d_p_pre = d_tilde_pre * (alpha_pre_f32 * rms_inv)
        d_p_post = d_tilde_post * (alpha_post_f32 * rms_inv)
        d_p_res = d_tilde_res * (alpha_res_f32 * rms_inv.view(B, 1, 1))
        d_p_res_flat = d_p_res.reshape(B, n * n)

        d_phi_pre = d_p_pre.t() @ x_flat
        d_phi_post = d_p_post.t() @ x_flat
        d_phi_res = d_p_res_flat.t() @ x_flat

        d_x_flat = d_p_pre @ phi_pre_f32
        d_x_flat += d_p_post @ phi_post_f32
        d_x_flat += d_p_res_flat @ phi_res_f32

        d_r = -(d_tilde_pre * (alpha_pre_f32 * p_pre) * rms_inv2).sum(dim=1)
        d_r -= (d_tilde_post * (alpha_post_f32 * p_post) * rms_inv2).sum(dim=1)
        d_r -= (d_tilde_res * (alpha_res_f32 * p_res) * rms_inv2.view(B, 1, 1)).sum(
            dim=(1, 2)
        )
        d_x_flat += d_r[:, None] * x_flat * (rms_inv / float(nC))

        d_x = d_x_flat.view(B, n, C).to(x_expanded.dtype)

        return (
            d_x,
            d_phi_pre.to(phi_pre.dtype),
            d_phi_post.to(phi_post.dtype),
            d_phi_res.to(phi_res.dtype),
            d_alpha_pre.to(alpha_pre.dtype),
            d_alpha_post.to(alpha_post.dtype),
            d_alpha_res.to(alpha_res.dtype),
            d_b_pre.to(b_pre.dtype),
            d_b_post.to(b_post.dtype),
            d_b_res.to(b_res.dtype),
            None,
            None,
        )


class AggregateFusedFunction(Function):
    @staticmethod
    def forward(
        ctx, x_expanded: torch.Tensor, H_pre_activated: torch.Tensor
    ) -> torch.Tensor:
        if mhc_cuda is None:
            raise RuntimeError("mhc_cuda is required for fused mHC")
        x_agg = mhc_cuda.mhc_dynamic_aggregate_fwd(
            x_expanded.contiguous(),
            H_pre_activated.contiguous(),
        )
        ctx.save_for_backward(x_expanded, H_pre_activated)
        return x_agg

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_expanded, H_pre_activated = ctx.saved_tensors
        d_x, d_H_pre = mhc_cuda.mhc_dynamic_aggregate_bwd(
            grad_output.contiguous(),
            x_expanded.contiguous(),
            H_pre_activated.contiguous(),
        )
        return d_x, d_H_pre


class MixFusedFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x_expanded: torch.Tensor,
        y_bf16: torch.Tensor,
        H_post_activated: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        if mhc_cuda is None:
            raise RuntimeError("mhc_cuda is required for fused mHC")
        output = mhc_cuda.mhc_dynamic_mix_fwd(
            x_expanded.contiguous(),
            y_bf16.contiguous(),
            H_post_activated.contiguous(),
            M.contiguous(),
        )
        ctx.save_for_backward(x_expanded, y_bf16, H_post_activated, M)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_expanded, y_bf16, H_post_activated, M = ctx.saved_tensors
        d_x, d_y, d_H_post, d_M = mhc_cuda.mhc_dynamic_mix_bwd(
            grad_output.contiguous(),
            x_expanded.contiguous(),
            y_bf16.contiguous(),
            H_post_activated.contiguous(),
            M.contiguous(),
        )
        return d_x, d_y, d_H_post, d_M


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rope_dim: int,
        rope_theta: float,
        dropout: float,
        sdp_kernel: str = "auto",
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_dim = min(rope_dim, self.head_dim)
        self.rope = (
            RotaryEmbedding(self.rope_dim, rope_theta) if self.rope_dim > 0 else None
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.sdp_kernel = sdp_kernel

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x_rope = x[..., : self.rope_dim]
        x_pass = x[..., self.rope_dim :]
        x_rope = x_rope * cos + _rotate_half(x_rope) * sin
        return torch.cat([x_rope, x_pass], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(0, 3, 1, 4, 2)
        q, k, v = qkv.unbind(dim=-1)

        if self.rope_dim > 0 and self.rope is not None:
            cos, sin = self.rope(seq_len, q.device, q.dtype)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        sdp_ctx = _sdp_kernel_context(self.sdp_kernel, q.device)
        with sdp_ctx:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        out = attn.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
        out = self.proj(out)
        if self.dropout:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def sinkhorn_batched(inp: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    out = inp
    for _ in range(iters):
        out = out / (out.sum(dim=-1, keepdim=True) + eps)
        out = out / (out.sum(dim=-2, keepdim=True) + eps)
    return out


class MHCResidual(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int,
        sinkhorn_iters: int,
        alpha_init: float,
        rmsnorm_eps: float,
        sinkhorn_eps: float,
        layer_fn: nn.Module,
        use_dynamic_h: bool,
        use_fused_mhc: bool,
        recompute_mhc: bool,
    ) -> None:
        super().__init__()
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.alpha_init = alpha_init
        self.rmsnorm_eps = rmsnorm_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.layer_fn = layer_fn
        self.use_dynamic_h = use_dynamic_h
        self.use_fused_mhc = bool(use_fused_mhc and mhc_cuda is not None)
        self.recompute_mhc = recompute_mhc
        if self.use_fused_mhc and self.sinkhorn_eps != self.rmsnorm_eps:
            self.sinkhorn_eps = self.rmsnorm_eps
        self.rmsnorm = RMSNorm(hidden_dim, rmsnorm_eps)

        n = expansion_rate
        nC = n * hidden_dim

        if use_dynamic_h:
            self.phi_pre = nn.Parameter(torch.randn(n, nC) * 0.02)
            self.phi_post = nn.Parameter(torch.randn(n, nC) * 0.02)
            self.phi_res = nn.Parameter(torch.randn(n * n, nC) * 0.02)
            self.b_pre = nn.Parameter(torch.zeros(n))
            self.b_post = nn.Parameter(torch.zeros(n))
            self.b_res = nn.Parameter(torch.zeros(n, n))
            self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
            self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
            self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.H_pre = nn.Parameter(torch.zeros(n))
            self.H_post = nn.Parameter(torch.zeros(n))
            self.H_res = nn.Parameter(torch.randn(n, n) * alpha_init)

    def _compute_dynamic_h_python(
        self, x_stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, n, c = x_stream.shape
        nC = n * c
        x_flat = x_stream.reshape(bsz * seq_len, nC)
        rms = x_flat.pow(2).mean(dim=-1, keepdim=True).add(self.rmsnorm_eps).sqrt()
        x_norm = x_flat / rms

        phi_pre = self.phi_pre.float()
        phi_post = self.phi_post.float()
        phi_res = self.phi_res.float()

        p_pre = x_norm @ phi_pre.t()
        p_post = x_norm @ phi_post.t()
        p_res = x_norm @ phi_res.t()

        tilde_pre = self.alpha_pre.float() * p_pre + self.b_pre.float()
        tilde_post = self.alpha_post.float() * p_post + self.b_post.float()
        tilde_res = self.alpha_res.float() * p_res + self.b_res.float().view(1, n * n)
        tilde_res = torch.clamp(tilde_res, max=H_RES_EXP_CLAMP)

        H_pre = torch.sigmoid(tilde_pre)
        H_post = 2.0 * torch.sigmoid(tilde_post)

        H_res = tilde_res.view(bsz * seq_len, n, n)
        M = sinkhorn_batched(torch.exp(H_res), self.sinkhorn_iters, self.sinkhorn_eps)

        return H_pre, H_post, M

    def _compute_dynamic_h_fused(
        self, x_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not x_tokens.is_cuda:
            raise RuntimeError("fused mHC requires CUDA tensors")
        if mhc_cuda is None:
            raise RuntimeError("mhc_cuda is required for fused mHC")
        return DynamicHFusedFunction.apply(
            x_tokens,
            self.phi_pre,
            self.phi_post,
            self.phi_res,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            self.b_pre,
            self.b_post,
            self.b_res,
            self.sinkhorn_iters,
            self.sinkhorn_eps,
        )

    def _compute_static_h(
        self, bsz: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.expansion_rate
        H_pre = torch.sigmoid(self.H_pre.float()).view(1, n).expand(bsz * seq_len, n)
        H_post = 2.0 * torch.sigmoid(self.H_post.float()).view(1, n).expand(
            bsz * seq_len, n
        )
        H_res = torch.clamp(self.H_res.float(), max=H_RES_EXP_CLAMP)
        M = sinkhorn_batched(
            torch.exp(H_res).view(1, n, n).expand(bsz * seq_len, n, n),
            self.sinkhorn_iters,
            self.sinkhorn_eps,
        )
        return H_pre, H_post, M

    def _forward_recompute(self, x_stream: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n, c = x_stream.shape

        def _pre(x_in: torch.Tensor):
            x_f32 = x_in.float()
            if self.use_dynamic_h:
                H_pre, H_post, M = self._compute_dynamic_h_python(x_f32)
            else:
                H_pre, H_post, M = self._compute_static_h(bsz, seq_len)
            x_flat = x_f32.reshape(bsz * seq_len, n, c)
            x_agg = torch.einsum("bn,bnc->bc", H_pre, x_flat)
            x_agg = x_agg.view(bsz, seq_len, c)
            x_norm = self.rmsnorm(x_agg)
            return x_norm, H_post, M

        if self.training and torch.is_grad_enabled():
            x_norm, H_post, M = _checkpoint_fn(_pre, x_stream)
        else:
            x_norm, H_post, M = _pre(x_stream)

        layer_in = x_norm
        if layer_in.dtype != x_stream.dtype:
            layer_in = layer_in.to(x_stream.dtype)
        y = self.layer_fn(layer_in)

        def _mix(
            x_in: torch.Tensor,
            y_in: torch.Tensor,
            H_post_in: torch.Tensor,
            M_in: torch.Tensor,
        ) -> torch.Tensor:
            x_f32 = x_in.float()
            x_flat = x_f32.reshape(bsz * seq_len, n, c)
            y_flat = y_in.reshape(bsz * seq_len, c).float()
            y_dist = H_post_in.unsqueeze(-1) * y_flat.unsqueeze(1)
            x_mixed = torch.einsum("bij,bjc->bic", M_in, x_flat)
            out = x_mixed + y_dist
            return out.view(bsz, seq_len, n, c)

        if self.training and torch.is_grad_enabled():
            out = _checkpoint_fn(_mix, x_stream, y, H_post, M)
        else:
            out = _mix(x_stream, y, H_post, M)
        return out.to(x_stream.dtype)

    def forward(self, x_stream: torch.Tensor) -> torch.Tensor:
        if self.recompute_mhc and self.training and torch.is_grad_enabled():
            return self._forward_recompute(x_stream)
        bsz, seq_len, n, c = x_stream.shape
        use_fused = self.use_fused_mhc and x_stream.is_cuda and mhc_cuda is not None
        if use_fused:
            x_tokens = x_stream.reshape(bsz * seq_len, n, c).contiguous()
        else:
            x_stream_f32 = x_stream.float()
            x_tokens = x_stream_f32.reshape(bsz * seq_len, n, c).contiguous()
        if self.use_dynamic_h:
            if use_fused:
                H_pre, H_post, M = self._compute_dynamic_h_fused(x_tokens)
            else:
                H_pre, H_post, M = self._compute_dynamic_h_python(x_stream_f32)
        else:
            H_pre, H_post, M = self._compute_static_h(bsz, seq_len)

        if use_fused:
            H_pre = H_pre.contiguous()
            H_post = H_post.contiguous()
            M = M.contiguous()
            x_agg_flat = AggregateFusedFunction.apply(x_tokens, H_pre)
            if mhc_ops is not None:
                x_norm_flat = mhc_ops.rmsnorm(
                    x_agg_flat, self.rmsnorm.weight, self.rmsnorm_eps
                )
            else:
                x_norm_flat = self.rmsnorm(x_agg_flat.float())
            layer_in = x_norm_flat.reshape(bsz, seq_len, c)
            if layer_in.dtype != x_stream.dtype:
                layer_in = layer_in.to(x_stream.dtype)
            y = self.layer_fn(layer_in)
            y_flat = y.reshape(bsz * seq_len, c).contiguous()
            y_bf16 = y_flat.to(torch.bfloat16).contiguous()
            out_flat = MixFusedFunction.apply(x_tokens, y_bf16, H_post, M)
            return out_flat.reshape(bsz, seq_len, n, c).to(x_stream.dtype)

        x_flat = x_tokens
        x_agg = torch.einsum("bn,bnc->bc", H_pre, x_flat)
        x_agg = x_agg.view(bsz, seq_len, c)
        x_norm = self.rmsnorm(x_agg)
        layer_in = x_norm
        if layer_in.dtype != x_stream.dtype:
            layer_in = layer_in.to(x_stream.dtype)
        y = self.layer_fn(layer_in)
        y_flat = y.reshape(bsz * seq_len, c).float()

        y_dist = H_post.unsqueeze(-1) * y_flat.unsqueeze(1)
        x_mixed = torch.einsum("bij,bjc->bic", M, x_flat)
        out = x_mixed + y_dist
        return out.view(bsz, seq_len, n, c).to(x_stream.dtype)


class MHCTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        attn = CausalSelfAttention(
            config.hidden_dim,
            config.n_heads,
            config.rope_dim,
            config.rope_theta,
            config.dropout,
            config.sdp_kernel,
        )
        mlp = MLP(config.hidden_dim, config.ffn_dim, config.dropout)

        self.attn_resid = MHCResidual(
            config.hidden_dim,
            config.expansion_rate,
            config.sinkhorn_iters,
            config.alpha_init,
            config.rmsnorm_eps,
            config.sinkhorn_eps,
            attn,
            config.use_dynamic_h,
            config.use_fused_mhc,
            config.recompute_mhc,
        )
        self.mlp_resid = MHCResidual(
            config.hidden_dim,
            config.expansion_rate,
            config.sinkhorn_iters,
            config.alpha_init,
            config.rmsnorm_eps,
            config.sinkhorn_eps,
            mlp,
            config.use_dynamic_h,
            config.use_fused_mhc,
            config.recompute_mhc,
        )

    def forward(self, x_stream: torch.Tensor) -> torch.Tensor:
        x_stream = self.attn_resid(x_stream)
        x_stream = self.mlp_resid(x_stream)
        return x_stream


class MHCTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.recompute_ratio = max(0.0, min(1.0, config.recompute_ratio))
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [MHCTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_dim, config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._checkpoint_indices = self._build_checkpoint_indices(
            len(self.blocks), self.recompute_ratio
        )

    @staticmethod
    def _build_checkpoint_indices(num_layers: int, ratio: float) -> set[int]:
        if ratio <= 0.0 or num_layers <= 0:
            return set()
        if ratio >= 1.0:
            return set(range(num_layers))
        num_checkpoint = max(1, int(round(num_layers * ratio)))
        indices = set()
        for i in range(num_checkpoint):
            idx = int(i * num_layers / num_checkpoint)
            indices.add(min(idx, num_layers - 1))
        return indices

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("seq_len exceeds config.max_seq_len")
        x = self.token_emb(input_ids)
        x_stream = (
            x.unsqueeze(2)
            .expand(bsz, seq_len, self.config.expansion_rate, self.config.hidden_dim)
            .contiguous()
        )
        use_checkpoint = (
            self.recompute_ratio > 0.0
            and self.training
            and torch.is_grad_enabled()
            and self._checkpoint_indices
        )
        if use_checkpoint:
            for idx, block in enumerate(self.blocks):
                if idx in self._checkpoint_indices:
                    x_stream = _checkpoint_block(block, x_stream)
                else:
                    x_stream = block(x_stream)
        else:
            for block in self.blocks:
                x_stream = block(x_stream)
        x_out = x_stream.mean(dim=2)
        x_out = self.final_norm(x_out)
        if return_hidden:
            return x_out
        return self.lm_head(x_out)
