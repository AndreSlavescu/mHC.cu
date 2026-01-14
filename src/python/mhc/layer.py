import torch
import torch.nn as nn
from .ops import (
    mhc_layer_fused,
    mhc_layer_fused_dynamic,
    mhc_layer_fused_dynamic_inference,
    mhc_layer_fused_inference,
)


class MHCLayer(nn.Module):
    """
    Args:
        hidden_dim: The hidden dimension (C).
        expansion_rate: The expansion rate (n).
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations.
        eps: Epsilon for numerical stability.
        alpha_init: Initialization scale for alpha parameters.
        use_dynamic_h: If True, uses per-batch H values computed from x.
                       If False, uses static shared H values.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
        use_dynamic_h: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.use_dynamic_h = use_dynamic_h

        n = expansion_rate
        C = hidden_dim
        nC = n * C

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))

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
            self.H_pre = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
            self.H_post = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
            H_res_init = alpha_init * torch.randn(expansion_rate, expansion_rate)
            self.H_res = nn.Parameter(H_res_init.float())

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape
        assert n == self.expansion_rate
        assert C == self.hidden_dim

        if self.use_dynamic_h:
            if not self.training and not torch.is_grad_enabled():
                return mhc_layer_fused_dynamic_inference(
                    x_expanded,
                    self.rmsnorm_weight,
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
                    self.eps,
                )
            return mhc_layer_fused_dynamic(
                x_expanded,
                self.rmsnorm_weight,
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
                self.eps,
            )
        else:
            if not self.training and not torch.is_grad_enabled():
                return mhc_layer_fused_inference(
                    x_expanded,
                    self.rmsnorm_weight,
                    self.H_pre,
                    self.H_post,
                    self.H_res,
                    self.sinkhorn_iters,
                    self.eps,
                )
            return mhc_layer_fused(
                x_expanded,
                self.rmsnorm_weight,
                self.H_pre,
                self.H_post,
                self.H_res,
                self.sinkhorn_iters,
                self.eps,
            )
