import torch
import torch.nn as nn
from .ops import (
    sinkhorn_knopp,
    rmsnorm,
    stream_aggregate,
    stream_distribute,
    stream_mix,
)


class MHCLayer(nn.Module):
    """
    Based on the following paper: "mHC: Manifold-Constrained Hyper-Connections" (DeepSeek-AI, 2025)
    https://arxiv.org/abs/2512.24880
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))
        self.H_pre = nn.Parameter(
            torch.ones(expansion_rate, dtype=torch.float32) / expansion_rate
        )
        self.H_post = nn.Parameter(torch.ones(expansion_rate, dtype=torch.float32))
        H_res_init = torch.eye(expansion_rate) + alpha_init * torch.randn(
            expansion_rate, expansion_rate
        )
        self.H_res = nn.Parameter(H_res_init.float())

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape
        assert n == self.expansion_rate
        assert C == self.hidden_dim

        x = x_expanded.float().contiguous()

        x_agg = stream_aggregate(x, self.H_pre)
        y_norm = rmsnorm(x_agg, self.rmsnorm_weight, self.eps)
        y_dist = stream_distribute(y_norm.float(), self.H_post)
        M = sinkhorn_knopp(self.H_res, self.sinkhorn_iters, self.eps)
        x_mixed = stream_mix(x, M)
        out = x_mixed + y_dist

        return out
