import torch
import pytest

torch.manual_seed(42)


def test_sinkhorn_knopp_doubly_stochastic():
    from mhc import sinkhorn_knopp

    inp = torch.rand(32, 32, device="cuda") + 0.1
    out = sinkhorn_knopp(inp, num_iters=20)

    row_sums = out.sum(dim=1)
    col_sums = out.sum(dim=0)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)
    assert (out >= 0).all()


def test_sinkhorn_knopp_gradient():
    from mhc import sinkhorn_knopp

    inp = (torch.rand(16, 16, device="cuda") + 0.1).requires_grad_(True)
    out = sinkhorn_knopp(inp, num_iters=10)
    loss = out.sum()
    loss.backward()

    assert inp.grad is not None
    assert not torch.isnan(inp.grad).any()
    assert not torch.isinf(inp.grad).any()


def test_rmsnorm():
    from mhc import rmsnorm

    B, C = 8, 128
    inp = torch.randn(B, C, device="cuda")
    weight = torch.ones(C, device="cuda")

    out = rmsnorm(inp, weight)

    assert out.shape == (B, C)
    assert out.dtype == torch.bfloat16


def test_rmsnorm_gradient():
    from mhc import rmsnorm

    B, C = 8, 128
    inp = (torch.randn(B, C, device="cuda") + 0.1).requires_grad_(True)
    weight = torch.ones(C, device="cuda", requires_grad=True)

    out = rmsnorm(inp, weight)
    loss = out.float().sum()
    loss.backward()

    assert inp.grad is not None
    assert weight.grad is not None


def test_numerical_stability_large_values():
    from mhc import sinkhorn_knopp

    inp = torch.rand(32, 32, device="cuda") * 1000
    out = sinkhorn_knopp(inp, num_iters=50)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_numerical_stability_small_values():
    from mhc import sinkhorn_knopp

    inp = torch.rand(32, 32, device="cuda") * 1e-6 + 1e-8
    out = sinkhorn_knopp(inp, num_iters=20)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
