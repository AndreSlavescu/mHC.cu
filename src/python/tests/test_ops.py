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


def test_stream_aggregate():
    from mhc import stream_aggregate

    B, n, C = 8, 4, 128
    inp = torch.randn(B, n, C, device="cuda")
    H_pre = torch.ones(n, device="cuda") / n

    out = stream_aggregate(inp, H_pre)
    expected = (inp * H_pre.view(1, n, 1)).sum(dim=1)

    assert out.shape == (B, C)
    assert torch.allclose(out, expected, atol=1e-5)


def test_stream_aggregate_gradient():
    from mhc import stream_aggregate

    B, n, C = 8, 4, 128
    inp = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    H_pre = torch.randn(n, device="cuda", requires_grad=True)

    out = stream_aggregate(inp, H_pre)
    loss = out.sum()
    loss.backward()

    assert inp.grad is not None
    assert H_pre.grad is not None
    assert not torch.isnan(inp.grad).any()
    assert not torch.isnan(H_pre.grad).any()


def test_stream_distribute():
    from mhc import stream_distribute

    B, n, C = 8, 4, 128
    inp = torch.randn(B, C, device="cuda")
    H_post = torch.ones(n, device="cuda")

    out = stream_distribute(inp, H_post)
    expected = inp.unsqueeze(1) * H_post.view(1, n, 1)

    assert out.shape == (B, n, C)
    assert torch.allclose(out, expected, atol=1e-5)


def test_stream_distribute_gradient():
    from mhc import stream_distribute

    B, n, C = 8, 4, 128
    inp = (torch.randn(B, C, device="cuda") + 0.1).requires_grad_(True)
    H_post = torch.randn(n, device="cuda", requires_grad=True)

    out = stream_distribute(inp, H_post)
    loss = out.sum()
    loss.backward()

    assert inp.grad is not None
    assert H_post.grad is not None


def test_stream_mix():
    from mhc import stream_mix

    B, n, C = 8, 4, 128
    inp = torch.randn(B, n, C, device="cuda")
    M = torch.eye(n, device="cuda")

    out = stream_mix(inp, M)

    assert out.shape == (B, n, C)
    assert torch.allclose(out, inp, atol=1e-5)


def test_stream_mix_gradient():
    from mhc import stream_mix

    B, n, C = 8, 4, 128
    inp = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    M = torch.randn(n, n, device="cuda", requires_grad=True)

    out = stream_mix(inp, M)
    loss = out.sum()
    loss.backward()

    assert inp.grad is not None
    assert M.grad is not None


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
    from mhc import sinkhorn_knopp, stream_aggregate, stream_distribute, stream_mix

    inp = torch.rand(32, 32, device="cuda") * 1000
    out = sinkhorn_knopp(inp, num_iters=50)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    B, n, C = 8, 4, 128
    x = torch.randn(B, n, C, device="cuda") * 100
    H = torch.randn(n, device="cuda")

    out = stream_aggregate(x, H)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    y = torch.randn(B, C, device="cuda") * 100
    out = stream_distribute(y, H)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    M = torch.randn(n, n, device="cuda") * 10
    out = stream_mix(x, M)
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
