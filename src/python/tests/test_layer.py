import torch
import pytest

torch.manual_seed(42)


def test_layer_forward():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    x = torch.randn(B, n, C, device="cuda")
    out = layer(x)

    assert out.shape == (B, n, C)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_backward():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


def test_layer_parameters():
    from mhc import MHCLayer

    C, n = 128, 4
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    params = dict(layer.named_parameters())
    assert "rmsnorm_weight" in params
    assert "H_pre" in params
    assert "H_post" in params
    assert "H_res" in params

    assert params["rmsnorm_weight"].shape == (C,)
    assert params["H_pre"].shape == (n,)
    assert params["H_post"].shape == (n,)
    assert params["H_res"].shape == (n, n)


def test_layer_training_step():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    x = torch.randn(B, n, C, device="cuda")
    target = torch.randn(B, n, C, device="cuda")

    for _ in range(3):
        optimizer.zero_grad()
        out = layer(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    assert not torch.isnan(loss)


def test_layer_different_batch_sizes():
    from mhc import MHCLayer

    n, C = 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    for B in [1, 4, 16, 32]:
        x = torch.randn(B, n, C, device="cuda")
        out = layer(x)
        assert out.shape == (B, n, C)


def test_layer_numerical_stability():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    x_large = torch.randn(B, n, C, device="cuda") * 100
    out = layer(x_large)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    x_small = torch.randn(B, n, C, device="cuda") * 0.001
    out = layer(x_small)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_deterministic():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    x = torch.randn(B, n, C, device="cuda")
    out1 = layer(x)
    out2 = layer(x)

    assert torch.allclose(out1, out2)


def test_layer_gradient_flow():
    from mhc import MHCLayer

    B, n, C = 8, 4, 128
    layer = MHCLayer(hidden_dim=C, expansion_rate=n).cuda()

    x = (torch.randn(B, n, C, device="cuda") + 0.1).requires_grad_(True)
    out = layer(x)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    assert x.grad is not None
    grad_norm = x.grad.norm()
    assert grad_norm > 0, "Gradient should flow through"
    assert grad_norm < 1e6, "Gradient should not explode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
