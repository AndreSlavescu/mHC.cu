import pytest
import torch

from mhc.model import ModelConfig, MHCTransformer


def test_model_forward_backward_cpu():
    config = ModelConfig(
        vocab_size=128,
        n_layers=2,
        hidden_dim=64,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=16,
        expansion_rate=2,
        sinkhorn_iters=2,
        rmsnorm_eps=1e-5,
        sinkhorn_eps=1e-5,
        use_fused_mhc=False,
    )
    model = MHCTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_model_forward_backward_fused_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused mHC")
    try:
        import mhc_cuda  # noqa: F401
    except ImportError:
        pytest.skip("mhc_cuda extension not available")
    config = ModelConfig(
        vocab_size=128,
        n_layers=1,
        hidden_dim=64,
        n_heads=4,
        ffn_dim=128,
        max_seq_len=16,
        expansion_rate=2,
        sinkhorn_iters=2,
        rmsnorm_eps=1e-5,
        sinkhorn_eps=1e-5,
        use_fused_mhc=True,
    )
    model = MHCTransformer(config).cuda()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    logits = model(input_ids)
    assert logits.is_cuda
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None
