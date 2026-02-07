"""
test_optimizers.py — Tests for Muon + Lion optimizers

Tests:
  - test_muon_step: verify params change after a step on a small random 2D parameter
  - test_lion_step: verify params change after a step on a small random 1D parameter
  - test_newton_schulz_orthogonal: verify output has approximately orthogonal rows
  - test_muon_lion_training_loop: 10 steps with a tiny model config, verify loss decreases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from optimizers import Muon, Lion, newton_schulz_orthogonalize


def test_muon_step():
    """Verify Muon updates a 2D parameter after one step."""
    torch.manual_seed(42)
    param = nn.Parameter(torch.randn(64, 32))
    initial = param.data.clone()

    opt = Muon([param], lr=0.02, momentum=0.95, ns_steps=5)

    # Simulate a gradient
    param.grad = torch.randn_like(param)
    opt.step()

    # Parameter must have changed
    assert not torch.allclose(param.data, initial), "Muon did not update 2D parameter"
    # Parameter must be finite
    assert param.data.isfinite().all(), "Muon produced non-finite values"


def test_lion_step():
    """Verify Lion updates a 1D parameter after one step."""
    torch.manual_seed(42)
    param = nn.Parameter(torch.randn(128))
    initial = param.data.clone()

    opt = Lion([param], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)

    # Simulate a gradient
    param.grad = torch.randn_like(param)
    opt.step()

    # Parameter must have changed
    assert not torch.allclose(param.data, initial), "Lion did not update 1D parameter"
    # Lion uses sign updates, so the update magnitude should be approximately lr
    # (small FP32 rounding errors are expected at the 1e-4 scale)
    diff = (param.data - initial).abs()
    assert torch.allclose(diff, torch.full_like(diff, 1e-4), atol=1e-6), \
        "Lion update magnitude should be approximately lr"


def test_newton_schulz_orthogonal():
    """Verify Newton-Schulz produces approximately orthogonal rows.

    The quintic NS iteration with coefficients (3.4445, -4.7750, 2.0315) is
    tuned for speed over exact convergence. After 5 steps, singular values
    are approximately in [0.5, 1.5] (not exactly 1.0). This is by design --
    the Muon optimizer does not need exact orthogonality for good training.

    We test that:
    1. The output has the correct shape
    2. The output is finite
    3. The rows are significantly more orthogonal than the input
    4. G_orth @ G_orth^T is approximately diagonal (off-diag elements small)
    """
    torch.manual_seed(123)
    G = torch.randn(16, 32)

    G_orth = newton_schulz_orthogonalize(G, ns_steps=5)

    # Shape preserved
    assert G_orth.shape == G.shape

    # Output is finite
    assert G_orth.isfinite().all(), "NS produced non-finite values"

    # G_orth @ G_orth^T should be approximately proportional to identity
    GGT = G_orth @ G_orth.T
    I = torch.eye(16)

    # Off-diagonal elements should be small (near zero)
    off_diag_mask = ~torch.eye(16, dtype=torch.bool)
    off_diag_max = GGT[off_diag_mask].abs().max().item()
    assert off_diag_max < 0.5, f"Off-diagonal elements too large: {off_diag_max:.6f}"

    # Diagonal elements should be positive (rows have non-trivial norm)
    diag_min = GGT.diag().min().item()
    assert diag_min > 0.3, f"Diagonal too small: {diag_min:.6f}"


def test_newton_schulz_tall_matrix():
    """Verify Newton-Schulz works for tall matrices (rows > cols) too."""
    torch.manual_seed(456)
    G = torch.randn(32, 16)

    G_orth = newton_schulz_orthogonalize(G, ns_steps=5)

    assert G_orth.shape == G.shape
    assert G_orth.isfinite().all(), "NS produced non-finite values for tall matrix"

    # G_orth^T @ G_orth should be approximately proportional to identity
    GTG = G_orth.T @ G_orth
    off_diag_mask = ~torch.eye(16, dtype=torch.bool)
    off_diag_max = GTG[off_diag_mask].abs().max().item()
    assert off_diag_max < 0.5, f"Off-diagonal elements too large: {off_diag_max:.6f}"

    diag_min = GTG.diag().min().item()
    assert diag_min > 0.3, f"Diagonal too small: {diag_min:.6f}"


def test_lion_weight_decay():
    """Verify Lion applies multiplicative weight decay."""
    torch.manual_seed(42)
    param = nn.Parameter(torch.ones(64) * 2.0)
    initial = param.data.clone()

    opt = Lion([param], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)

    # Zero gradient: only weight decay should apply (plus sign(m) which is 0 initially)
    param.grad = torch.zeros_like(param)
    opt.step()

    # With zero grad and zero momentum, update = sign(0) = 0, so only WD applies
    # p = p * (1 - lr * wd) = 2.0 * (1 - 1e-4 * 0.1) = 2.0 * 0.99999
    expected = initial * (1.0 - 1e-4 * 0.1)
    assert torch.allclose(param.data, expected, atol=1e-7), \
        f"Weight decay not applied correctly: {param.data[0]:.8f} vs {expected[0]:.8f}"


def test_muon_lion_training_loop():
    """10 steps with a tiny nanochat model, verify loss decreases."""
    from model import NanochatConfig, NanochatTernary
    from train import build_muon_lion_optimizers

    torch.manual_seed(42)

    # Tiny config for fast test
    config = NanochatConfig(
        dim=64, n_layers=2, n_heads=2, ffn_mult=2.0,
        vocab_size=256, max_seq_len=32, group_size=64,
        mhc_n_streams=2,
    )
    model = NanochatTernary(config)

    optimizers = build_muon_lion_optimizers(
        model, lr=0.02, mhc_lr=1e-3, weight_decay=0.0,
    )
    assert len(optimizers) == 2, f"Expected 2 optimizers (Muon + Lion), got {len(optimizers)}"
    assert isinstance(optimizers[0], Muon), f"First optimizer should be Muon, got {type(optimizers[0])}"
    assert isinstance(optimizers[1], Lion), f"Second optimizer should be Lion, got {type(optimizers[1])}"

    # Use a fixed batch of repeating tokens for a learnable pattern
    torch.manual_seed(123)
    tokens = torch.randint(0, config.vocab_size, (8, 16))

    # Run 20 training steps on the same batch (overfit deliberately)
    losses = []
    for step in range(20):
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size),
                               tokens[:, 1:].reshape(-1))
        loss.backward()

        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        losses.append(loss.item())

    # All losses should be finite
    assert all(l == l for l in losses), "NaN loss detected"
    assert all(l < float('inf') for l in losses), "Inf loss detected"

    # Loss should decrease over 20 steps when overfitting a fixed batch
    first_5_avg = sum(losses[:5]) / 5
    last_5_avg = sum(losses[-5:]) / 5
    assert last_5_avg < first_5_avg, \
        f"Loss did not decrease: first 5 avg={first_5_avg:.4f}, last 5 avg={last_5_avg:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
