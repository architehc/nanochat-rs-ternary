"""
ternary_qat.py — Ternary Quantization-Aware Training for nanochat-rs

Implements BitLinear with Straight-Through Estimator (STE) for training
with ternary {-1, 0, +1} weight quantization and INT8 activation quantization.

Quantization scheme:
  - Weights: per-group absmean scaling → ternary {-1, 0, +1}
  - Activations: per-token absmax scaling → INT8 [-127, 127]
  - Both use STE for gradient flow through quantization

References:
  BitNet b1.58: arXiv 2402.17764
  1-bit LLMs: arXiv 2310.11453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# Weight Quantization: absmean ternary
# =============================================================================

def absmean_quantize(w: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 weights to ternary {-1, 0, +1} with per-group absmean scales.

    Algorithm (BitNet b1.58):
      1. Reshape weights into groups of `group_size`
      2. Per-group scale = mean(|w|) (clamped to avoid division by zero)
      3. Normalize: w_scaled = w / scale
      4. Round + clip to {-1, 0, +1}

    Args:
        w: Weight tensor of any shape. Total elements must be divisible by group_size.
        group_size: Number of elements per quantization group (default 128).

    Returns:
        w_ternary: Same shape as w, values in {-1.0, 0.0, +1.0}
        scales: [n_groups] tensor of per-group scale factors (f32)
    """
    orig_shape = w.shape
    numel = w.numel()

    # Handle case where total elements not divisible by group_size:
    # pad with zeros, quantize, then trim
    if numel % group_size != 0:
        pad_len = group_size - (numel % group_size)
        w_flat = F.pad(w.reshape(-1), (0, pad_len), value=0.0)
    else:
        w_flat = w.reshape(-1)
        pad_len = 0

    n_groups = w_flat.shape[0] // group_size
    w_grouped = w_flat.reshape(n_groups, group_size)

    # Per-group scale = mean absolute value
    scales = w_grouped.abs().mean(dim=-1).clamp(min=1e-8)  # [n_groups]

    # Normalize, round, clip to ternary
    w_scaled = w_grouped / scales.unsqueeze(-1)
    w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)

    # Remove padding and reshape
    if pad_len > 0:
        w_ternary = w_ternary.reshape(-1)[:numel].reshape(orig_shape)
    else:
        w_ternary = w_ternary.reshape(orig_shape)

    return w_ternary, scales


def dequantize_ternary(w_ternary: torch.Tensor, scales: torch.Tensor,
                       group_size: int = 128) -> torch.Tensor:
    """Reconstruct FP32 weights from ternary values + scales.

    Args:
        w_ternary: Ternary weight tensor {-1, 0, +1}
        scales: Per-group scale factors [n_groups]
        group_size: Elements per group

    Returns:
        Reconstructed FP32 weight tensor (same shape as w_ternary)
    """
    orig_shape = w_ternary.shape
    numel = w_ternary.numel()

    if numel % group_size != 0:
        pad_len = group_size - (numel % group_size)
        w_flat = F.pad(w_ternary.reshape(-1), (0, pad_len), value=0.0)
    else:
        w_flat = w_ternary.reshape(-1)
        pad_len = 0

    n_groups = w_flat.shape[0] // group_size
    w_grouped = w_flat.reshape(n_groups, group_size)

    # Multiply by scales
    w_recon = w_grouped * scales.unsqueeze(-1)

    if pad_len > 0:
        return w_recon.reshape(-1)[:numel].reshape(orig_shape)
    else:
        return w_recon.reshape(orig_shape)


# =============================================================================
# Activation Quantization: per-token absmax INT8
# =============================================================================

def per_token_absmax_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to INT8 using per-token absmax scaling.

    For each token (last dimension), find absmax and scale to [-127, 127].

    Args:
        x: Activation tensor [..., dim]

    Returns:
        x_q: Quantized INT8 tensor (as float for STE), same shape
        scales: Per-token scale factors [..., 1]
    """
    # absmax over last dimension
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = absmax / 127.0  # [..., 1]
    inv_scales = 127.0 / absmax

    # Quantize: scale, round, clip
    x_q = torch.clamp(torch.round(x * inv_scales), -127.0, 127.0)

    return x_q, scales


# =============================================================================
# BitLinear with STE: the core ternary training layer
# =============================================================================

class BitLinearSTE(nn.Module):
    """Linear layer with ternary weight quantization and INT8 activation quantization,
    using Straight-Through Estimator (STE) for gradient flow.

    During forward:
      1. Weights: FP32 shadow -> ternary {-1,0,+1} via absmean (STE bypass)
      2. Activations: FP32 -> INT8 via per-token absmax (STE bypass)
      3. Compute: F.linear(x_ste, w_ste)

    During backward:
      STE passes gradients through as if quantization were identity.
      This allows the FP32 shadow weights to accumulate gradients.

    At export:
      Extract w_ternary and scales from the FP32 shadow weights.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # FP32 shadow weights (trained with optimizer)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with STE quantization.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # --- Weight quantization with STE ---
        w_ternary, scales = absmean_quantize(self.weight, self.group_size)

        # Reconstruct the dequantized weight
        w_deq = dequantize_ternary(w_ternary, scales, self.group_size)

        # STE: forward uses quantized, backward passes through to FP32 shadow
        # w_ste = self.weight + (w_deq - self.weight).detach()
        #       = w_deq in forward, grad flows to self.weight in backward
        w_ste = self.weight + (w_deq - self.weight).detach()

        # --- Activation quantization with STE ---
        x_q, act_scales = per_token_absmax_quantize(x)

        # Dequantize: x_deq = x_q * act_scales (back to approximate FP32)
        x_deq = x_q * act_scales

        # STE for activations
        x_ste = x + (x_deq - x).detach()

        # --- Linear operation ---
        return F.linear(x_ste, w_ste, self.bias)

    def get_ternary_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract ternary weights and scales for export.

        Returns:
            w_ternary: Ternary weight tensor {-1, 0, +1}, shape [out, in]
            scales: Per-group scales, shape [n_groups]
        """
        with torch.no_grad():
            return absmean_quantize(self.weight, self.group_size)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'group_size={self.group_size}, bias={self.bias is not None}')


# =============================================================================
# Utility: pack ternary weights to 2-bit format
# =============================================================================

def pack_ternary_2bit(w_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary {-1, 0, +1} weights into 2-bit representation.

    Encoding (BitNet standard, GGUF-compatible):
      -1 -> 0b11
       0 -> 0b00
      +1 -> 0b01
      0b10 -> invalid

    Packs 4 trits per byte.

    Args:
        w_ternary: Tensor of ternary values {-1.0, 0.0, +1.0}

    Returns:
        packed: uint8 tensor, 4x smaller
    """
    w_flat = w_ternary.reshape(-1).to(torch.int8)

    # Pad to multiple of 4
    numel = w_flat.shape[0]
    if numel % 4 != 0:
        pad_len = 4 - (numel % 4)
        w_flat = F.pad(w_flat.float(), (0, pad_len), value=0.0).to(torch.int8)

    n_bytes = w_flat.shape[0] // 4
    w_groups = w_flat.reshape(n_bytes, 4)

    # Encode each value to 2 bits: -1->11, 0->00, +1->01
    # Using: encoded = value & 0b11 (works because -1 in int8 = 0xFF, & 0x3 = 0x3)
    encoded = w_groups.to(torch.uint8) & 0x03

    # Pack 4 trits into 1 byte: [t0, t1, t2, t3] -> t0 | (t1<<2) | (t2<<4) | (t3<<6)
    packed = (encoded[:, 0]
              | (encoded[:, 1] << 2)
              | (encoded[:, 2] << 4)
              | (encoded[:, 3] << 6))

    return packed


def unpack_ternary_2bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack 2-bit packed ternary weights.

    Args:
        packed: uint8 tensor of packed trits
        numel: number of original elements

    Returns:
        w_ternary: Tensor of values {-1.0, 0.0, +1.0}
    """
    # Extract 4 trits per byte
    t0 = packed & 0x03
    t1 = (packed >> 2) & 0x03
    t2 = (packed >> 4) & 0x03
    t3 = (packed >> 6) & 0x03

    # Interleave: [t0_0, t1_0, t2_0, t3_0, t0_1, ...]
    w_flat = torch.stack([t0, t1, t2, t3], dim=-1).reshape(-1)[:numel]

    # Decode: 00->0, 01->+1, 11->-1, 10->0 (invalid)
    result = torch.zeros_like(w_flat, dtype=torch.float32)
    result[w_flat == 0x01] = 1.0
    result[w_flat == 0x03] = -1.0
    # 0x00 -> 0.0 (already set), 0x02 -> 0.0 (invalid, decode as 0)

    return result


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    """Run verification checks for ternary QAT."""
    print("=" * 60)
    print("Ternary QAT Self-Test")
    print("=" * 60)

    # --- absmean quantization test ---
    print("\n[1] Testing absmean quantization...")
    torch.manual_seed(42)
    w = torch.randn(256, 128) * 0.5
    w_t, scales = absmean_quantize(w, group_size=128)

    # Check ternary
    unique_vals = w_t.unique().sort().values
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals.tolist()), \
        f"Non-ternary values: {unique_vals}"
    print(f"  Unique values: {unique_vals.tolist()}")
    print(f"  Scales shape: {scales.shape}, mean={scales.mean():.4f}")

    # Check roundtrip
    w_recon = dequantize_ternary(w_t, scales, group_size=128)
    err = (w - w_recon).abs().max().item()
    print(f"  Max reconstruction error: {err:.4f}")

    # --- Activation quantization test ---
    print("\n[2] Testing per-token absmax quantization...")
    x = torch.randn(4, 16, 256)
    x_q, act_scales = per_token_absmax_quantize(x)
    assert x_q.shape == x.shape
    assert act_scales.shape == (4, 16, 1)
    assert x_q.abs().max() <= 127.0
    x_deq = x_q * act_scales
    err = (x - x_deq).abs().max().item()
    print(f"  Max quantization error: {err:.4f}")
    print(f"  x_q range: [{x_q.min():.0f}, {x_q.max():.0f}]")

    # --- BitLinearSTE forward + backward ---
    print("\n[3] Testing BitLinearSTE forward/backward...")
    layer = BitLinearSTE(128, 256, group_size=128)
    x = torch.randn(2, 8, 128, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 8, 256), f"Bad output shape: {y.shape}"

    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None, "No gradient for weight"
    assert x.grad is not None, "No gradient for input"
    print(f"  Output shape: {y.shape}")
    print(f"  Weight grad norm: {layer.weight.grad.norm():.4f}")
    print(f"  Input grad norm: {x.grad.norm():.4f}")

    # --- STE gradient check: gradients should be non-trivial ---
    print("\n[4] Testing STE gradient flow quality...")
    layer2 = BitLinearSTE(64, 64, group_size=64)
    x2 = torch.randn(1, 1, 64, requires_grad=True)
    y2 = layer2(x2)
    y2.sum().backward()
    # Gradient should not be all zeros (STE is working)
    assert layer2.weight.grad.abs().sum() > 0, "STE gradient is zero!"
    # Gradient should not be all same value (diversity check)
    grad_std = layer2.weight.grad.std()
    print(f"  Grad std: {grad_std:.6f} (should be > 0)")
    assert grad_std > 1e-8, "Gradient is constant (STE may be broken)"

    # --- 2-bit packing test ---
    print("\n[5] Testing 2-bit pack/unpack roundtrip...")
    w_t = torch.tensor([-1., 0., 1., -1., 0., 1., 0., -1.])
    packed = pack_ternary_2bit(w_t)
    w_rt = unpack_ternary_2bit(packed, len(w_t))
    assert torch.equal(w_t, w_rt), f"Roundtrip failed: {w_t} != {w_rt}"
    print(f"  Input:  {w_t.tolist()}")
    print(f"  Packed: {packed.tolist()} ({len(packed)} bytes for {len(w_t)} trits)")
    print(f"  Output: {w_rt.tolist()}")

    # Large random roundtrip
    torch.manual_seed(123)
    w_big = torch.randn(1024, 512)
    w_big_t, _ = absmean_quantize(w_big, 128)
    packed_big = pack_ternary_2bit(w_big_t)
    w_big_rt = unpack_ternary_2bit(packed_big, w_big_t.numel()).reshape(w_big_t.shape)
    assert torch.equal(w_big_t, w_big_rt), "Large roundtrip failed"
    print(f"  Large tensor: {w_big_t.shape} -> {packed_big.shape[0]} bytes -> roundtrip OK")

    print("\n" + "=" * 60)
    print("ALL TERNARY QAT TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    _self_test()
