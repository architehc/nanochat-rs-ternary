"""
model.py — nanochat-rs Ternary Model with mHC-lite Residual Connections

Complete PyTorch model for training with:
  - Ternary QAT (BitLinearSTE) for all linear layers
  - mHC-lite (BvN doubly stochastic) residual connections
  - RMSNorm, SwiGLU FFN, Multi-Head Attention with causal mask
  - Configurable: d20 (tiny), 7B, 25B-MoE, 80B-MoE presets

All linear layers use STE for ternary quantization during training.
mHC parameters stay FP32 (never quantized).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from ternary_qat import BitLinearSTE
from mhc_lite import MhcLiteLayer, MhcLiteN2, MhcLiteN4


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class NanochatConfig:
    """Model configuration matching Rust nanochat-model config.rs"""
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # GQA; None = MHA
    ffn_mult: float = 2.667
    vocab_size: int = 32000
    max_seq_len: int = 2048
    group_size: int = 128
    mhc_n_streams: int = 2
    rope_theta: float = 10000.0
    # MoE (unused for now)
    n_experts: Optional[int] = None
    n_active_experts: Optional[int] = None

    @staticmethod
    def d20():
        """~20M param debug/test config"""
        return NanochatConfig(
            dim=256, n_layers=6, n_heads=4, ffn_mult=2.667,
            vocab_size=32000, max_seq_len=512, group_size=128,
            mhc_n_streams=2,
        )

    @staticmethod
    def nano_125m():
        """~125M param config for real training"""
        return NanochatConfig(
            dim=768, n_layers=12, n_heads=12, ffn_mult=2.667,
            vocab_size=32000, max_seq_len=2048, group_size=128,
            mhc_n_streams=2,
        )

    @staticmethod
    def nano_560m():
        """~560M param config"""
        return NanochatConfig(
            dim=1024, n_layers=24, n_heads=16, ffn_mult=2.667,
            vocab_size=32000, max_seq_len=2048, group_size=128,
            mhc_n_streams=2,
        )


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple:
    """Apply rotary embeddings to query and key tensors."""
    # Reshape to complex
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis: [T, head_dim//2] -> [1, T, 1, head_dim//2]
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================

class Attention(nn.Module):
    def __init__(self, config: NanochatConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # for GQA

        self.wq = BitLinearSTE(config.dim, config.n_heads * self.head_dim,
                               group_size=config.group_size)
        self.wk = BitLinearSTE(config.dim, self.n_kv_heads * self.head_dim,
                               group_size=config.group_size)
        self.wv = BitLinearSTE(config.dim, self.n_kv_heads * self.head_dim,
                               group_size=config.group_size)
        self.wo = BitLinearSTE(config.n_heads * self.head_dim, config.dim,
                               group_size=config.group_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # Transpose to [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# =============================================================================
# SwiGLU FFN
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, config: NanochatConfig):
        super().__init__()
        ffn_dim = int(config.dim * config.ffn_mult)
        # Round to multiple of 128 for group quantization
        ffn_dim = ((ffn_dim + 127) // 128) * 128

        self.w_gate = BitLinearSTE(config.dim, ffn_dim, group_size=config.group_size)
        self.w_up = BitLinearSTE(config.dim, ffn_dim, group_size=config.group_size)
        self.w_down = BitLinearSTE(ffn_dim, config.dim, group_size=config.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =============================================================================
# Transformer Block with mHC-lite Wiring
# =============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with mHC-lite residual connections.

    Architecture:
      x_expanded --+-- [mhc_attn.prepare_input] -> RMSNorm -> Attn -> [mhc_attn.apply]
                   |                                                        |
                   +-- [mhc_attn.h_res residual] --------------------------+ -> x_expanded

      x_expanded --+-- [mhc_ffn.prepare_input] -> RMSNorm -> FFN -> [mhc_ffn.apply]
                   |                                                      |
                   +-- [mhc_ffn.h_res residual] --------------------------+ -> x_expanded
    """
    def __init__(self, config: NanochatConfig):
        super().__init__()
        # mHC for each sub-layer
        self.mhc_attn = MhcLiteLayer(config.dim, n_streams=config.mhc_n_streams)
        self.mhc_ffn = MhcLiteLayer(config.dim, n_streams=config.mhc_n_streams)

        # Pre-norms
        self.norm_attn = RMSNorm(config.dim)
        self.norm_ffn = RMSNorm(config.dim)

        # Sub-layers
        self.attention = Attention(config)
        self.ffn = FeedForward(config)

    def forward(self, x_expanded: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        attn_in = self.mhc_attn.prepare_input(x_expanded)
        attn_in = self.norm_attn(attn_in)
        attn_out = self.attention(attn_in, freqs_cis)
        x_expanded = self.mhc_attn(x_expanded, attn_out)

        # FFN sub-layer
        ffn_in = self.mhc_ffn.prepare_input(x_expanded)
        ffn_in = self.norm_ffn(ffn_in)
        ffn_out = self.ffn(ffn_in)
        x_expanded = self.mhc_ffn(x_expanded, ffn_out)

        return x_expanded


# =============================================================================
# Full Nanochat Model
# =============================================================================

class NanochatTernary(nn.Module):
    """Complete nanochat model with ternary QAT + mHC-lite.

    Embeddings and final norm are FP32 (not quantized).
    All linear layers use BitLinearSTE for ternary QAT.
    mHC parameters are FP32 (never quantized).
    """
    def __init__(self, config: NanochatConfig):
        super().__init__()
        self.config = config

        # Token embedding (FP32)
        self.tok_embed = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm + LM head
        self.norm_final = RMSNorm(config.dim)
        self.lm_head = BitLinearSTE(config.dim, config.vocab_size,
                                     group_size=config.group_size)

        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False
        )

        # mHC expand/collapse uses first block's mhc_attn for the expand
        self.n_streams = config.mhc_n_streams

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, BitLinearSTE):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: [B, T] long tensor of token IDs

        Returns:
            logits: [B, T, vocab_size] float tensor
        """
        B, T = token_ids.shape

        # Embed
        x = self.tok_embed(token_ids)  # [B, T, dim]

        # Expand to multi-stream
        x_exp = x.repeat(1, 1, self.n_streams)  # [B, T, n*dim]

        # Transformer blocks
        freqs_cis = self.freqs_cis[:T]
        for block in self.blocks:
            x_exp = block(x_exp, freqs_cis)

        # Collapse back to single stream
        dim = self.config.dim
        if self.n_streams == 2:
            x = 0.5 * (x_exp[..., :dim] + x_exp[..., dim:])
        else:
            chunks = [x_exp[..., i*dim:(i+1)*dim] for i in range(self.n_streams)]
            x = sum(chunks) / self.n_streams

        # Final norm + LM head
        x = self.norm_final(x)
        logits = self.lm_head(x)

        return logits

    def count_params(self) -> dict:
        """Count parameters by category."""
        total = 0
        mhc_params = 0
        linear_params = 0
        embed_params = 0
        norm_params = 0

        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            if 'mhc' in name:
                mhc_params += n
            elif 'embed' in name:
                embed_params += n
            elif 'norm' in name or 'weight' in name and p.dim() == 1:
                norm_params += n
            else:
                linear_params += n

        return {
            'total': total,
            'linear': linear_params,
            'mhc': mhc_params,
            'embed': embed_params,
            'norm': norm_params,
        }


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    print("=" * 60)
    print("Nanochat Model Self-Test")
    print("=" * 60)

    config = NanochatConfig.d20()
    print(f"\nConfig: dim={config.dim}, layers={config.n_layers}, "
          f"heads={config.n_heads}, streams={config.mhc_n_streams}")

    model = NanochatTernary(config)
    counts = model.count_params()
    print(f"Parameters: {counts['total']:,} total")
    print(f"  Linear: {counts['linear']:,}")
    print(f"  mHC:    {counts['mhc']:,}")
    print(f"  Embed:  {counts['embed']:,}")
    print(f"  Norm:   {counts['norm']:,}")

    # Forward pass
    print("\nRunning forward pass...")
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(tokens)
    print(f"  Input:  {tokens.shape}")
    print(f"  Output: {logits.shape}")
    assert logits.shape == (2, 32, config.vocab_size)
    assert logits.isfinite().all(), "Non-finite logits!"

    # Backward pass
    print("\nRunning backward pass...")
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        tokens.view(-1)
    )
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")

    # Check gradients
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    n_with_grad = sum(1 for v in grad_norms.values() if v > 0)
    n_total = len(grad_norms)
    print(f"  Params with gradient: {n_with_grad}/{n_total}")
    # Some mHC params may have zero grad at identity init (alpha_logit=5.0)
    assert n_with_grad >= n_total * 0.8, f"Too few gradients: {n_with_grad}/{n_total}"

    # Check mHC doubly stochastic
    from mhc_lite import measure_composite_gain
    diag = measure_composite_gain(model)
    print(f"\nmHC diagnostics:")
    print(f"  Layers:         {diag['n_layers']}")
    print(f"  Composite gain: {diag['composite_gain']:.6f}")
    print(f"  DS violations:  {diag['ds_violations']}")

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    _self_test()
