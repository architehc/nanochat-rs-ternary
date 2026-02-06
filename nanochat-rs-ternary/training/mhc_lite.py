"""
mhc_lite.py — PyTorch Training Module for Manifold-Constrained Hyper-Connections (mHC-lite)

Exact Birkhoff-von-Neumann parameterization.
No Sinkhorn-Knopp iterations — doubly stochastic by construction.

References:
  mHC:      arXiv 2512.24880 (DeepSeek, Dec 2025)
  mHC-lite: arXiv 2601.05732 (Jan 2026)

Usage:
  # Wrap any transformer block
  mhc = MhcLiteLayer(dim=2048, n_streams=2)
  x_expanded = mhc.expand(x)            # [B, T, C] → [B, T, n*C]
  layer_in = mhc.prepare_input(x_exp)   # [B, T, n*C] → [B, T, C]
  layer_out = my_attn_or_ffn(layer_in)  # [B, T, C] → [B, T, C]
  x_expanded = mhc(x_expanded, layer_out)  # residual update
  y = mhc.collapse(x_expanded)          # [B, T, n*C] → [B, T, C]
"""

import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from itertools import permutations


# =============================================================================
# Core: Permutation Matrix Registry
# =============================================================================

def _build_perm_matrices(n: int) -> torch.Tensor:
    """Build all n! permutation matrices as a stacked tensor.
    
    Returns: [n!, n, n] tensor of permutation matrices.
    """
    perms = list(permutations(range(n)))
    matrices = torch.zeros(len(perms), n, n)
    for k, perm in enumerate(perms):
        for i, j in enumerate(perm):
            matrices[k, i, j] = 1.0
    return matrices

# Precompute and cache
_PERM_CACHE = {}

def get_perm_matrices(n: int, device: torch.device = None) -> torch.Tensor:
    """Get cached permutation matrices for S_n."""
    key = (n, device)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = _build_perm_matrices(n).to(device)
    return _PERM_CACHE[key]


# =============================================================================
# N=2: Minimal mHC-lite (1 learnable parameter for H_res)
# =============================================================================

class MhcLiteN2(nn.Module):
    """mHC-lite with expansion rate n=2.
    
    H_res = α·I + (1-α)·J where α = sigmoid(logit).
    Exact doubly stochastic by construction.
    
    Total learnable params per layer: 9
      - alpha_logit: 1
      - pre_logits + pre_bias: 4
      - post_logits + post_bias: 4
    """
    
    def __init__(self, dim: int, init_alpha: float = 5.0):
        super().__init__()
        self.dim = dim
        self.n_streams = 2
        
        # Residual mixing: single scalar
        self.alpha_logit = nn.Parameter(torch.tensor(init_alpha))
        
        # Pre-projection: how streams → layer input
        self.pre_logits = nn.Parameter(torch.zeros(2))
        self.pre_bias = nn.Parameter(torch.full((2,), 0.5))
        
        # Post-projection: how layer output → streams (2x scaled)
        self.post_logits = nn.Parameter(torch.zeros(2))
        self.post_bias = nn.Parameter(torch.full((2,), 0.5))
    
    def h_res(self) -> torch.Tensor:
        """Compute 2x2 doubly stochastic residual mixing matrix."""
        alpha = torch.sigmoid(self.alpha_logit)
        # [[α, 1-α], [1-α, α]]
        return torch.stack([
            torch.stack([alpha, 1.0 - alpha]),
            torch.stack([1.0 - alpha, alpha])
        ])
    
    def h_pre(self) -> torch.Tensor:
        """Compute non-negative pre-projection weights [2]."""
        return torch.sigmoid(self.pre_logits + self.pre_bias)
    
    def h_post(self) -> torch.Tensor:
        """Compute non-negative post-projection weights [2], 2x scaled."""
        return 2.0 * torch.sigmoid(self.post_logits + self.post_bias)
    
    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single stream to 2 streams.
        [B, T, C] → [B, T, 2*C]
        """
        return x.repeat(1, 1, 2)
    
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse 2 streams back to 1.
        [B, T, 2*C] → [B, T, C]
        """
        s0 = x[..., :self.dim]
        s1 = x[..., self.dim:]
        return 0.5 * (s0 + s1)
    
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Mix 2 streams → 1 stream for layer function.
        [B, T, 2*C] → [B, T, C]
        """
        h = self.h_pre()  # [2]
        s0 = x[..., :self.dim]
        s1 = x[..., self.dim:]
        return h[0] * s0 + h[1] * s1
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """Apply mHC residual update.
        
        x:            [B, T, 2*C]  (expanded state)
        layer_output: [B, T, C]    (output of layer function F)
        Returns:      [B, T, 2*C]  (updated expanded state)
        """
        H = self.h_res()       # [2, 2]
        hp = self.h_post()     # [2]
        
        s0 = x[..., :self.dim]
        s1 = x[..., self.dim:]
        
        # Residual mixing + layer contribution
        o0 = H[0, 0] * s0 + H[0, 1] * s1 + hp[0] * layer_output
        o1 = H[1, 0] * s0 + H[1, 1] * s1 + hp[1] * layer_output
        
        return torch.cat([o0, o1], dim=-1)


# =============================================================================
# N=4: Full BvN Parameterization (24 Permutation Matrices)
# =============================================================================

class MhcLiteN4(nn.Module):
    """mHC-lite with expansion rate n=4.
    
    H_res = Σ_{k=0}^{23} θ_k · P_k
    where θ = softmax(logits), P_k ∈ S_4 (24 permutation matrices).
    
    Exact doubly stochastic by construction.
    
    Total learnable params per layer: 40
      - res_logits: 24
      - pre_logits + pre_bias: 8
      - post_logits + post_bias: 8
    """
    
    def __init__(self, dim: int, init_identity_weight: float = 10.0):
        super().__init__()
        self.dim = dim
        self.n_streams = 4
        self.n_perms = 24  # 4!
        
        # Residual mixing: 24 logits → softmax → convex weights
        res_logits = torch.zeros(24)
        res_logits[0] = init_identity_weight  # Bias toward identity
        self.res_logits = nn.Parameter(res_logits)
        
        # Pre-projection
        self.pre_logits = nn.Parameter(torch.zeros(4))
        self.pre_bias = nn.Parameter(torch.full((4,), 0.5))
        
        # Post-projection (2x scaled)
        self.post_logits = nn.Parameter(torch.zeros(4))
        self.post_bias = nn.Parameter(torch.full((4,), 0.5))
        
        # Register permutation matrices as buffer (not learned)
        self.register_buffer('perm_matrices', _build_perm_matrices(4))
    
    def h_res(self) -> torch.Tensor:
        """Compute 4x4 doubly stochastic residual mixing matrix.
        
        H_res = Σ_k softmax(logits)_k · P_k
        """
        theta = F.softmax(self.res_logits, dim=0)  # [24]
        # Einsum: theta_k * P_k summed over k → [4, 4]
        return torch.einsum('k,kij->ij', theta, self.perm_matrices)
    
    def h_pre(self) -> torch.Tensor:
        """Non-negative pre-projection [4]."""
        return torch.sigmoid(self.pre_logits + self.pre_bias)
    
    def h_post(self) -> torch.Tensor:
        """Non-negative post-projection [4], 2x scaled."""
        return 2.0 * torch.sigmoid(self.post_logits + self.post_bias)
    
    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, T, 4*C]"""
        return x.repeat(1, 1, 4)
    
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, 4*C] → [B, T, C]"""
        chunks = x.unfold(-1, self.dim, self.dim)  # [B, T, 4, C]
        return chunks.mean(dim=-2)
    
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Mix 4 streams → 1 stream.
        [B, T, 4*C] → [B, T, C]
        """
        h = self.h_pre()  # [4]
        result = torch.zeros_like(x[..., :self.dim])
        for s in range(4):
            result = result + h[s] * x[..., s*self.dim:(s+1)*self.dim]
        return result
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """Apply mHC residual update.
        
        x:            [B, T, 4*C]
        layer_output: [B, T, C]
        Returns:      [B, T, 4*C]
        """
        H = self.h_res()       # [4, 4]
        hp = self.h_post()     # [4]
        
        # Split streams
        streams = [x[..., s*self.dim:(s+1)*self.dim] for s in range(4)]
        
        # Apply: o_s = Σ_t H[s,t] * streams[t] + hp[s] * layer_output
        outputs = []
        for s in range(4):
            o = hp[s] * layer_output
            for t in range(4):
                o = o + H[s, t] * streams[t]
            outputs.append(o)
        
        return torch.cat(outputs, dim=-1)


# =============================================================================
# Generic wrapper: auto-select N=2 or N=4
# =============================================================================

class MhcLiteLayer(nn.Module):
    """Drop-in mHC-lite layer. Wraps a transformer sub-layer (attn or FFN).
    
    Usage:
        mhc = MhcLiteLayer(dim=2048, n_streams=2)
        
        # At model init, expand the residual stream:
        x = mhc.expand(embed_output)  # [B, T, C] → [B, T, n*C]
        
        # Per-layer:
        layer_in = mhc.prepare_input(x)
        layer_out = attn_or_ffn(layer_in)
        x = mhc(x, layer_out)
        
        # At model output, collapse:
        y = mhc.collapse(x)  # [B, T, n*C] → [B, T, C]
    """
    
    def __init__(self, dim: int, n_streams: int = 2, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        
        if n_streams == 2:
            self.core = MhcLiteN2(dim, **kwargs)
        elif n_streams == 4:
            self.core = MhcLiteN4(dim, **kwargs)
        else:
            raise ValueError(f"n_streams must be 2 or 4, got {n_streams}")
    
    def expand(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.expand(x)
    
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.collapse(x)
    
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.prepare_input(x)
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        return self.core(x, layer_output)
    
    @property
    def h_res(self) -> torch.Tensor:
        return self.core.h_res()


# =============================================================================
# Ternary QAT Integration: BitLinear + mHC
# =============================================================================

class TernaryBitLinearWithMhc(nn.Module):
    """BitLinear layer with STE for ternary QAT, wrapped in mHC residual.
    
    This combines:
    1. FP32 shadow weights → ternary quantization via STE
    2. mHC-lite residual connection management
    
    For training only. At inference, export ternary weights + mHC FP32 params separately.
    """
    
    def __init__(self, in_features: int, out_features: int, n_streams: int = 2,
                 group_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Shadow weights (FP32, trained with Muon)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # mHC layer (trained with Lion)
        self.mhc = MhcLiteLayer(in_features, n_streams=n_streams)
    
    @staticmethod
    def ternary_quantize(w: torch.Tensor, group_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize FP32 weights to ternary {-1, 0, +1} with per-group scales.
        
        Uses absmean scaling (BitNet b1.58 style):
          scale = mean(|w|) per group
          w_ternary = round(clip(w / scale, -1, 1))
        
        Returns: (w_ternary, scales)
        """
        orig_shape = w.shape
        # Reshape into groups
        w_grouped = w.reshape(-1, group_size)
        
        # Per-group scale = mean absolute value
        scales = w_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Quantize: scale → clip → round
        w_scaled = w_grouped / scales
        w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        
        return w_ternary.reshape(orig_shape), scales.reshape(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with STE ternary quantization.
        
        x: [B, T, C]
        Returns: [B, T, out_features]
        """
        # Ternary quantize with STE
        w_ternary, scales = self.ternary_quantize(self.weight, self.group_size)
        # STE: forward uses ternary, backward uses FP32 shadow
        w_q = self.weight + (w_ternary * scales.unsqueeze(-1).expand_as(
            w_ternary.reshape(-1, self.group_size)
        ).reshape_as(self.weight) - self.weight).detach()
        
        return F.linear(x, w_q)


# =============================================================================
# Full Transformer Block with mHC Integration
# =============================================================================

class MhcTransformerBlock(nn.Module):
    """Complete transformer block demonstrating mHC integration.
    
    Architecture per block:
      x_expanded ──┬── [H_pre mix] → RMSNorm → Attention → [H_post distribute]
                   │                                          │
                   └── [H_res residual mix] ─────────────────→ + → x_expanded
                   
      x_expanded ──┬── [H_pre mix] → RMSNorm → FFN → [H_post distribute]
                   │                                    │
                   └── [H_res residual mix] ───────────→ + → x_expanded
    
    Each sub-layer (attn, FFN) gets its own mHC instance.
    """
    
    def __init__(self, dim: int, n_heads: int, n_streams: int = 2,
                 ffn_mult: float = 2.667, group_size: int = 64):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        
        # mHC for attention sub-layer
        self.mhc_attn = MhcLiteLayer(dim, n_streams=n_streams)
        
        # mHC for FFN sub-layer
        self.mhc_ffn = MhcLiteLayer(dim, n_streams=n_streams)
        
        # RMSNorm (applied to single-stream input)
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        
        # Attention (simplified — replace with your actual implementation)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # SwiGLU FFN
        ffn_dim = int(dim * ffn_mult)
        self.w_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, dim, bias=False)
    
    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified multi-head attention."""
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)
    
    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU FFN."""
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
    
    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        x_expanded: [B, T, n*C] (already in multi-stream form)
        Returns:    [B, T, n*C]
        """
        # === Attention sub-layer ===
        attn_in = self.mhc_attn.prepare_input(x_expanded)  # [B,T,n*C] → [B,T,C]
        attn_in = self.norm_attn(attn_in)
        attn_out = self._attention(attn_in)                 # [B,T,C]
        x_expanded = self.mhc_attn(x_expanded, attn_out)    # mHC residual update
        
        # === FFN sub-layer ===
        ffn_in = self.mhc_ffn.prepare_input(x_expanded)    # [B,T,n*C] → [B,T,C]
        ffn_in = self.norm_ffn(ffn_in)
        ffn_out = self._ffn(ffn_in)                        # [B,T,C]
        x_expanded = self.mhc_ffn(x_expanded, ffn_out)     # mHC residual update
        
        return x_expanded


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================================================================
# Optimizer Integration: Parameter Groups for Muon + Lion
# =============================================================================

def get_mhc_param_groups(model: nn.Module, 
                          muon_lr: float = 0.02,
                          lion_lr: float = 1e-4,
                          lion_wd: float = 0.1) -> List[dict]:
    """Split model parameters into optimizer groups.
    
    - Linear weight matrices → Muon (smooth manifold, benefits from orthogonal projections)
    - mHC params + norms + embeddings → Lion (small, high-impact, sign-based)
    
    Returns list of param group dicts for separate optimizers.
    """
    muon_params = []
    lion_params = []
    lion_no_wd_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(k in name for k in ['mhc', 'norm', 'bias']):
            # mHC mixing params, norm weights, biases → Lion
            if 'bias' in name or 'logit' in name:
                lion_no_wd_params.append(param)
            else:
                lion_params.append(param)
        elif 'embed' in name:
            # Embeddings → Lion (no weight decay)
            lion_no_wd_params.append(param)
        elif param.dim() >= 2:
            # Weight matrices → Muon
            muon_params.append(param)
        else:
            # Fallback: Lion
            lion_params.append(param)
    
    groups = []
    if muon_params:
        groups.append({
            'params': muon_params,
            'lr': muon_lr,
            'optimizer': 'muon',
            'momentum': 0.95,
        })
    if lion_params:
        groups.append({
            'params': lion_params,
            'lr': lion_lr,
            'optimizer': 'lion',
            'weight_decay': lion_wd,
        })
    if lion_no_wd_params:
        groups.append({
            'params': lion_no_wd_params,
            'lr': lion_lr,
            'optimizer': 'lion',
            'weight_decay': 0.0,
        })
    
    return groups


# =============================================================================
# Diagnostics: Composite Gain Measurement (the mHC paper's key metric)
# =============================================================================

@torch.no_grad()
def measure_composite_gain(model: nn.Module) -> dict:
    """Measure Amax Gain Magnitude across all mHC layers.
    
    This is the diagnostic from the mHC paper (Fig. 3, 7).
    For a healthy model:
      - Single-layer gains should be ≈ 1.0
      - Composite gain should be bounded (< 2.0 for mHC-lite)
    
    For unconstrained HC, composite gain can reach 3000+.
    """
    h_res_matrices = []
    
    for name, module in model.named_modules():
        if isinstance(module, (MhcLiteN2, MhcLiteN4)):
            h = module.h_res()
            h_res_matrices.append((name, h.cpu()))
    
    if not h_res_matrices:
        return {'error': 'No mHC layers found'}
    
    # Single-layer gains
    single_gains = []
    for name, h in h_res_matrices:
        max_row_sum = h.abs().sum(dim=-1).max().item()
        max_col_sum = h.abs().sum(dim=-2).max().item()
        gain = max(max_row_sum, max_col_sum)
        single_gains.append((name, gain))
    
    # Composite gain (product of all H_res)
    composite = torch.eye(h_res_matrices[0][1].shape[0])
    for _, h in h_res_matrices:
        composite = composite @ h
    
    max_row_sum = composite.abs().sum(dim=-1).max().item()
    max_col_sum = composite.abs().sum(dim=-2).max().item()
    composite_gain = max(max_row_sum, max_col_sum)
    
    # Verify all are doubly stochastic
    ds_errors = []
    for name, h in h_res_matrices:
        row_err = (h.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (h.sum(dim=-2) - 1.0).abs().max().item()
        neg_err = (-h).clamp(min=0).max().item()
        if max(row_err, col_err, neg_err) > 1e-5:
            ds_errors.append(f"{name}: row_err={row_err:.2e}, col_err={col_err:.2e}, neg={neg_err:.2e}")
    
    return {
        'n_layers': len(h_res_matrices),
        'single_layer_gains': single_gains,
        'single_gain_range': (
            min(g for _, g in single_gains),
            max(g for _, g in single_gains)
        ),
        'composite_gain': composite_gain,
        'ds_violations': ds_errors if ds_errors else 'none',
    }


# =============================================================================
# Export: Save mHC weights for Rust inference
# =============================================================================

def export_mhc_weights(model: nn.Module, path: str):
    """Export all mHC layer weights in binary format for Rust loading.
    
    Format per layer:
      N=2: [alpha_logit:f32][pre_logits:2*f32][pre_bias:2*f32]
           [post_logits:2*f32][post_bias:2*f32] = 36 bytes
      N=4: [res_logits:24*f32][pre_logits:4*f32][pre_bias:4*f32]
           [post_logits:4*f32][post_bias:4*f32] = 160 bytes
    
    Header: [magic:u32][version:u32][n_layers:u32][n_streams:u32]
    """
    layers = []
    n_streams = None
    
    for name, module in model.named_modules():
        if isinstance(module, MhcLiteN2):
            if n_streams is None:
                n_streams = 2
            layers.append(module)
        elif isinstance(module, MhcLiteN4):
            if n_streams is None:
                n_streams = 4
            layers.append(module)
    
    if not layers:
        raise ValueError("No mHC layers found in model")
    
    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x6D484321))  # magic: "mHC!"
        f.write(struct.pack('<I', 1))            # version
        f.write(struct.pack('<I', len(layers)))   # n_layers
        f.write(struct.pack('<I', n_streams))     # n_streams
        
        for layer in layers:
            if isinstance(layer, MhcLiteN2):
                f.write(struct.pack('<f', layer.alpha_logit.item()))
                for v in layer.pre_logits.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.pre_bias.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.post_logits.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.post_bias.tolist():
                    f.write(struct.pack('<f', v))
            elif isinstance(layer, MhcLiteN4):
                for v in layer.res_logits.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.pre_logits.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.pre_bias.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.post_logits.tolist():
                    f.write(struct.pack('<f', v))
                for v in layer.post_bias.tolist():
                    f.write(struct.pack('<f', v))
    
    print(f"Exported {len(layers)} mHC layers ({n_streams} streams) to {path}")
    print(f"  Total size: {16 + len(layers) * (36 if n_streams == 2 else 160)} bytes")


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    """Run all verification checks."""
    print("=" * 60)
    print("mHC-lite Self-Test")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # --- N=2 tests ---
    print("\n[N=2] Testing doubly stochastic property...")
    for logit in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        mhc = MhcLiteN2(dim=64, init_alpha=logit)
        h = mhc.h_res()
        row_err = (h.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (h.sum(dim=-2) - 1.0).abs().max().item()
        neg = (-h).clamp(min=0).max().item()
        assert row_err < 1e-6, f"Row sum error: {row_err}"
        assert col_err < 1e-6, f"Col sum error: {col_err}"
        assert neg < 1e-6, f"Negative entry: {neg}"
        alpha = torch.sigmoid(torch.tensor(logit)).item()
        print(f"  logit={logit:6.1f} → α={alpha:.4f} | "
              f"H_res=[[{h[0,0]:.4f}, {h[0,1]:.4f}], "
              f"[{h[1,0]:.4f}, {h[1,1]:.4f}]] ✓")
    
    # --- N=4 tests ---
    print("\n[N=4] Testing doubly stochastic property...")
    for seed in range(5):
        torch.manual_seed(seed)
        mhc = MhcLiteN4(dim=64, init_identity_weight=0.0)
        # Randomize
        mhc.res_logits.data = torch.randn(24) * 3.0
        h = mhc.h_res()
        row_err = (h.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (h.sum(dim=-2) - 1.0).abs().max().item()
        neg = (-h).clamp(min=0).max().item()
        assert row_err < 1e-5, f"Row sum error: {row_err}"
        assert col_err < 1e-5, f"Col sum error: {col_err}"
        assert neg < 1e-6, f"Negative entry: {neg}"
        print(f"  seed={seed} | row_err={row_err:.2e} col_err={col_err:.2e} neg={neg:.2e} ✓")
    
    # --- Composite gain test ---
    print("\n[N=4] Testing composite gain (64 layers)...")
    composite = torch.eye(4)
    for i in range(64):
        torch.manual_seed(i * 7 + 3)
        mhc = MhcLiteN4(dim=64, init_identity_weight=0.0)
        mhc.res_logits.data = torch.randn(24) * 3.0
        h = mhc.h_res()
        composite = composite @ h
    
    max_row = composite.abs().sum(dim=-1).max().item()
    max_col = composite.abs().sum(dim=-2).max().item()
    gain = max(max_row, max_col)
    print(f"  Composite Amax Gain after 64 layers: {gain:.6f}")
    # BvN-exact DS matrices: product is also DS, gain should be ≤ 1.0
    assert gain <= 1.0 + 1e-4, f"Gain {gain} exceeds bound!"
    print(f"  ≤ 1.0 ✓  (HC would be ~3000+ here)")
    
    # --- Forward pass test ---
    print("\n[N=2] Testing forward pass shape...")
    mhc = MhcLiteLayer(dim=128, n_streams=2)
    x = torch.randn(2, 16, 128)
    x_exp = mhc.expand(x)
    assert x_exp.shape == (2, 16, 256), f"Bad expand shape: {x_exp.shape}"
    
    layer_in = mhc.prepare_input(x_exp)
    assert layer_in.shape == (2, 16, 128), f"Bad prepare shape: {layer_in.shape}"
    
    layer_out = layer_in * 0.1  # dummy layer
    x_exp = mhc(x_exp, layer_out)
    assert x_exp.shape == (2, 16, 256), f"Bad forward shape: {x_exp.shape}"
    
    y = mhc.collapse(x_exp)
    assert y.shape == (2, 16, 128), f"Bad collapse shape: {y.shape}"
    print(f"  expand: {x.shape} → {(2,16,256)} ✓")
    print(f"  prepare_input: {(2,16,256)} → {layer_in.shape} ✓")
    print(f"  forward: {(2,16,256)} → {x_exp.shape} ✓")
    print(f"  collapse: {(2,16,256)} → {y.shape} ✓")
    
    # --- Gradient flow test ---
    print("\n[N=2] Testing gradient flow through STE + mHC...")
    mhc2 = MhcLiteN2(dim=32)
    x = torch.randn(1, 4, 32, requires_grad=True)
    x_exp = mhc2.expand(x)
    layer_in = mhc2.prepare_input(x_exp)
    layer_out = layer_in * 0.5
    x_exp = mhc2(x_exp, layer_out)
    y = mhc2.collapse(x_exp)
    loss = y.sum()
    loss.backward()
    
    assert mhc2.alpha_logit.grad is not None, "No gradient for alpha_logit"
    assert mhc2.pre_logits.grad is not None, "No gradient for pre_logits"
    assert mhc2.post_logits.grad is not None, "No gradient for post_logits"
    print(f"  alpha_logit grad: {mhc2.alpha_logit.grad.item():.6f} ✓")
    print(f"  pre_logits grad:  {mhc2.pre_logits.grad.tolist()} ✓")
    print(f"  post_logits grad: {mhc2.post_logits.grad.tolist()} ✓")
    
    # --- Full block test ---
    print("\n[Block] Testing MhcTransformerBlock...")
    block = MhcTransformerBlock(dim=64, n_heads=4, n_streams=2)
    x = torch.randn(1, 8, 64)
    x_exp = block.mhc_attn.expand(x)
    x_exp = block(x_exp)
    y = block.mhc_ffn.collapse(x_exp)
    loss = y.sum()
    loss.backward()
    
    diag = measure_composite_gain(block)
    print(f"  Output shape: {y.shape} ✓")
    print(f"  Composite gain: {diag['composite_gain']:.6f}")
    print(f"  DS violations: {diag['ds_violations']}")
    
    # --- Export test ---
    print("\n[Export] Testing binary export...")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        tmppath = f.name
    
    # Build a small model with mHC layers
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MhcLiteN2(64) for _ in range(4)])
    
    model = TinyModel()
    export_mhc_weights(model, tmppath)
    
    # Verify file
    with open(tmppath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_layers = struct.unpack('<I', f.read(4))[0]
        n_streams = struct.unpack('<I', f.read(4))[0]
    
    assert magic == 0x6D484321, f"Bad magic: {hex(magic)}"
    assert n_layers == 4, f"Bad n_layers: {n_layers}"
    assert n_streams == 2, f"Bad n_streams: {n_streams}"
    expected_size = 16 + 4 * 36
    actual_size = os.path.getsize(tmppath)
    assert actual_size == expected_size, f"Bad size: {actual_size} vs {expected_size}"
    os.unlink(tmppath)
    print(f"  magic=0x{magic:08X} version={version} layers={n_layers} streams={n_streams} ✓")
    print(f"  file size: {actual_size} bytes (expected {expected_size}) ✓")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == '__main__':
    _self_test()
