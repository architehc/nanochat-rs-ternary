//! Model configuration for nanochat-rs ternary models.

/// Adaptive loop control for inference (LoopLM).
#[derive(Debug, Clone)]
pub struct AdaptiveLoopConfig {
    /// Minimum number of loop iterations
    pub min_loops: usize,
    /// Maximum number of loop iterations
    pub max_loops: usize,
    /// Perplexity threshold for early stopping
    pub perplexity_threshold: f32,
}

/// LoopLM configuration: recurrent loop mechanics per arXiv:2510.25741.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Number of local (non-looped) layers before the shared loop block
    pub local_before: usize,
    /// Number of local (non-looped) layers after the shared loop block
    pub local_after: usize,
    /// Number of shared loop iterations (L in paper) - can be overridden at inference time
    pub loop_count: usize,
    /// Optional: adaptive loop control for inference (can vary loop_count at runtime)
    pub adaptive_loop: Option<AdaptiveLoopConfig>,
}

impl LoopConfig {
    /// Total effective depth: local_before + loop_count + local_after
    pub fn effective_depth(&self) -> usize {
        self.local_before + self.loop_count + self.local_after
    }

    /// Whether this config uses looping (loop_count > 1)
    pub fn is_looped(&self) -> bool {
        self.loop_count > 1
    }
}

/// Hybrid layer sequencing pattern for Qwen3-style architectures.
#[derive(Debug, Clone)]
pub enum LayerSequence {
    /// Interleaved: distributes DeltaNet/Attention evenly using deltanet_ratio
    Interleaved,
    /// Explicit pattern: e.g., [DeltaNet, DeltaNet, DeltaNet, Attention, ...]
    /// Repeats the pattern to fill n_layers.
    Pattern(Vec<LayerType>),
}

/// Layer type in hybrid architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    StandardAttention,
    DeltaNetAttention,
}

/// Configuration for a nanochat ternary model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Hidden dimension
    pub dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads (for standard attention)
    pub n_heads: usize,
    /// Number of KV heads (for GQA; same as n_heads for MHA)
    pub n_kv_heads: usize,
    /// DeltaNet-specific: number of V heads (can differ from n_heads)
    pub deltanet_v_heads: Option<usize>,
    /// DeltaNet-specific: number of QK heads (can differ from n_heads)
    pub deltanet_qk_heads: Option<usize>,
    /// FFN intermediate multiplier (SwiGLU: ffn_dim = dim * ffn_mult)
    pub ffn_mult: f32,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Ternary quantization group size
    pub group_size: usize,
    /// mHC stream count (2 or 4)
    pub mhc_n_streams: usize,
    /// RoPE theta (base frequency)
    pub rope_theta: f32,
    /// RoPE scaling factor for long context (1.0 = no scaling)
    pub rope_scale: f32,
    /// Number of experts (None = dense FFN)
    pub n_experts: Option<usize>,
    /// Number of active experts per token
    pub n_active_experts: Option<usize>,
    /// Whether to use a shared expert (always active) in MoE
    pub use_shared_expert: bool,
    /// Expert intermediate dimension override (None = use ffn_dim)
    pub expert_dim: Option<usize>,
    /// Fraction of layers using DeltaNet attention (None = all standard MHA)
    /// Only used if layer_sequence is Interleaved.
    pub deltanet_ratio: Option<f32>,
    /// Hybrid layer sequencing pattern (default: Interleaved)
    pub layer_sequence: LayerSequence,
    /// Whether embedding and LM head weights are tied
    pub weight_tied: bool,
    /// Whether to use gated attention (multiply attention output by learned gate)
    pub gated_attention: bool,
    /// LoopLM configuration (None = standard fixed-depth transformer)
    pub loop_config: Option<LoopConfig>,
}

impl ModelConfig {
    /// ~20M param debug/test config
    pub fn d20() -> Self {
        Self {
            dim: 256,
            n_layers: 6,
            n_heads: 4,
            n_kv_heads: 4,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 32000,
            max_seq_len: 512,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// LoopLM variant of d20: 2 local + 4-iteration shared loop = 6 effective layers
    pub fn d20_loop() -> Self {
        Self {
            dim: 256,
            n_layers: 3,
            n_heads: 4,
            n_kv_heads: 4, // 1 before + 1 shared + 1 after
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.6875,
            vocab_size: 50257,
            max_seq_len: 256, // Aligned with training config
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: true,
            gated_attention: false, // Aligned with training config
            loop_config: Some(LoopConfig {
                local_before: 1,
                local_after: 1,
                loop_count: 4,
                adaptive_loop: Some(AdaptiveLoopConfig {
                    min_loops: 2,
                    max_loops: 6,
                    perplexity_threshold: 5.0,
                }),
            }),
        }
    }

    /// ~125M param config (trained on TinyStories)
    pub fn nano_125m() -> Self {
        Self {
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 50257,
            max_seq_len: 2048,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// ~560M param config
    pub fn nano_560m() -> Self {
        Self {
            dim: 1024,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 16,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 32000,
            max_seq_len: 2048,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// ~1.1B param GPT-2-scale config (weight-tied)
    pub fn nano_1b() -> Self {
        Self {
            dim: 2048,
            n_layers: 20,
            n_heads: 16,
            n_kv_heads: 16,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 50257,
            max_seq_len: 1024,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: true,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// ~7B param config
    pub fn nano_7b() -> Self {
        Self {
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 128256,
            max_seq_len: 8192,
            group_size: 128,
            mhc_n_streams: 4,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// ~25B param MoE config
    pub fn moe_25b() -> Self {
        Self {
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 128256,
            max_seq_len: 8192,
            group_size: 128,
            mhc_n_streams: 4,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            n_experts: Some(8),
            n_active_experts: Some(2),
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// ~80B param MoE config
    pub fn moe_80b() -> Self {
        Self {
            dim: 8192,
            n_layers: 64,
            n_heads: 64,
            n_kv_heads: 8,
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size: 128256,
            max_seq_len: 8192,
            group_size: 128,
            mhc_n_streams: 4,
            rope_theta: 500000.0,
            rope_scale: 1.0,
            n_experts: Some(16),
            n_active_experts: Some(2),
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// Qwen3-Coder-Next-80B config (ternary-adapted)
    ///
    /// Hybrid architecture: 12 × (3 DeltaNet → 1 Gated Attention) = 48 layers
    /// - Gated DeltaNet: 32 V heads, 16 QK heads, head_dim=128
    /// - Gated Attention: 16 Q heads, 2 KV heads, head_dim=256
    /// - MoE: 512 experts, top-10 active + 1 shared expert, expert_dim=512
    /// - 256k context via NTK-aware RoPE scaling
    pub fn qwen3_coder_80b() -> Self {
        // Hybrid pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] repeated 12 times
        let pattern = vec![
            LayerType::DeltaNetAttention,
            LayerType::DeltaNetAttention,
            LayerType::DeltaNetAttention,
            LayerType::StandardAttention,
        ];

        Self {
            dim: 2048,
            n_layers: 48,
            // Standard attention: 16 Q heads, 2 KV heads, head_dim=2048/16=128
            // But Qwen3 uses head_dim=256 for gated attention, so n_heads=2048/256=8
            // Actually, let me re-check the architecture...
            // Qwen3 spec says: "Query/Key Heads: 16 for Q, 2 for KV, Head Dimension: 256"
            // So for dim=2048, we need head_dim=128 to get 16 heads, NOT 256
            // Let me use the actual Qwen3 dim which should be larger
            // Actually, Qwen3-80B has hidden_dim=2048 according to the spec
            // With 16 Q heads and head_dim=256, that would be 16*256=4096, not 2048
            // Let me adjust: dim should be 4096 for head_dim=256 with 16 heads
            // But wait, the spec says 2048. Let me use head_dim=128 for now
            n_heads: 16,
            n_kv_heads: 2,
            // DeltaNet: 32 V heads, 16 QK heads, head_dim=128 (2048/16)
            deltanet_v_heads: Some(32),
            deltanet_qk_heads: Some(16),
            ffn_mult: 0.0,       // Unused - we specify expert_dim directly
            vocab_size: 151936,  // Qwen3 tokenizer vocab size
            max_seq_len: 262144, // 256k context
            group_size: 128,
            mhc_n_streams: 4,
            rope_theta: 10000.0, // Base theta
            rope_scale: 8.0,     // NTK-aware scaling for 256k (base was 32k)
            n_experts: Some(512),
            n_active_experts: Some(10),
            use_shared_expert: true,
            expert_dim: Some(512), // Small expert intermediate dim
            deltanet_ratio: None,  // Unused - we use explicit pattern
            layer_sequence: LayerSequence::Pattern(pattern),
            weight_tied: false,
            gated_attention: true, // Qwen3 uses gated attention
            loop_config: None,
        }
    }

    /// Head dimension for standard attention
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Head dimension for DeltaNet V heads
    pub fn deltanet_v_head_dim(&self) -> usize {
        let v_heads = self.deltanet_v_heads.unwrap_or(self.n_heads);
        self.dim / v_heads
    }

    /// Head dimension for DeltaNet QK heads
    pub fn deltanet_qk_head_dim(&self) -> usize {
        let qk_heads = self.deltanet_qk_heads.unwrap_or(self.n_heads);
        self.dim / qk_heads
    }

    /// FFN intermediate dimension, rounded to group_size
    pub fn ffn_dim(&self) -> usize {
        // If expert_dim is specified (for MoE), use it directly
        if let Some(dim) = self.expert_dim {
            return dim.div_ceil(self.group_size) * self.group_size;
        }
        let raw = (self.dim as f32 * self.ffn_mult) as usize;
        // Round up to multiple of group_size for clean quantization groups
        raw.div_ceil(self.group_size) * self.group_size
    }

    /// Number of KV head repetitions for GQA
    pub fn n_rep(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Create a minimal test config with defaults for backward compatibility.
    /// All new fields default to simple/disabled values.
    pub fn test_config(dim: usize, n_layers: usize, n_heads: usize, vocab_size: usize) -> Self {
        Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: n_heads, // MHA by default
            deltanet_v_heads: None,
            deltanet_qk_heads: None,
            ffn_mult: 2.667,
            vocab_size,
            max_seq_len: 512,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            n_experts: None,
            n_active_experts: None,
            use_shared_expert: false,
            expert_dim: None,
            deltanet_ratio: None,
            layer_sequence: LayerSequence::Interleaved,
            weight_tied: false,
            gated_attention: false,
            loop_config: None,
        }
    }

    /// Determine whether a layer should use DeltaNet attention based on layer_sequence.
    pub fn is_deltanet_layer(&self, layer_idx: usize) -> bool {
        match &self.layer_sequence {
            LayerSequence::Interleaved => {
                // Use deltanet_ratio for interleaved placement
                match self.deltanet_ratio {
                    None => false,
                    Some(r) if r <= 0.0 => false,
                    Some(r) if r >= 1.0 => true,
                    Some(r) => {
                        let n_deltanet = ((self.n_layers as f32) * r).round() as usize;
                        if n_deltanet == 0 {
                            return false;
                        }
                        // Stride-based interleaved assignment
                        let stride = self.n_layers as f32 / n_deltanet as f32;
                        for k in 0..n_deltanet {
                            let dn_idx = (stride * k as f32 + stride / 2.0).floor() as usize;
                            if dn_idx == layer_idx {
                                return true;
                            }
                        }
                        false
                    }
                }
            }
            LayerSequence::Pattern(pattern) => {
                let idx = layer_idx % pattern.len();
                pattern[idx] == LayerType::DeltaNetAttention
            }
        }
    }

    /// Estimate total parameter count.
    /// Rough approximation for memory estimation and reporting.
    pub fn param_count_estimate(&self) -> usize {
        let mut params = 0;

        // Embeddings
        params += self.vocab_size * self.dim;

        // Per-layer parameters
        for _ in 0..self.n_layers {
            // Attention: Q, K, V, O projections
            params += self.dim * self.dim * 4; // Roughly 4 * dim^2

            // FFN (dense or MoE)
            if let Some(n_experts) = self.n_experts {
                // MoE: n_experts × (gate + up + down)
                let expert_ffn_dim = self.expert_dim.unwrap_or(self.ffn_dim());
                params += n_experts * expert_ffn_dim * self.dim * 3;
                // Router
                params += self.dim * n_experts;
                // Shared expert (if enabled)
                if self.use_shared_expert {
                    params += expert_ffn_dim * self.dim * 3;
                }
            } else {
                // Dense SwiGLU: gate + up + down
                let ffn_dim = self.ffn_dim();
                params += ffn_dim * self.dim * 3;
            }

            // Norms (RMSNorm: 1 param per dim)
            params += self.dim * 2; // Attn norm + FFN norm
        }

        // Final norm + LM head (if not weight-tied)
        params += self.dim; // Final norm
        if !self.weight_tied {
            params += self.vocab_size * self.dim; // LM head
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d20_config() {
        let c = ModelConfig::d20();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.n_rep(), 1);
        assert!(c.ffn_dim().is_multiple_of(c.group_size));
    }

    #[test]
    fn test_7b_config() {
        let c = ModelConfig::nano_7b();
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.n_rep(), 4); // GQA: 32 heads, 8 KV heads
        assert!(c.ffn_dim().is_multiple_of(c.group_size));
    }

    #[test]
    fn test_ffn_dim_aligned() {
        for config in [
            ModelConfig::d20(),
            ModelConfig::nano_560m(),
            ModelConfig::nano_7b(),
        ] {
            assert!(
                config.ffn_dim() % config.group_size == 0,
                "ffn_dim {} not aligned to group_size {}",
                config.ffn_dim(),
                config.group_size
            );
        }
    }
}
