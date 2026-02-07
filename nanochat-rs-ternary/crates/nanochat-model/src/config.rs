//! Model configuration for nanochat-rs ternary models.

/// Configuration for a nanochat ternary model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Hidden dimension
    pub dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (for GQA; same as n_heads for MHA)
    pub n_kv_heads: usize,
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
    /// RoPE theta
    pub rope_theta: f32,
    /// Number of experts (None = dense FFN)
    pub n_experts: Option<usize>,
    /// Number of active experts per token
    pub n_active_experts: Option<usize>,
    /// Fraction of layers using DeltaNet attention (None = all standard MHA)
    pub deltanet_ratio: Option<f32>,
    /// Whether embedding and LM head weights are tied
    pub weight_tied: bool,
}

impl ModelConfig {
    /// ~20M param debug/test config
    pub fn d20() -> Self {
        Self {
            dim: 256, n_layers: 6, n_heads: 4, n_kv_heads: 4,
            ffn_mult: 2.667, vocab_size: 32000, max_seq_len: 512,
            group_size: 128, mhc_n_streams: 2, rope_theta: 10000.0,
            n_experts: None, n_active_experts: None, deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// ~125M param config (trained on TinyStories)
    pub fn nano_125m() -> Self {
        Self {
            dim: 768, n_layers: 12, n_heads: 12, n_kv_heads: 12,
            ffn_mult: 2.667, vocab_size: 50257, max_seq_len: 2048,
            group_size: 128, mhc_n_streams: 2, rope_theta: 10000.0,
            n_experts: None, n_active_experts: None, deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// ~560M param config
    pub fn nano_560m() -> Self {
        Self {
            dim: 1024, n_layers: 24, n_heads: 16, n_kv_heads: 16,
            ffn_mult: 2.667, vocab_size: 32000, max_seq_len: 2048,
            group_size: 128, mhc_n_streams: 2, rope_theta: 10000.0,
            n_experts: None, n_active_experts: None, deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// ~7B param config
    pub fn nano_7b() -> Self {
        Self {
            dim: 4096, n_layers: 32, n_heads: 32, n_kv_heads: 8,
            ffn_mult: 2.667, vocab_size: 128256, max_seq_len: 8192,
            group_size: 128, mhc_n_streams: 4, rope_theta: 500000.0,
            n_experts: None, n_active_experts: None, deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// ~25B param MoE config
    pub fn moe_25b() -> Self {
        Self {
            dim: 4096, n_layers: 32, n_heads: 32, n_kv_heads: 8,
            ffn_mult: 2.667, vocab_size: 128256, max_seq_len: 8192,
            group_size: 128, mhc_n_streams: 4, rope_theta: 500000.0,
            n_experts: Some(8), n_active_experts: Some(2), deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// ~80B param MoE config
    pub fn moe_80b() -> Self {
        Self {
            dim: 8192, n_layers: 64, n_heads: 64, n_kv_heads: 8,
            ffn_mult: 2.667, vocab_size: 128256, max_seq_len: 8192,
            group_size: 128, mhc_n_streams: 4, rope_theta: 500000.0,
            n_experts: Some(16), n_active_experts: Some(2), deltanet_ratio: None,
            weight_tied: false,
        }
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// FFN intermediate dimension, rounded to group_size
    pub fn ffn_dim(&self) -> usize {
        let raw = (self.dim as f32 * self.ffn_mult) as usize;
        // Round up to multiple of group_size for clean quantization groups
        (raw + self.group_size - 1) / self.group_size * self.group_size
    }

    /// Number of KV head repetitions for GQA
    pub fn n_rep(&self) -> usize {
        self.n_heads / self.n_kv_heads
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
        assert!(c.ffn_dim() % c.group_size == 0);
    }

    #[test]
    fn test_7b_config() {
        let c = ModelConfig::nano_7b();
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.n_rep(), 4); // GQA: 32 heads, 8 KV heads
        assert!(c.ffn_dim() % c.group_size == 0);
    }

    #[test]
    fn test_ffn_dim_aligned() {
        for config in [ModelConfig::d20(), ModelConfig::nano_560m(), ModelConfig::nano_7b()] {
            assert!(
                config.ffn_dim() % config.group_size == 0,
                "ffn_dim {} not aligned to group_size {}",
                config.ffn_dim(),
                config.group_size
            );
        }
    }
}
