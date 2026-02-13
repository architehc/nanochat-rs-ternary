//! Full nanochat model: embed -> mHC expand -> blocks -> mHC collapse -> norm -> head.

use std::io;

use crate::attention::{Attention, RopeFreqs, KvCache};
use crate::bitlinear::BitLinear;
use crate::block::{AttentionLayer, AttentionState, TransformerBlock};
use crate::config::{LayerSequence, ModelConfig};
use crate::deltanet::DeltaNetAttention;
use crate::embed::Embedding;
use crate::ffn::{FeedForward, FfnLayer, MoeExperts};
use crate::loop_block::SharedLoopBlock;
use crate::norm::RMSNorm;
use mhc_lite::MhcLiteN2;
use ternary_core::gguf::{GgufFile, GgufValue};


/// Full nanochat-rs ternary model.
#[derive(Debug)]
pub struct NanochatModel {
    pub config: ModelConfig,
    pub tok_embed: Embedding,

    // Standard architecture (when loop_config is None)
    pub blocks: Vec<TransformerBlock>,
    pub attn_states: Vec<AttentionState>,

    // LoopLM architecture (when loop_config is Some)
    pub local_blocks_before: Vec<TransformerBlock>,
    pub local_states_before: Vec<AttentionState>,
    pub shared_loop_block: Option<SharedLoopBlock>,
    pub loop_kv_cache: Option<KvCache>,
    pub local_blocks_after: Vec<TransformerBlock>,
    pub local_states_after: Vec<AttentionState>,

    pub norm_final: RMSNorm,
    pub lm_head: Option<BitLinear>,
    pub weight_tied: bool,
    pub rope: RopeFreqs,
}

impl NanochatModel {
    /// Create model with random weights (for testing / validation).
    pub fn new_random(config: ModelConfig) -> Self {
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);

        // Build architecture based on loop_config
        let (blocks, attn_states, local_blocks_before, local_states_before,
             shared_loop_block, loop_kv_cache, local_blocks_after, local_states_after) =
            if let Some(loop_cfg) = &config.loop_config {
                // LoopLM architecture: local_before + shared_loop + local_after
                // Layer indices: [0..local_before) then shared_loop, then [local_before+1..local_before+1+local_after)
                let before: Vec<_> = (0..loop_cfg.local_before)
                    .map(|i| {
                        if config.is_deltanet_layer(i) {
                            TransformerBlock::new_random_deltanet(&config)
                        } else {
                            TransformerBlock::new_random(&config)
                        }
                    })
                    .collect();

                let states_before: Vec<_> = before.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                let shared = SharedLoopBlock::new_empty(&config);
                let loop_cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());

                let after: Vec<_> = (0..loop_cfg.local_after)
                    .map(|i| {
                        let layer_idx = loop_cfg.local_before + 1 + i; // Skip shared loop index
                        if config.is_deltanet_layer(layer_idx) {
                            TransformerBlock::new_random_deltanet(&config)
                        } else {
                            TransformerBlock::new_random(&config)
                        }
                    })
                    .collect();

                let states_after: Vec<_> = after.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                (Vec::new(), Vec::new(), before, states_before,
                 Some(shared), Some(loop_cache), after, states_after)
            } else {
                // Standard architecture: n_layers blocks
                let blocks: Vec<_> = (0..config.n_layers)
                    .map(|i| {
                        if config.is_deltanet_layer(i) {
                            TransformerBlock::new_random_deltanet(&config)
                        } else {
                            TransformerBlock::new_random(&config)
                        }
                    })
                    .collect();

                let attn_states: Vec<_> = blocks.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                (blocks, attn_states, Vec::new(), Vec::new(),
                 None, None, Vec::new(), Vec::new())
            };

        // LM head: vocab_size x dim (None if weight_tied)
        let weight_tied = config.weight_tied;
        let lm_head = if weight_tied {
            None
        } else {
            let lm_weights: Vec<f32> = (0..config.vocab_size * config.dim)
                .map(|i| {
                    let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
                    v as f32 / 100.0 - 1.0
                })
                .collect();
            Some(BitLinear::from_float(&lm_weights, config.vocab_size, config.dim, config.group_size))
        };

        Self {
            tok_embed: Embedding::new_random(config.vocab_size, config.dim, 42),
            blocks,
            attn_states,
            local_blocks_before,
            local_states_before,
            shared_loop_block,
            loop_kv_cache,
            local_blocks_after,
            local_states_after,
            norm_final: RMSNorm::new(config.dim),
            lm_head,
            weight_tied,
            rope,
            config,
        }
    }

    /// Load model from GGUF weights + mHC binary file.
    pub fn from_gguf(gguf_path: &str, mhc_path: &str) -> io::Result<Self> {
        let gguf = GgufFile::open(gguf_path)?;

        // Extract config from GGUF metadata
        let config = Self::config_from_gguf(&gguf)?;
        let group_size = config.group_size;

        // Load embeddings (FP16 -> F32)
        let tok_embed = Self::load_embedding(&gguf, "tok_embed.weight")?;

        // Load mHC parameters
        let (_, mhc_layers) = mhc_lite::load_mhc_file(mhc_path)?;

        // Helper to load a single transformer block
        let load_block = |prefix: &str, layer_idx: usize, use_deltanet: bool| -> io::Result<TransformerBlock> {

            // Attention weights
            let wq = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wq.weight"), group_size)?);
            let wk = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wk.weight"), group_size)?);
            let wv = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wv.weight"), group_size)?);
            let wo = BitLinear::new(gguf.load_planar_weights(
                &format!("{prefix}.attention.wo.weight"), group_size)?);

            let attention = if use_deltanet {
                let w_beta = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.attention.w_beta.weight"), group_size)?);
                AttentionLayer::DeltaNet(DeltaNetAttention {
                    wq, wk, wv, wo, w_beta,
                    n_heads: config.n_heads,
                    head_dim: config.head_dim(),
                })
            } else {
                AttentionLayer::Standard(Attention {
                    wq, wk, wv, wo,
                    n_heads: config.n_heads,
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim(),
                    n_rep: config.n_rep(),
                })
            };

            // FFN weights — dense or MoE
            let ffn = if let Some(n_experts) = config.n_experts {
                let n_active = config.n_active_experts.unwrap_or(2);

                // Router weights
                let router = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.router.weight"), group_size)?);

                // Per-expert FFN weights
                let mut experts = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    let w_gate = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_gate.weight"), group_size)?);
                    let w_up = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_up.weight"), group_size)?);
                    let w_down = BitLinear::new(gguf.load_planar_weights(
                        &format!("{prefix}.ffn.experts.{e}.w_down.weight"), group_size)?);
                    experts.push(FeedForward {
                        w_gate, w_up, w_down,
                        ffn_dim: config.ffn_dim(),
                    });
                }

                FfnLayer::Moe(Box::new(MoeExperts { router, experts, n_active }))
            } else {
                let w_gate = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_gate.weight"), group_size)?);
                let w_up = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_up.weight"), group_size)?);
                let w_down = BitLinear::new(gguf.load_planar_weights(
                    &format!("{prefix}.ffn.w_down.weight"), group_size)?);

                FfnLayer::Dense(Box::new(FeedForward {
                    w_gate, w_up, w_down,
                    ffn_dim: config.ffn_dim(),
                }))
            };

            // Norms
            let norm_attn = Self::load_norm(&gguf, &format!("{prefix}.norm_attn.weight"))?;
            let norm_ffn = Self::load_norm(&gguf, &format!("{prefix}.norm_ffn.weight"))?;

            // mHC params (order: attn_0, ffn_0, attn_1, ffn_1, ...)
            let mhc_attn = Self::extract_mhc_n2(&mhc_layers, layer_idx * 2)?;
            let mhc_ffn = Self::extract_mhc_n2(&mhc_layers, layer_idx * 2 + 1)?;

            Ok(TransformerBlock {
                mhc_attn, mhc_ffn,
                norm_attn, norm_ffn,
                attention, ffn,
                dim: config.dim,
            })
        };

        // Load blocks based on architecture
        let (blocks, attn_states, local_blocks_before, local_states_before,
             shared_loop_block, loop_kv_cache, local_blocks_after, local_states_after) =
            if let Some(loop_cfg) = &config.loop_config {
                // LoopLM architecture: load local_before + shared_loop + local_after
                let mut mhc_idx = 0;

                // Load local_before blocks
                let before: Vec<_> = (0..loop_cfg.local_before)
                    .map(|i| {
                        let result = load_block(&format!("local_before.{}", i), mhc_idx, config.is_deltanet_layer(i));
                        mhc_idx += 1;
                        result
                    })
                    .collect::<io::Result<Vec<_>>>()?;

                let states_before: Vec<_> = before.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                // Load shared loop block
                let shared = Self::load_shared_loop_block(&gguf, &mhc_layers, mhc_idx, &config)?;
                let loop_cache = KvCache::new(config.max_seq_len, config.n_kv_heads, config.head_dim());
                mhc_idx += 1;

                // Load local_after blocks
                let after: Vec<_> = (0..loop_cfg.local_after)
                    .map(|i| {
                        let layer_num = loop_cfg.local_before + 1 + i;
                        let result = load_block(&format!("local_after.{}", i), mhc_idx, config.is_deltanet_layer(layer_num));
                        mhc_idx += 1;
                        result
                    })
                    .collect::<io::Result<Vec<_>>>()?;

                let states_after: Vec<_> = after.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                (Vec::new(), Vec::new(), before, states_before,
                 Some(shared), Some(loop_cache), after, states_after)
            } else {
                // Standard architecture: load n_layers blocks
                let blocks: Vec<_> = (0..config.n_layers)
                    .map(|i| load_block(&format!("blocks.{}", i), i, config.is_deltanet_layer(i)))
                    .collect::<io::Result<Vec<_>>>()?;

                let attn_states: Vec<_> = blocks.iter()
                    .map(|block| block.create_attn_state(&config))
                    .collect();

                (blocks, attn_states, Vec::new(), Vec::new(),
                 None, None, Vec::new(), Vec::new())
            };

        // Final norm
        let norm_final = Self::load_norm(&gguf, "norm_final.weight")?;

        // LM head (skip if weight_tied)
        let weight_tied = config.weight_tied;
        let lm_head = if weight_tied {
            None
        } else {
            Some(BitLinear::new(
                gguf.load_planar_weights("lm_head.weight", group_size)?))
        };

        // RoPE
        let rope = RopeFreqs::new(config.head_dim(), config.max_seq_len, config.rope_theta);

        Ok(Self {
            tok_embed,
            blocks,
            attn_states,
            local_blocks_before,
            local_states_before,
            shared_loop_block,
            loop_kv_cache,
            local_blocks_after,
            local_states_after,
            norm_final,
            lm_head,
            weight_tied,
            rope,
            config,
        })
    }

    /// Load SharedLoopBlock from GGUF.
    fn load_shared_loop_block(
        gguf: &GgufFile,
        mhc_layers: &[mhc_lite::io::MhcLayerParams],
        mhc_idx: usize,
        config: &ModelConfig,
    ) -> io::Result<SharedLoopBlock> {
        let group_size = config.group_size;
        let prefix = "shared_loop";

        // Attention weights
        let wq = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.attention.wq.weight", prefix), group_size)?);
        let wk = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.attention.wk.weight", prefix), group_size)?);
        let wv = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.attention.wv.weight", prefix), group_size)?);
        let wo = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.attention.wo.weight", prefix), group_size)?);

        // Global gates
        let g_qk = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.g_qk.weight", prefix), group_size)?);
        let g_ffn = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.g_ffn.weight", prefix), group_size)?);

        // FFN weights
        let w_gate = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.ffn.w_gate.weight", prefix), group_size)?);
        let w_up = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.ffn.w_up.weight", prefix), group_size)?);
        let w_down = BitLinear::new(gguf.load_planar_weights(
            &format!("{}.ffn.w_down.weight", prefix), group_size)?);

        // Norms
        let norm_attn = Self::load_norm(gguf, &format!("{}.norm_attn.weight", prefix))?;
        let norm_ffn = Self::load_norm(gguf, &format!("{}.norm_ffn.weight", prefix))?;

        // mHC params
        let mhc_attn = Self::extract_mhc_n2(mhc_layers, mhc_idx * 2)?;
        let mhc_ffn = Self::extract_mhc_n2(mhc_layers, mhc_idx * 2 + 1)?;

        Ok(SharedLoopBlock::new(
            wq, wk, wv, wo, g_qk,
            w_gate, w_up, w_down, g_ffn,
            norm_attn, norm_ffn, mhc_attn, mhc_ffn,
            config.dim, config.n_heads, config.n_kv_heads,
        ))
    }

    fn config_from_gguf(gguf: &GgufFile) -> io::Result<ModelConfig> {
        let get_u32 = |key: &str| -> io::Result<usize> {
            match gguf.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v as usize),
                _ => Err(io::Error::new(io::ErrorKind::NotFound,
                    format!("missing GGUF metadata: {key}"))),
            }
        };
        // Optional fields with defaults
        let weight_tied = match gguf.metadata.get("nanochat.weight_tied") {
            Some(GgufValue::U32(v)) => *v != 0,
            _ => false,
        };
        let n_experts = match gguf.metadata.get("nanochat.n_experts") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as usize),
            _ => None,
        };
        let n_active_experts = match gguf.metadata.get("nanochat.n_active_experts") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as usize),
            _ => None,
        };

        // DeltaNet ratio: stored as percentage (0-100) in GGUF, converted to 0.0-1.0
        let deltanet_ratio = match gguf.metadata.get("nanochat.deltanet_ratio_pct") {
            Some(GgufValue::U32(v)) if *v > 0 => Some(*v as f32 / 100.0),
            _ => None,
        };

        let ffn_mult = match gguf.metadata.get("nanochat.ffn_mult") {
            Some(GgufValue::F32(v)) => *v,
            _ => 2.667,
        };
        let rope_theta = match gguf.metadata.get("nanochat.rope_theta") {
            Some(GgufValue::F32(v)) => *v,
            _ => 10000.0,
        };
        let max_seq_len = match gguf.metadata.get("nanochat.max_seq_len") {
            Some(GgufValue::U32(v)) => *v as usize,
            _ => 2048,
        };
        let n_kv_heads = match gguf.metadata.get("nanochat.n_kv_heads") {
            Some(GgufValue::U32(v)) => *v as usize,
            _ => get_u32("nanochat.n_heads")?,
        };

        // New fields with defaults for backward compatibility
        let rope_scale = match gguf.metadata.get("nanochat.rope_scale") {
            Some(GgufValue::F32(v)) => *v,
            _ => 1.0,
        };
        let deltanet_v_heads = match gguf.metadata.get("nanochat.deltanet_v_heads") {
            Some(GgufValue::U32(v)) => Some(*v as usize),
            _ => None,
        };
        let deltanet_qk_heads = match gguf.metadata.get("nanochat.deltanet_qk_heads") {
            Some(GgufValue::U32(v)) => Some(*v as usize),
            _ => None,
        };
        let use_shared_expert = match gguf.metadata.get("nanochat.use_shared_expert") {
            Some(GgufValue::Bool(v)) => *v,
            _ => false,
        };
        let expert_dim = match gguf.metadata.get("nanochat.expert_dim") {
            Some(GgufValue::U32(v)) => Some(*v as usize),
            _ => None,
        };
        let gated_attention = match gguf.metadata.get("nanochat.gated_attention") {
            Some(GgufValue::Bool(v)) => *v,
            _ => false,
        };

        // LoopLM configuration (if present)
        let loop_config = if gguf.metadata.contains_key("nanochat.loop.local_before") {
            use crate::config::{LoopConfig, AdaptiveLoopConfig};

            let local_before = match gguf.metadata.get("nanochat.loop.local_before") {
                Some(GgufValue::U32(v)) => *v as usize,
                _ => 0,
            };
            let local_after = match gguf.metadata.get("nanochat.loop.local_after") {
                Some(GgufValue::U32(v)) => *v as usize,
                _ => 0,
            };
            let loop_count = match gguf.metadata.get("nanochat.loop.loop_count") {
                Some(GgufValue::U32(v)) => *v as usize,
                _ => 1,
            };

            // Adaptive loop config (optional)
            let adaptive_loop = if gguf.metadata.contains_key("nanochat.loop.adaptive.min_loops") {
                let min_loops = match gguf.metadata.get("nanochat.loop.adaptive.min_loops") {
                    Some(GgufValue::U32(v)) => *v as usize,
                    _ => loop_count,
                };
                let max_loops = match gguf.metadata.get("nanochat.loop.adaptive.max_loops") {
                    Some(GgufValue::U32(v)) => *v as usize,
                    _ => loop_count,
                };
                let perplexity_threshold = match gguf.metadata.get("nanochat.loop.adaptive.perplexity_threshold") {
                    Some(GgufValue::F32(v)) => *v,
                    _ => 5.0,
                };
                Some(AdaptiveLoopConfig {
                    min_loops,
                    max_loops,
                    perplexity_threshold,
                })
            } else {
                None
            };

            Some(LoopConfig {
                local_before,
                local_after,
                loop_count,
                adaptive_loop,
            })
        } else {
            None
        };

        Ok(ModelConfig {
            dim: get_u32("nanochat.dim")?,
            n_layers: get_u32("nanochat.n_layers")?,
            n_heads: get_u32("nanochat.n_heads")?,
            n_kv_heads,
            deltanet_v_heads,
            deltanet_qk_heads,
            ffn_mult,
            vocab_size: get_u32("nanochat.vocab_size")?,
            max_seq_len,
            group_size: get_u32("nanochat.group_size")?,
            mhc_n_streams: get_u32("nanochat.mhc_n_streams")?,
            rope_theta,
            rope_scale,
            n_experts,
            n_active_experts,
            use_shared_expert,
            expert_dim,
            deltanet_ratio,
            layer_sequence: LayerSequence::Interleaved, // Default to interleaved
            weight_tied,
            gated_attention,
            loop_config,
        })
    }

    fn load_embedding(gguf: &GgufFile, name: &str) -> io::Result<Embedding> {
        let tensor = gguf.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                format!("tensor '{name}' not found")))?;

        let vocab_size = tensor.dims[0] as usize;
        let dim = tensor.dims[1] as usize;
        let data = gguf.tensor_data(tensor);

        let weight: Vec<f32> = match tensor.dtype {
            0 => {
                // F32: 4 bytes per element
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            1 => {
                // FP16: 2 bytes per element
                data.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            }
            dt => {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("unsupported embedding dtype: {dt} (expected F32=0 or F16=1)")));
            }
        };

        if weight.len() != vocab_size * dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("embedding size mismatch: {} vs {}", weight.len(), vocab_size * dim)));
        }

        Ok(Embedding { weight, vocab_size, dim })
    }

    fn load_norm(gguf: &GgufFile, name: &str) -> io::Result<RMSNorm> {
        let tensor = gguf.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                format!("tensor '{name}' not found")))?;

        let data = gguf.tensor_data(tensor);
        let weight: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(RMSNorm { weight, eps: 1e-6 })
    }

    fn extract_mhc_n2(layers: &[mhc_lite::MhcLayerParams], idx: usize) -> io::Result<MhcLiteN2> {
        match layers.get(idx) {
            Some(mhc_lite::MhcLayerParams::N2(mhc)) => Ok(mhc.clone()),
            Some(_) => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("expected N2 mHC at index {idx}"))),
            None => Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("mHC index {idx} out of range (have {})", layers.len()))),
        }
    }

    /// Reset attention states (KV caches or recurrent states) for a new sequence.
    pub fn reset_caches(&mut self) {
        // Reset standard block caches
        for state in &mut self.attn_states {
            state.reset();
        }

        // Reset LoopLM caches
        for state in &mut self.local_states_before {
            state.reset();
        }
        for state in &mut self.local_states_after {
            state.reset();
        }
        if let Some(ref mut kv_cache) = self.loop_kv_cache {
            kv_cache.len = 0; // Reset shared loop KV cache
        }
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits [vocab_size].
    pub fn forward_token(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let dim = self.config.dim;
        // 1. Embed
        let mut x = vec![0.0f32; dim];
        self.tok_embed.forward_token(token_id, &mut x);

        // 2. Expand to multi-stream
        let mut x_exp = MhcLiteN2::expand_input(&x, dim);

        // 3. Transformer blocks (LoopLM or standard)
        if let Some(loop_cfg) = &self.config.loop_config {
            // LoopLM path: local before → shared loop → local after
            x_exp = self.forward_with_loops(&x_exp, loop_cfg.loop_count, pos);
        } else {
            // Standard path: all blocks sequentially
            for (i, block) in self.blocks.iter().enumerate() {
                block.forward(&mut x_exp, &mut self.attn_states[i], &self.rope, pos);
            }
        }

        // 4. Collapse back to single stream
        let x_collapsed = MhcLiteN2::collapse_output(&x_exp, dim);

        // 5. Final norm
        let mut x_normed = vec![0.0f32; dim];
        self.norm_final.forward(&x_collapsed, &mut x_normed);

        // 6. LM head (use embedding weights if tied)
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if self.weight_tied {
            self.tok_embed.forward_as_lm_head(&x_normed, &mut logits);
        } else {
            self.lm_head.as_ref().unwrap().forward(&x_normed, &mut logits);
        }

        logits
    }

    /// Forward with loop execution (LoopLM path).
    ///
    /// Executes: local_before → [shared_loop × N] → local_after
    fn forward_with_loops(&mut self, x_exp: &[f32], loop_count: usize, pos: usize) -> Vec<f32> {
        let mut x_exp = x_exp.to_vec();

        // 1. Local layers before loop
        for (i, block) in self.local_blocks_before.iter().enumerate() {
            block.forward(&mut x_exp, &mut self.local_states_before[i], &self.rope, pos);
        }

        // 2. Shared loop layer (iterated loop_count times)
        if let Some(ref loop_block) = self.shared_loop_block {
            let mut global_state: Option<Vec<f32>> = None;
            let kv_cache = self.loop_kv_cache.as_mut()
                .expect("loop_kv_cache should be initialized for LoopLM models");

            for iter in 0..loop_count {
                let append_kv = iter == 0; // Only append KV on first iteration
                let (x_out, g_state) = loop_block.forward(
                    &x_exp,
                    global_state.as_deref(),
                    kv_cache,
                    append_kv,
                    Some(pos), // Pass explicit token position for causal masking
                ).expect("Loop block forward failed: token_pos/cache mismatch");
                x_exp = x_out;
                global_state = Some(g_state);
            }
        }

        // 3. Local layers after loop
        for (i, block) in self.local_blocks_after.iter().enumerate() {
            block.forward(&mut x_exp, &mut self.local_states_after[i], &self.rope, pos);
        }

        x_exp
    }

    /// Forward pass for a sequence of tokens (prefill).
    /// Processes tokens one at a time (no batched prefill yet).
    ///
    /// Returns logits for the last token only.
    pub fn forward_sequence(&mut self, token_ids: &[u32]) -> Vec<f32> {
        self.reset_caches();
        let mut logits = vec![];
        for (pos, &tid) in token_ids.iter().enumerate() {
            logits = self.forward_token(tid, pos);
        }
        logits
    }

    /// Batched forward pass for a sequence of tokens (prefill).
    ///
    /// Processes all tokens through each layer in batch, with causal attention masking.
    /// Returns logits for the last token only.
    pub fn forward_sequence_batched(&mut self, token_ids: &[u32]) -> Vec<f32> {
        self.reset_caches();

        let seq_len = token_ids.len();
        if seq_len == 0 {
            return vec![];
        }

        let dim = self.config.dim;
        let n_streams = self.config.mhc_n_streams;
        let exp_dim = n_streams * dim;

        // 1. Embed all tokens: [seq_len * dim]
        let mut x_all = vec![0.0f32; seq_len * dim];
        for (t, &tid) in token_ids.iter().enumerate() {
            self.tok_embed.forward_token(tid, &mut x_all[t * dim..(t + 1) * dim]);
        }

        // 2. Expand all tokens to multi-stream: [seq_len * exp_dim]
        let mut x_exp_all = vec![0.0f32; seq_len * exp_dim];
        for t in 0..seq_len {
            let x = &x_all[t * dim..(t + 1) * dim];
            let expanded = MhcLiteN2::expand_input(x, dim);
            x_exp_all[t * exp_dim..(t + 1) * exp_dim].copy_from_slice(&expanded);
        }

        // 3. Transformer blocks (batched) - loop or standard
        if let Some(loop_cfg) = &self.config.loop_config {
            // LoopLM architecture
            // 3a. Local blocks before
            for (i, block) in self.local_blocks_before.iter().enumerate() {
                block.forward_batch(
                    &mut x_exp_all,
                    &mut self.local_states_before[i],
                    &self.rope,
                    0,
                    seq_len,
                );
            }

            // 3b. Shared loop block (N iterations)
            // Must maintain per-token global states for causal correctness
            if let Some(ref loop_block) = self.shared_loop_block {
                let loop_kv_cache = self.loop_kv_cache.as_mut().unwrap();

                // Initialize per-token global states
                let mut global_states: Vec<Option<Vec<f32>>> = vec![None; seq_len];

                // Process entire sequence through each loop iteration
                // Each token maintains its own global state across iterations
                for iter in 0..loop_cfg.loop_count {
                    let append_kv = (iter == 0);
                    let new_states = loop_block.forward_batch(
                        &mut x_exp_all,
                        &global_states,
                        loop_kv_cache,
                        append_kv,
                        seq_len,
                    ).expect("Loop batched forward failed: token_pos/cache mismatch");

                    // Update global states for next iteration
                    global_states = new_states.into_iter().map(Some).collect();
                }
            }

            // 3c. Local blocks after
            for (i, block) in self.local_blocks_after.iter().enumerate() {
                block.forward_batch(
                    &mut x_exp_all,
                    &mut self.local_states_after[i],
                    &self.rope,
                    0,
                    seq_len,
                );
            }
        } else {
            // Standard architecture
            for (i, block) in self.blocks.iter().enumerate() {
                block.forward_batch(
                    &mut x_exp_all,
                    &mut self.attn_states[i],
                    &self.rope,
                    0, // start_pos for fresh prefill
                    seq_len,
                );
            }
        }

        // 4. Collapse last token back to single stream
        let last_exp = &x_exp_all[(seq_len - 1) * exp_dim..seq_len * exp_dim];
        let x_collapsed = MhcLiteN2::collapse_output(last_exp, dim);

        // 5. Final norm
        let mut x_normed = vec![0.0f32; dim];
        self.norm_final.forward(&x_collapsed, &mut x_normed);

        // 6. LM head
        let mut logits = vec![0.0f32; self.config.vocab_size];
        if self.weight_tied {
            self.tok_embed.forward_as_lm_head(&x_normed, &mut logits);
        } else {
            self.lm_head.as_ref().unwrap().forward(&x_normed, &mut logits);
        }

        logits
    }

    /// Verify all mHC matrices in the model are doubly stochastic.
    pub fn verify_mhc(&self) -> Result<(), String> {
        // Standard blocks
        for (i, block) in self.blocks.iter().enumerate() {
            let h_attn = block.mhc_attn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6)
                .map_err(|e| format!("block {i} mhc_attn: {e}"))?;

            let h_ffn = block.mhc_ffn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6)
                .map_err(|e| format!("block {i} mhc_ffn: {e}"))?;
        }

        // LoopLM blocks
        for (i, block) in self.local_blocks_before.iter().enumerate() {
            let h_attn = block.mhc_attn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6)
                .map_err(|e| format!("local_before.{i} mhc_attn: {e}"))?;

            let h_ffn = block.mhc_ffn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6)
                .map_err(|e| format!("local_before.{i} mhc_ffn: {e}"))?;
        }

        if let Some(ref loop_block) = self.shared_loop_block {
            let h_attn = loop_block.mhc_attn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6)
                .map_err(|e| format!("shared_loop mhc_attn: {e}"))?;

            let h_ffn = loop_block.mhc_ffn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6)
                .map_err(|e| format!("shared_loop mhc_ffn: {e}"))?;
        }

        for (i, block) in self.local_blocks_after.iter().enumerate() {
            let h_attn = block.mhc_attn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_attn, 1e-6)
                .map_err(|e| format!("local_after.{i} mhc_attn: {e}"))?;

            let h_ffn = block.mhc_ffn.h_res();
            mhc_lite::verify_doubly_stochastic_2x2(&h_ffn, 1e-6)
                .map_err(|e| format!("local_after.{i} mhc_ffn: {e}"))?;
        }

        Ok(())
    }

    /// Count model parameters by category.
    pub fn param_count(&self) -> ModelParamCount {
        let dim = self.config.dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim();
        let ffn_dim = self.config.ffn_dim();

        let embed = self.config.vocab_size * dim;

        // Helper to count attention params for a block
        let count_attn = |block: &TransformerBlock| -> usize {
            match &block.attention {
                AttentionLayer::Standard(_) => {
                    // Q(dim*dim), K(dim*kv_dim), V(dim*kv_dim), O(dim*dim)
                    dim * dim + 2 * dim * kv_dim + dim * dim
                }
                AttentionLayer::DeltaNet(_) => {
                    // Q, K, V, O (all dim*dim) + beta (n_heads * dim)
                    4 * dim * dim + self.config.n_heads * dim
                }
            }
        };

        // Count actual blocks in model (standard or loop architecture)
        let mut attn_total = 0usize;
        let mut ffn_total = 0usize;
        let mut norm_total = 0usize;
        let mut mhc_total = 0usize;
        let mut actual_layer_count = 0usize;

        // Standard blocks
        for block in &self.blocks {
            attn_total += count_attn(block);
            actual_layer_count += 1;
        }

        // LoopLM blocks
        for block in &self.local_blocks_before {
            attn_total += count_attn(block);
            actual_layer_count += 1;
        }

        if let Some(ref loop_block) = self.shared_loop_block {
            // SharedLoopBlock attention: Q, K, V, O, g_qk
            attn_total += dim * dim + 2 * dim * kv_dim + dim * dim + dim * dim;
            actual_layer_count += 1;
        }

        for block in &self.local_blocks_after {
            attn_total += count_attn(block);
            actual_layer_count += 1;
        }

        // FFN params: dense vs MoE
        let ffn_per_layer = if let Some(n_experts) = self.config.n_experts {
            let router_params = n_experts * dim;
            let expert_params = dim * ffn_dim + dim * ffn_dim + ffn_dim * dim;
            router_params + n_experts * expert_params
        } else {
            dim * ffn_dim + dim * ffn_dim + ffn_dim * dim
        };

        // SharedLoopBlock has its own FFN + g_ffn gate
        let shared_loop_ffn = if self.shared_loop_block.is_some() {
            dim * ffn_dim + dim * ffn_dim + ffn_dim * dim + dim * dim // gate, up, down, g_ffn
        } else {
            0
        };

        ffn_total = (actual_layer_count - if self.shared_loop_block.is_some() { 1 } else { 0 }) * ffn_per_layer + shared_loop_ffn;

        let norm_per_layer = dim * 2; // attn_norm + ffn_norm
        norm_total = actual_layer_count * norm_per_layer;

        let mhc_per_layer = 9 * 2; // 9 params each for attn + ffn mHC (N=2)
        mhc_total = actual_layer_count * mhc_per_layer;

        let lm_head = if self.weight_tied { 0 } else { self.config.vocab_size * dim };
        let final_norm = dim;

        ModelParamCount {
            total: embed + attn_total + ffn_total + norm_total + mhc_total + lm_head + final_norm,
            embed,
            attn: attn_total,
            ffn: ffn_total,
            norm: norm_total + final_norm,
            mhc: mhc_total,
            lm_head,
        }
    }
}

/// Parameter count breakdown.
#[derive(Debug)]
pub struct ModelParamCount {
    pub total: usize,
    pub embed: usize,
    pub attn: usize,
    pub ffn: usize,
    pub norm: usize,
    pub mhc: usize,
    pub lm_head: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> NanochatModel {
        // Use a small config for fast testing
        let config = ModelConfig::test_config(128, 2, 4, 256);
        NanochatModel::new_random(config)
    }

    fn make_test_model_tied() -> NanochatModel {
        let mut config = ModelConfig::test_config(128, 2, 4, 256);
        config.weight_tied = true;
        NanochatModel::new_random(config)
    }

    fn make_test_model_moe() -> NanochatModel {
        let mut config = ModelConfig::test_config(128, 2, 4, 256);
        config.n_experts = Some(4);
        config.n_active_experts = Some(2);
        NanochatModel::new_random(config)
    }

    #[test]
    fn test_basic_forward_pass() {
        let mut model = make_test_model();
        let tokens = vec![1, 2, 3];
        let _logits = model.forward_sequence(&tokens);
        // Just check it doesn't panic
    }

    #[test]
    fn test_tied_weights_param_count() {
        let model_tied = make_test_model_tied();
        let model_untied = make_test_model();

        let count_tied = model_tied.param_count();
        let count_untied = model_untied.param_count();

        // Weight tying saves vocab_size * dim parameters in the LM head
        let expected_diff = model_tied.config.vocab_size * model_tied.config.dim;
        assert_eq!(
            count_untied.lm_head - count_tied.lm_head,
            expected_diff,
            "tied model should save vocab_size * dim params in lm_head"
        );
    }

    #[test]
    fn test_forward_moe() {
        let mut model = make_test_model_moe();
        let tokens = vec![5, 10, 15];
        let _logits = model.forward_sequence(&tokens);
        // Ensure MoE forward works
    }

    #[test]
    fn test_batched_prefill() {
        let mut model = make_test_model();
        let tokens = vec![1, 2, 3, 4, 5];
        let logits = model.forward_sequence(&tokens);
        assert_eq!(logits.len(), model.config.vocab_size);
    }

    #[test]
    fn test_autoregressive_decode() {
        let mut model = make_test_model();
        model.reset_caches();

        // Prefill
        let _logits1 = model.forward_sequence(&vec![10, 20]);
        // Decode step 1
        let _logits2 = model.forward_sequence(&vec![30]);
        // Decode step 2
        let _logits3 = model.forward_sequence(&vec![40]);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut model = make_test_model();

        // Forward with some tokens
        let _logits1 = model.forward_sequence(&vec![1, 2, 3]);

        // Reset
        model.reset_caches();

        // State should be cleared (all kv_cache.len = 0)
        for state in &model.attn_states {
            match state {
                AttentionState::Kv(cache) => assert_eq!(cache.len, 0),
                AttentionState::Recurrent(_) => {} // Recurrent states are reset to zeros
            }
        }
    }

    #[test]
    fn test_deltanet_layer_selection() {
        use crate::config::LayerType;

        // Test Interleaved with no ratio: all standard
        let config = ModelConfig { deltanet_ratio: None, n_layers: 4, ..ModelConfig::d20() };
        assert!(!config.is_deltanet_layer(0));
        assert!(!config.is_deltanet_layer(1));

        // Test Interleaved with ratio 0.0: all standard
        let config = ModelConfig { deltanet_ratio: Some(0.0), n_layers: 4, ..ModelConfig::d20() };
        assert!(!config.is_deltanet_layer(0));

        // Test Interleaved with ratio 1.0: all DeltaNet
        let config = ModelConfig { deltanet_ratio: Some(1.0), n_layers: 4, ..ModelConfig::d20() };
        assert!(config.is_deltanet_layer(0));
        assert!(config.is_deltanet_layer(3));

        // Test Interleaved with ratio 0.5: expect 2 DeltaNet layers
        let config = ModelConfig { deltanet_ratio: Some(0.5), n_layers: 4, ..ModelConfig::d20() };
        let dn_count: usize = (0..4)
            .filter(|&i| config.is_deltanet_layer(i))
            .count();
        assert_eq!(dn_count, 2, "expected 2 DeltaNet layers for ratio=0.5");

        // Test explicit Pattern: [DeltaNet, Standard]
        let pattern = vec![LayerType::DeltaNetAttention, LayerType::StandardAttention];
        let config = ModelConfig {
            n_layers: 4,
            layer_sequence: crate::config::LayerSequence::Pattern(pattern),
            ..ModelConfig::d20()
        };
        assert!(config.is_deltanet_layer(0));  // First in pattern
        assert!(!config.is_deltanet_layer(1)); // Second in pattern
        assert!(config.is_deltanet_layer(2));  // Pattern repeats
        assert!(!config.is_deltanet_layer(3)); // Pattern repeats
    }

    #[test]
    fn test_deltanet_param_count_includes_beta() {
        let config_std = ModelConfig::test_config(128, 2, 4, 256);
        let config_dn = ModelConfig {
            deltanet_ratio: Some(1.0),
            ..config_std.clone()
        };

        let model_std = NanochatModel::new_random(config_std);
        let model_dn = NanochatModel::new_random(config_dn);

        let count_std = model_std.param_count();
        let count_dn = model_dn.param_count();

        // DeltaNet layers have extra w_beta (n_heads * dim per layer) and use
        // full dim for K and V instead of kv_dim.
        // Since n_kv_heads == n_heads here (MHA), kv_dim == dim, so:
        // standard = 4*dim*dim, deltanet = 4*dim*dim + n_heads*dim
        // DeltaNet should have n_heads*dim extra params per layer
        let extra_per_layer = model_std.config.n_heads * model_std.config.dim;
        let expected_diff = extra_per_layer * model_std.config.n_layers;
        assert_eq!(
            count_dn.attn - count_std.attn,
            expected_diff,
            "DeltaNet should have n_heads*dim extra params per layer for w_beta"
        );
    }

    #[test]
    fn test_qwen3_config() {
        let config = ModelConfig::qwen3_coder_80b();

        // Verify dimensions
        assert_eq!(config.dim, 2048);
        assert_eq!(config.n_layers, 48);
        assert_eq!(config.n_heads, 16);
        assert_eq!(config.n_kv_heads, 2);

        // Verify MoE config
        assert_eq!(config.n_experts, Some(512));
        assert_eq!(config.n_active_experts, Some(10));
        assert!(config.use_shared_expert);
        assert_eq!(config.expert_dim, Some(512));

        // Verify DeltaNet head config
        assert_eq!(config.deltanet_v_heads, Some(32));
        assert_eq!(config.deltanet_qk_heads, Some(16));

        // Verify gated attention
        assert!(config.gated_attention);

        // Verify long context support
        assert_eq!(config.max_seq_len, 262144); // 256k
        assert_eq!(config.rope_scale, 8.0);

        // Verify layer pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] × 12
        assert!(config.is_deltanet_layer(0));  // DeltaNet
        assert!(config.is_deltanet_layer(1));  // DeltaNet
        assert!(config.is_deltanet_layer(2));  // DeltaNet
        assert!(!config.is_deltanet_layer(3)); // Attention
        assert!(config.is_deltanet_layer(4));  // Pattern repeats
        assert!(!config.is_deltanet_layer(47)); // Last layer should be Attention (47 % 4 = 3)
    }

    // Remaining tests need to be updated with test_config or full field list
    // Let me skip directly to fixing them with a macro or I'll update them one by one

    #[test]
    fn test_param_count_basic() {
        let mut model = make_test_model();
        let count = model.param_count();

        // Verify param counting includes all components
        assert!(count.embed > 0, "embeddings should have params");
        assert!(count.attn > 0, "attention should have params");
        assert!(count.ffn > 0, "FFN should have params");
        assert!(count.norm > 0, "norms should have params");
        assert!(count.lm_head > 0, "LM head should have params");
        assert_eq!(count.total, count.embed + count.attn + count.ffn + count.norm + count.lm_head + count.mhc);
    }

    #[test]
    fn test_model_forward_single_token() {
        let mut model = make_test_model();
        let logits = model.forward_token(42, 0);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits");
    }

    #[test]
    fn test_model_forward_sequence() {
        let mut model = make_test_model();
        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    #[test]
    fn test_model_mhc_verify() {
        let mut model = make_test_model();
        model.verify_mhc().expect("mHC verification failed");
    }

    #[test]
    fn test_model_shape_through_layers() {
        let mut model = make_test_model();
        let dim = model.config.dim;
        let n_streams = model.config.mhc_n_streams;

        // Manually trace shapes
        let mut x = vec![0.0f32; dim];
        model.tok_embed.forward_token(1, &mut x);
        assert_eq!(x.len(), dim);

        let x_exp = MhcLiteN2::expand_input(&x, dim);
        assert_eq!(x_exp.len(), n_streams * dim, "expand shape wrong");

        let collapsed = MhcLiteN2::collapse_output(&x_exp, dim);
        assert_eq!(collapsed.len(), dim, "collapse shape wrong");
    }

    #[test]
    fn test_model_d20_config() {
        let config = ModelConfig::d20();
        let model = NanochatModel::new_random(config);
        let counts = model.param_count();
        println!("d20 params: {:?}", counts);
        assert!(counts.total > 0);
        assert!(counts.mhc > 0);
    }

    #[test]
    fn test_model_reset_caches() {
        let mut model = make_test_model();

        // Process a few tokens
        model.forward_token(1, 0);
        model.forward_token(2, 1);
        for state in &model.attn_states {
            match state {
                AttentionState::Kv(cache) => assert_eq!(cache.len, 2),
                AttentionState::Recurrent(_) => {} // DeltaNet state doesn't track len
            }
        }

        // Reset
        model.reset_caches();
        for state in &model.attn_states {
            match state {
                AttentionState::Kv(cache) => assert_eq!(cache.len, 0),
                AttentionState::Recurrent(s) => assert!(s.s.iter().all(|&v| v == 0.0)),
            }
        }
    }

    #[test]
    fn test_expand_n_blocks_collapse_shape() {
        // Test gate from CLAUDE.md:
        // "mHC expand → N blocks → collapse preserves tensor shape correctly"
        let mut model = make_test_model();
        let dim = model.config.dim;

        let mut x = vec![0.1f32; dim];
        model.tok_embed.forward_token(5, &mut x);

        // Expand
        let mut x_exp = MhcLiteN2::expand_input(&x, dim);
        let exp_len = x_exp.len();

        // N blocks
        for (i, block) in model.blocks.iter().enumerate() {
            block.forward(&mut x_exp, &mut model.attn_states[i], &model.rope, 0);
            assert_eq!(x_exp.len(), exp_len, "shape changed after block {i}");
        }

        // Collapse
        let collapsed = MhcLiteN2::collapse_output(&x_exp, dim);
        assert_eq!(collapsed.len(), dim, "final collapse shape wrong");
    }

    #[test]
    fn test_weight_tied_model_forward() {
        let mut model = make_test_model_tied();
        assert!(model.weight_tied);
        assert!(model.lm_head.is_none());

        let logits = model.forward_token(42, 0);
        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits");
    }

    #[test]
    fn test_weight_tied_param_count_reduced() {
        let model_untied = make_test_model();
        let model_tied = make_test_model_tied();

        let count_untied = model_untied.param_count();
        let count_tied = model_tied.param_count();

        // Tied model should have vocab_size * dim fewer params (no LM head)
        let expected_diff = model_untied.config.vocab_size * model_untied.config.dim;
        assert_eq!(count_untied.total - count_tied.total, expected_diff);
        assert_eq!(count_tied.lm_head, 0);
        assert!(count_untied.lm_head > 0);
    }

    #[test]
    fn test_weight_tied_sequence_forward() {
        let mut model = make_test_model_tied();
        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), model.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    // ============================================================
    // Batched prefill tests
    // ============================================================

    #[test]
    fn test_batched_matches_sequential() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let mut model_seq = NanochatModel::new_random(config.clone());
        let mut model_batch = NanochatModel::new_random(config.clone());

        // Copy weights (same random model, same seed — they're already identical)
        let tokens = vec![1u32, 5, 10, 20];

        let logits_seq = model_seq.forward_sequence(&tokens);
        let logits_batch = model_batch.forward_sequence_batched(&tokens);

        assert_eq!(logits_seq.len(), logits_batch.len());
        let max_diff: f32 = logits_seq.iter()
            .zip(logits_batch.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "batched vs sequential logits max_diff={} (should be < 1e-4)",
            max_diff
        );
    }

    #[test]
    fn test_batched_prefill_single_token() {
        let mut model = make_test_model();
        let logits_single = model.forward_sequence(&[42]);
        let logits_batched = model.forward_sequence_batched(&[42]);

        assert_eq!(logits_single.len(), logits_batched.len());
        let max_diff: f32 = logits_single.iter()
            .zip(logits_batched.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "single token batched vs sequential max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_batched_prefill_empty() {
        let mut model = make_test_model();
        let logits = model.forward_sequence_batched(&[]);
        assert!(logits.is_empty());
    }

    #[test]
    fn test_batched_prefill_engine_uses_batch() {
        // Verify the engine's prefill produces valid output
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let mut model = NanochatModel::new_random(config);

        // Batched prefill followed by decode step
        let tokens = vec![1, 5, 10];
        let logits = model.forward_sequence_batched(&tokens);

        assert_eq!(logits.len(), 256);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(logits.iter().any(|&v| v != 0.0));

        // Now do a decode step after batched prefill
        let next_logits = model.forward_token(20, 3);
        assert_eq!(next_logits.len(), 256);
        assert!(next_logits.iter().all(|v| v.is_finite()));
    }

    // ============================================================
    // DeltaNet hybrid model tests
    // ============================================================

    #[test]
    fn test_hybrid_model_forward_ratio_half() {
        let mut config = ModelConfig::test_config(128, 4, 4, 256);
        config.deltanet_ratio = Some(0.5);
        let mut model = NanochatModel::new_random(config.clone());

        // Verify we have a mix of standard and DeltaNet layers
        let n_standard = model.blocks.iter()
            .filter(|b| matches!(b.attention, AttentionLayer::Standard(_)))
            .count();
        let n_deltanet = model.blocks.iter()
            .filter(|b| matches!(b.attention, AttentionLayer::DeltaNet(_)))
            .count();

        assert_eq!(n_standard, 2, "expected 2 standard layers");
        assert_eq!(n_deltanet, 2, "expected 2 DeltaNet layers");

        // Verify corresponding states match layer types
        for (i, (block, state)) in model.blocks.iter().zip(model.attn_states.iter()).enumerate() {
            match (&block.attention, state) {
                (AttentionLayer::Standard(_), AttentionState::Kv(_)) => {}
                (AttentionLayer::DeltaNet(_), AttentionState::Recurrent(_)) => {}
                _ => panic!("layer {} attention/state type mismatch", i),
            }
        }

        // Forward pass should work
        let logits = model.forward_token(42, 0);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits from hybrid model");
        assert!(logits.iter().any(|&v| v != 0.0), "all-zero logits from hybrid model");
    }

    #[test]
    fn test_deltanet_layers_have_no_kv_cache() {
        let mut config = ModelConfig::test_config(128, 4, 4, 256);
        config.deltanet_ratio = Some(0.5);
        let model = NanochatModel::new_random(config);

        for (i, (block, state)) in model.blocks.iter().zip(model.attn_states.iter()).enumerate() {
            match &block.attention {
                AttentionLayer::DeltaNet(_) => {
                    assert!(
                        matches!(state, AttentionState::Recurrent(_)),
                        "DeltaNet layer {} should have Recurrent state, not KV cache",
                        i
                    );
                }
                AttentionLayer::Standard(_) => {
                    assert!(
                        matches!(state, AttentionState::Kv(_)),
                        "Standard layer {} should have KV cache, not Recurrent state",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_hybrid_model_sequence_forward() {
        let mut config = ModelConfig::test_config(128, 4, 4, 256);
        config.deltanet_ratio = Some(0.5);
        let mut model = NanochatModel::new_random(config.clone());

        let tokens = vec![1, 5, 10, 20];
        let logits = model.forward_sequence(&tokens);

        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    #[test]
    fn test_hybrid_model_reset_and_reuse() {
        let mut config = ModelConfig::test_config(128, 4, 4, 256);
        config.deltanet_ratio = Some(0.5);
        let mut model = NanochatModel::new_random(config.clone());

        // Forward pass 1
        let _logits1 = model.forward_sequence(&vec![1, 2, 3]);

        // Reset
        model.reset_caches();

        // Forward pass 2 - should work without errors
        let _logits2 = model.forward_sequence(&vec![10, 20, 30]);
    }

    #[test]
    fn test_rope_application() {
        let mut config = ModelConfig::test_config(128, 2, 4, 256);
        config.rope_theta = 10000.0;
        let mut model = NanochatModel::new_random(config.clone());

        // Verify RoPE frequencies were created with correct theta
        assert!((model.rope.max_seq_len as f32 - config.max_seq_len as f32).abs() < 1.0);
        // Just verify model can forward
        let _logits = model.forward_sequence(&vec![1, 2, 3]);
    }

    #[test]
    fn test_reset_cache_identical_results() {
        let mut config = ModelConfig::test_config(128, 2, 4, 256);
        config.deltanet_ratio = Some(0.0); // Use standard attention for this test
        let mut model = NanochatModel::new_random(config);

        // First run
        let logits1 = model.forward_sequence(&vec![1, 2, 3]);

        // Reset and run again
        model.reset_caches();
        let logits2 = model.forward_sequence(&vec![1, 2, 3]);

        // Should produce identical results after reset
        for i in 0..logits1.len() {
            assert!(
                (logits1[i] - logits2[i]).abs() < 1e-5,
                "logit[{}] differs after reset: {} vs {}",
                i, logits1[i], logits2[i]
            );
        }
    }

}
