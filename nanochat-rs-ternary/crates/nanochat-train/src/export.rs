//! Export trained model to GGUF + mHC binary for inference.

use candle_core::Result;

use crate::config::TrainConfig;
use crate::model::NanochatTrainModel;
use crate::quantize::dequantize_ternary;

/// Export trained model weights to GGUF format.
///
/// Extracts ternary weights from BitLinearSTE layers, packs them using
/// ternary-core, and writes to a GGUF file. Embeddings are stored as FP16,
/// norms as FP32.
pub fn export_gguf(model: &NanochatTrainModel, config: &TrainConfig, path: &str) -> Result<()> {
    use ternary_core::gguf::{GgufValue, GgufWriter};
    use ternary_core::planar::PlanarWeights;

    let mut writer = GgufWriter::new();

    // Metadata
    writer.add_metadata(
        "general.architecture",
        GgufValue::String("nanochat-ternary".to_string()),
    );
    writer.add_metadata("nanochat.dim", GgufValue::U32(config.dim as u32));
    writer.add_metadata("nanochat.n_layers", GgufValue::U32(config.n_layers as u32));
    writer.add_metadata("nanochat.n_heads", GgufValue::U32(config.n_heads as u32));
    writer.add_metadata(
        "nanochat.n_kv_heads",
        GgufValue::U32(config.n_kv_heads as u32),
    );
    writer.add_metadata(
        "nanochat.vocab_size",
        GgufValue::U32(config.vocab_size as u32),
    );
    writer.add_metadata(
        "nanochat.max_seq_len",
        GgufValue::U32(config.max_seq_len as u32),
    );
    writer.add_metadata(
        "nanochat.group_size",
        GgufValue::U32(config.group_size as u32),
    );
    writer.add_metadata(
        "nanochat.mhc_n_streams",
        GgufValue::U32(config.mhc_n_streams as u32),
    );
    writer.add_metadata(
        "nanochat.weight_tied",
        GgufValue::U32(config.weight_tied as u32),
    );
    writer.add_metadata("nanochat.rope_theta", GgufValue::F32(config.rope_theta));
    writer.add_metadata("nanochat.ffn_mult", GgufValue::F32(config.ffn_mult));

    // LoopLM metadata (if present)
    if let Some(ref loop_cfg) = config.loop_config {
        writer.add_metadata(
            "nanochat.loop.local_before",
            GgufValue::U32(loop_cfg.local_before as u32),
        );
        writer.add_metadata(
            "nanochat.loop.local_after",
            GgufValue::U32(loop_cfg.local_after as u32),
        );
        writer.add_metadata(
            "nanochat.loop.loop_count",
            GgufValue::U32(loop_cfg.loop_count as u32),
        );

        // Adaptive loop config (optional)
        if let Some(ref adaptive) = loop_cfg.adaptive_loop {
            writer.add_metadata(
                "nanochat.loop.adaptive.min_loops",
                GgufValue::U32(adaptive.min_loops as u32),
            );
            writer.add_metadata(
                "nanochat.loop.adaptive.max_loops",
                GgufValue::U32(adaptive.max_loops as u32),
            );
            writer.add_metadata(
                "nanochat.loop.adaptive.perplexity_threshold",
                GgufValue::F32(adaptive.perplexity_threshold),
            );
        }
    }

    // Wave field metadata (if enabled)
    if config.use_wave_field {
        let wf_heads = if config.wavefield_n_heads == 0 {
            config.n_heads
        } else {
            config.wavefield_n_heads
        };
        let wf_head_dim = config.dim / wf_heads;
        writer.add_metadata(
            "nanochat.wavefield.field_size",
            GgufValue::U32(config.wavefield_field_size as u32),
        );
        writer.add_metadata(
            "nanochat.wavefield.n_wave_heads",
            GgufValue::U32(wf_heads as u32),
        );
        writer.add_metadata(
            "nanochat.wavefield.head_dim",
            GgufValue::U32(wf_head_dim as u32),
        );
        writer.add_metadata(
            "nanochat.wavefield.head_coupling",
            GgufValue::Bool(config.wavefield_head_coupling),
        );
        writer.add_metadata(
            "nanochat.wavefield.ratio_pct",
            GgufValue::U32((config.wavefield_ratio * 100.0) as u32),
        );
        // Convolution mode for wave field (default: "fft" for backward compat)
        let convolve_mode_str = match config.wavefield_convolve_mode.as_deref() {
            Some("fwht") => "fwht",
            Some("haar") => "haar",
            Some("fft") | None => "fft",
            Some(other) => {
                return Err(candle_core::Error::Msg(format!(
                    "unknown wavefield_convolve_mode '{}'; expected 'fft', 'fwht', or 'haar'",
                    other
                )));
            }
        };
        writer.add_metadata(
            "nanochat.wavefield.convolve_mode",
            GgufValue::String(convolve_mode_str.to_string()),
        );
        if let Some(levels) = config.wavefield_haar_levels {
            writer.add_metadata(
                "nanochat.wavefield.haar_levels",
                GgufValue::U32(levels as u32),
            );
        }
    }

    // Token embedding (FP32 -> stored as f32 in GGUF)
    let embed_data = model
        .tok_embed
        .embeddings()
        .flatten_all()?
        .to_vec1::<f32>()?;
    writer.add_f32_tensor(
        "tok_embed.weight",
        &[config.vocab_size as u64, config.dim as u64],
        &embed_data,
    );

    // Helper to export a BitLinearSTE as a ternary tensor
    let export_ternary_layer = |writer: &mut GgufWriter,
                                 name: &str,
                                 layer: &crate::layers::BitLinearSTE|
     -> Result<()> {
        let (w_ternary, scales) = layer.get_ternary_weights()?;
        let w_deq = dequantize_ternary(&w_ternary, &scales, layer.group_size)?;
        let w_flat = w_deq.flatten_all()?.to_vec1::<f32>()?;
        let rows = layer.out_features;
        let cols = layer.in_features;
        let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
        writer.add_ternary_tensor(name, &pw);
        Ok(())
    };

    // Helper to export a single block's weights
    let export_block = |writer: &mut GgufWriter,
                        prefix: &str,
                        block: &crate::block::TransformerBlockTrain|
     -> Result<()> {
        // Attention weights — dispatch based on attention type
        match &block.attention {
            crate::block::AttentionTrainLayer::Standard(attn) => {
                for (name, layer) in [
                    ("attention.wq", &attn.wq),
                    ("attention.wk", &attn.wk),
                    ("attention.wv", &attn.wv),
                    ("attention.wo", &attn.wo),
                ] {
                    export_ternary_layer(writer, &format!("{}.{}.weight", prefix, name), layer)?;
                }
            }
            crate::block::AttentionTrainLayer::WaveField(wf) => {
                // Ternary projections
                for (name, layer) in [
                    ("wavefield.scatter", &wf.scatter_proj),
                    ("wavefield.gate", &wf.gate_proj),
                    ("wavefield.out", &wf.out_proj),
                ] {
                    export_ternary_layer(writer, &format!("{}.{}.weight", prefix, name), layer)?;
                }
                // Physics params (FP32)
                let omega_data = wf.omega.flatten_all()?.to_vec1::<f32>()?;
                writer.add_f32_tensor(
                    &format!("{}.wavefield.omega", prefix),
                    &[wf.n_heads as u64],
                    &omega_data,
                );
                let alpha_data = wf.alpha_raw.flatten_all()?.to_vec1::<f32>()?;
                writer.add_f32_tensor(
                    &format!("{}.wavefield.alpha_raw", prefix),
                    &[wf.n_heads as u64],
                    &alpha_data,
                );
                let phi_data = wf.phi.flatten_all()?.to_vec1::<f32>()?;
                writer.add_f32_tensor(
                    &format!("{}.wavefield.phi", prefix),
                    &[wf.n_heads as u64],
                    &phi_data,
                );
                // Coupling logits (FP32, optional)
                if let Some(ref logits) = wf.coupling_logits {
                    let logits_data = logits.flatten_all()?.to_vec1::<f32>()?;
                    writer.add_f32_tensor(
                        &format!("{}.wavefield.coupling_logits", prefix),
                        &[wf.n_heads as u64, wf.n_heads as u64],
                        &logits_data,
                    );
                }
            }
        }

        // FFN ternary weights
        for (name, layer) in [
            ("ffn.w_gate", &block.ffn.w_gate),
            ("ffn.w_up", &block.ffn.w_up),
            ("ffn.w_down", &block.ffn.w_down),
        ] {
            export_ternary_layer(writer, &format!("{}.{}.weight", prefix, name), layer)?;
        }

        // Norm weights (FP32)
        let norm_attn = block.norm_attn.weight().flatten_all()?.to_vec1::<f32>()?;
        writer.add_f32_tensor(
            &format!("{}.norm_attn.weight", prefix),
            &[config.dim as u64],
            &norm_attn,
        );
        let norm_ffn = block.norm_ffn.weight().flatten_all()?.to_vec1::<f32>()?;
        writer.add_f32_tensor(
            &format!("{}.norm_ffn.weight", prefix),
            &[config.dim as u64],
            &norm_ffn,
        );

        Ok(())
    };

    // Export blocks based on architecture
    if let Some(_loop_cfg) = &config.loop_config {
        // LoopLM architecture: local_before + shared_loop + local_after
        for (i, block) in model.local_blocks_before.iter().enumerate() {
            export_block(&mut writer, &format!("local_before.{}", i), block)?;
        }

        // Export shared loop block
        if let Some(ref loop_block) = model.shared_loop_block {
            let prefix = "shared_loop";

            // Attention weights (dequantize with original scales)
            for (name, layer) in [
                ("attention.wq", &loop_block.wq),
                ("attention.wk", &loop_block.wk),
                ("attention.wv", &loop_block.wv),
                ("attention.wo", &loop_block.wo),
            ] {
                let (w_ternary, scales) = layer.get_ternary_weights()?;
                let w_deq = dequantize_ternary(&w_ternary, &scales, layer.group_size)?;
                let w_flat = w_deq.flatten_all()?.to_vec1::<f32>()?;
                let rows = layer.out_features;
                let cols = layer.in_features;

                let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
                writer.add_ternary_tensor(&format!("{}.{}.weight", prefix, name), &pw);
            }

            // Global gates (dequantize with original scales)
            for (name, layer) in [("g_qk", &loop_block.g_qk), ("g_ffn", &loop_block.g_ffn)] {
                let (w_ternary, scales) = layer.get_ternary_weights()?;
                let w_deq = dequantize_ternary(&w_ternary, &scales, layer.group_size)?;
                let w_flat = w_deq.flatten_all()?.to_vec1::<f32>()?;
                let rows = layer.out_features;
                let cols = layer.in_features;

                let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
                writer.add_ternary_tensor(&format!("{}.{}.weight", prefix, name), &pw);
            }

            // FFN weights (dequantize with original scales)
            for (name, layer) in [
                ("ffn.w_gate", &loop_block.w_gate),
                ("ffn.w_up", &loop_block.w_up),
                ("ffn.w_down", &loop_block.w_down),
            ] {
                let (w_ternary, scales) = layer.get_ternary_weights()?;
                let w_deq = dequantize_ternary(&w_ternary, &scales, layer.group_size)?;
                let w_flat = w_deq.flatten_all()?.to_vec1::<f32>()?;
                let rows = layer.out_features;
                let cols = layer.in_features;

                let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
                writer.add_ternary_tensor(&format!("{}.{}.weight", prefix, name), &pw);
            }

            // Norms
            let norm_attn = loop_block
                .norm_attn
                .weight()
                .flatten_all()?
                .to_vec1::<f32>()?;
            writer.add_f32_tensor(
                &format!("{}.norm_attn.weight", prefix),
                &[config.dim as u64],
                &norm_attn,
            );

            let norm_ffn = loop_block
                .norm_ffn
                .weight()
                .flatten_all()?
                .to_vec1::<f32>()?;
            writer.add_f32_tensor(
                &format!("{}.norm_ffn.weight", prefix),
                &[config.dim as u64],
                &norm_ffn,
            );
        }

        for (i, block) in model.local_blocks_after.iter().enumerate() {
            export_block(&mut writer, &format!("local_after.{}", i), block)?;
        }
    } else {
        // Standard architecture
        for (i, block) in model.blocks.iter().enumerate() {
            export_block(&mut writer, &format!("blocks.{}", i), block)?;
        }
    }

    // Final norm
    let norm_final = model.norm_final.weight().flatten_all()?.to_vec1::<f32>()?;
    writer.add_f32_tensor("norm_final.weight", &[config.dim as u64], &norm_final);

    // LM head (if not weight-tied)
    if let Some(ref lm_head) = model.lm_head_weight {
        let lm_data = lm_head.flatten_all()?.to_vec1::<f32>()?;
        writer.add_f32_tensor(
            "lm_head.weight",
            &[config.vocab_size as u64, config.dim as u64],
            &lm_data,
        );
    }

    writer
        .write(path)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF write error: {}", e)))?;

    Ok(())
}

/// Export mHC parameters to binary file.
pub fn export_mhc(model: &NanochatTrainModel, config: &TrainConfig, path: &str) -> Result<()> {
    use mhc_lite::io::{save_mhc_file, MhcLayerParams};

    let mut layers = Vec::new();

    // Export mHC based on architecture
    if config.loop_config.is_some() {
        // LoopLM architecture
        for block in &model.local_blocks_before {
            let attn_mhc = block.mhc_attn.to_inference_values()?;
            let ffn_mhc = block.mhc_ffn.to_inference_values()?;
            layers.push(MhcLayerParams::N2(attn_mhc));
            layers.push(MhcLayerParams::N2(ffn_mhc));
        }

        if let Some(ref loop_block) = model.shared_loop_block {
            let attn_mhc = loop_block.mhc_attn.to_inference_values()?;
            let ffn_mhc = loop_block.mhc_ffn.to_inference_values()?;
            layers.push(MhcLayerParams::N2(attn_mhc));
            layers.push(MhcLayerParams::N2(ffn_mhc));
        }

        for block in &model.local_blocks_after {
            let attn_mhc = block.mhc_attn.to_inference_values()?;
            let ffn_mhc = block.mhc_ffn.to_inference_values()?;
            layers.push(MhcLayerParams::N2(attn_mhc));
            layers.push(MhcLayerParams::N2(ffn_mhc));
        }
    } else {
        // Standard architecture
        for block in &model.blocks {
            let attn_mhc = block.mhc_attn.to_inference_values()?;
            let ffn_mhc = block.mhc_ffn.to_inference_values()?;
            layers.push(MhcLayerParams::N2(attn_mhc));
            layers.push(MhcLayerParams::N2(ffn_mhc));
        }
    }

    save_mhc_file(path, config.mhc_n_streams as u32, &layers)
        .map_err(|e| candle_core::Error::Msg(format!("mHC save error: {}", e)))?;

    Ok(())
}

/// Full export: GGUF + mHC.
pub fn export_model(
    model: &NanochatTrainModel,
    config: &TrainConfig,
    gguf_path: &str,
    mhc_path: &str,
) -> Result<()> {
    export_gguf(model, config, gguf_path)?;
    export_mhc(model, config, mhc_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_config() -> TrainConfig {
        crate::config::TrainConfig {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.0,
            vocab_size: 256,
            max_seq_len: 32,
            group_size: 64,
            mhc_n_streams: 2,
            weight_tied: true,
            rope_theta: 10000.0,
            loop_config: None,
            lr: 0.02,
            mhc_lr: 1e-4,
            weight_decay: 0.0,
            batch_size: 2,
            grad_accum_steps: 1,
            warmup_steps: 10,
            total_steps: 100,
            decay_start_frac: 0.8,
            grad_clip: 1.0,
            ns_steps: 3,
            muon_momentum: 0.95,
            lion_betas: (0.9, 0.99),
            use_8bit_optim: false,
            use_galore: false,
            galore_rank: 256,
            galore_update_freq: 200,
            use_mtp: false,
            mtp_n_tokens: 3,
            mtp_weight: 0.2,
            use_collider: false,
            collider_threshold: 0.3,
            collider_sparsity: 0.35,
            use_async_loader: false,
            async_n_workers: 4,
            async_prefetch_size: 8,
            label_smooth_eps: 0.1,
            entropy_weight: 0.0,
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
            use_wave_field: false,
            wavefield_field_size: 1024,
            wavefield_n_heads: 0,
            wavefield_head_coupling: true,
            wavefield_ratio: 1.0,
            wavefield_convolve_mode: None,
            wavefield_haar_levels: None,
        }
    }

    #[test]
    fn test_export_gguf_creates_file() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("test.gguf");
        export_gguf(&model, &cfg, gguf_path.to_str().unwrap())?;
        assert!(gguf_path.exists());
        assert!(std::fs::metadata(&gguf_path).unwrap().len() > 0);
        Ok(())
    }

    #[test]
    fn test_export_mhc_creates_file() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let dir = tempfile::tempdir().unwrap();
        let mhc_path = dir.path().join("test.mhc");
        export_mhc(&model, &cfg, mhc_path.to_str().unwrap())?;
        assert!(mhc_path.exists());
        assert!(std::fs::metadata(&mhc_path).unwrap().len() > 0);
        Ok(())
    }

    #[test]
    fn test_export_full_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let model = NanochatTrainModel::new(&cfg, vb)?;

        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("test.gguf");
        let mhc_path = dir.path().join("test.mhc");

        export_model(
            &model,
            &cfg,
            gguf_path.to_str().unwrap(),
            mhc_path.to_str().unwrap(),
        )?;

        assert!(gguf_path.exists());
        assert!(mhc_path.exists());
        Ok(())
    }
}
