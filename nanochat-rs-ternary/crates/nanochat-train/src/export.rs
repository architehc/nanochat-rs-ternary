//! Export trained model to GGUF + mHC binary for inference.

use candle_core::Result;

use crate::config::TrainConfig;
use crate::model::NanochatTrainModel;

/// Export trained model weights to GGUF format.
///
/// Extracts ternary weights from BitLinearSTE layers, packs them using
/// ternary-core, and writes to a GGUF file. Embeddings are stored as FP16,
/// norms as FP32.
pub fn export_gguf(
    model: &NanochatTrainModel,
    config: &TrainConfig,
    path: &str,
) -> Result<()> {
    use ternary_core::gguf::{GgufWriter, GgufValue};
    use ternary_core::planar::PlanarWeights;

    let mut writer = GgufWriter::new();

    // Metadata
    writer.add_metadata("general.architecture", GgufValue::String("nanochat-ternary".to_string()));
    writer.add_metadata("nanochat.dim", GgufValue::U32(config.dim as u32));
    writer.add_metadata("nanochat.n_layers", GgufValue::U32(config.n_layers as u32));
    writer.add_metadata("nanochat.n_heads", GgufValue::U32(config.n_heads as u32));
    writer.add_metadata("nanochat.n_kv_heads", GgufValue::U32(config.n_kv_heads as u32));
    writer.add_metadata("nanochat.vocab_size", GgufValue::U32(config.vocab_size as u32));
    writer.add_metadata("nanochat.max_seq_len", GgufValue::U32(config.max_seq_len as u32));
    writer.add_metadata("nanochat.group_size", GgufValue::U32(config.group_size as u32));
    writer.add_metadata("nanochat.mhc_n_streams", GgufValue::U32(config.mhc_n_streams as u32));
    writer.add_metadata("nanochat.weight_tied", GgufValue::U32(config.weight_tied as u32));
    writer.add_metadata("nanochat.rope_theta", GgufValue::F32(config.rope_theta));
    writer.add_metadata("nanochat.ffn_mult", GgufValue::F32(config.ffn_mult));

    // Token embedding (FP32 -> stored as f32 in GGUF)
    let embed_data = model.tok_embed.embeddings().flatten_all()?.to_vec1::<f32>()?;
    writer.add_f32_tensor(
        "tok_embed.weight",
        &[config.vocab_size as u64, config.dim as u64],
        &embed_data,
    );

    // Per-layer weights
    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks.{}", i);

        // Attention ternary weights
        for (name, layer) in [
            ("attention.wq", &block.attention.wq),
            ("attention.wk", &block.attention.wk),
            ("attention.wv", &block.attention.wv),
            ("attention.wo", &block.attention.wo),
        ] {
            let (w_ternary, _scales) = layer.get_ternary_weights()?;
            let w_flat = w_ternary.flatten_all()?.to_vec1::<f32>()?;
            let rows = layer.out_features;
            let cols = layer.in_features;

            let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
            writer.add_ternary_tensor(
                &format!("{}.{}.weight", prefix, name),
                &pw,
            );
        }

        // FFN ternary weights
        for (name, layer) in [
            ("ffn.w_gate", &block.ffn.w_gate),
            ("ffn.w_up", &block.ffn.w_up),
            ("ffn.w_down", &block.ffn.w_down),
        ] {
            let (w_ternary, _scales) = layer.get_ternary_weights()?;
            let w_flat = w_ternary.flatten_all()?.to_vec1::<f32>()?;
            let rows = layer.out_features;
            let cols = layer.in_features;

            let pw = PlanarWeights::from_row_major(&w_flat, rows, cols, config.group_size);
            writer.add_ternary_tensor(
                &format!("{}.{}.weight", prefix, name),
                &pw,
            );
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

    writer.write(path)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF write error: {}", e)))?;

    Ok(())
}

/// Export mHC parameters to binary file.
pub fn export_mhc(
    model: &NanochatTrainModel,
    config: &TrainConfig,
    path: &str,
) -> Result<()> {
    use mhc_lite::io::{save_mhc_file, MhcLayerParams};

    let mut layers = Vec::new();
    for block in &model.blocks {
        let attn_mhc = block.mhc_attn.to_inference_values()?;
        let ffn_mhc = block.mhc_ffn.to_inference_values()?;
        layers.push(MhcLayerParams::N2(attn_mhc));
        layers.push(MhcLayerParams::N2(ffn_mhc));
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
    use candle_core::{Device, DType};
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

        export_model(&model, &cfg, gguf_path.to_str().unwrap(), mhc_path.to_str().unwrap())?;

        assert!(gguf_path.exists());
        assert!(mhc_path.exists());
        Ok(())
    }
}
