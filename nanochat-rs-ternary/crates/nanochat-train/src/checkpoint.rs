//! Training checkpoint save/load.

use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::config::TrainConfig;

const CHECKPOINT_VERSION_CURRENT: u32 = 1;

fn default_checkpoint_version() -> u32 {
    0 // Legacy checkpoints written before explicit versioning.
}

/// Metadata stored alongside model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    #[serde(default = "default_checkpoint_version")]
    pub version: u32,
    pub config: TrainConfig,
    pub step: usize,
    pub loss: f64,
}

/// Save training checkpoint to directory.
///
/// Creates:
///   - `<dir>/model.safetensors` — model weights
///   - `<dir>/meta.json` — training metadata
pub fn save_checkpoint(
    varmap: &VarMap,
    config: &TrainConfig,
    step: usize,
    loss: f64,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;

    // Save weights
    let weights_path = format!("{}/model.safetensors", dir);
    varmap.save(&weights_path)?;

    // Save metadata
    let meta = CheckpointMeta {
        version: CHECKPOINT_VERSION_CURRENT,
        config: config.clone(),
        step,
        loss,
    };
    let json = serde_json::to_string_pretty(&meta)?;
    std::fs::write(format!("{}/meta.json", dir), json)?;

    Ok(())
}

/// Load training checkpoint from directory.
///
/// Returns (VarMap, config, step) — VarMap can be used to reconstruct the model.
pub fn load_checkpoint(
    dir: &str,
    _device: &candle_core::Device,
) -> Result<(VarMap, TrainConfig, usize, f64), Box<dyn std::error::Error>> {
    // Load metadata
    let meta_json = std::fs::read_to_string(format!("{}/meta.json", dir))?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_json)?;
    if meta.version > CHECKPOINT_VERSION_CURRENT {
        return Err(format!(
            "unsupported checkpoint metadata version {} (max supported {})",
            meta.version, CHECKPOINT_VERSION_CURRENT
        )
        .into());
    }

    // Load weights
    let mut varmap = VarMap::new();
    let weights_path = format!("{}/model.safetensors", dir);
    varmap.load(&weights_path)?;

    Ok((varmap, meta.config, meta.step, meta.loss))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

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
            use_fp4: false,
            fp4_stochastic_rounding: true,
            distill_teacher: None,
            distill_kl_weight: 0.0,
            loop_scale_penalty: 0.0,
        }
    }

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let _model = crate::model::NanochatTrainModel::new(&cfg, vb).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();

        save_checkpoint(&varmap, &cfg, 42, std::f64::consts::PI, dir_path).unwrap();

        // Verify files exist
        assert!(std::path::Path::new(&format!("{}/model.safetensors", dir_path)).exists());
        assert!(std::path::Path::new(&format!("{}/meta.json", dir_path)).exists());
    }

    #[test]
    fn test_checkpoint_metadata_preserved() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = tiny_config();
        let _model = crate::model::NanochatTrainModel::new(&cfg, vb).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();

        save_checkpoint(&varmap, &cfg, 42, std::f64::consts::PI, dir_path).unwrap();

        // Load metadata
        let meta_json = std::fs::read_to_string(format!("{}/meta.json", dir_path)).unwrap();
        let meta: CheckpointMeta = serde_json::from_str(&meta_json).unwrap();

        assert_eq!(meta.step, 42);
        assert!((meta.loss - std::f64::consts::PI).abs() < 1e-10);
        assert_eq!(meta.config.dim, 64);
        assert_eq!(meta.config.n_layers, 2);
        assert_eq!(meta.version, CHECKPOINT_VERSION_CURRENT);
    }
}
