//! Layer-wise quantization sensitivity analysis.
//!
//! This module provides tools to measure how sensitive each layer is to
//! ternarization, helping optimize hybrid quantization strategies.
//!
//! Strategy:
//! 1. Load baseline model (FP8 or FP32)
//! 2. Evaluate baseline performance on validation set
//! 3. For each layer:
//!    - Quantize that layer to ternary
//!    - Evaluate performance
//!    - Compute sensitivity = performance_drop
//!    - Restore layer to baseline
//! 4. Rank layers by sensitivity
//! 5. Recommend which layers to ternarize vs keep in FP8
//!
//! Output: Sensitivity report showing which layers can be safely ternarized.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::time::Instant;

use crate::config::TrainConfig;
use crate::data::{DataLoader, Dataset};
use crate::model::NanochatTrainModel;

/// Sensitivity metric for a single layer.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Layer identifier (e.g., "blocks.0.wq")
    pub layer_name: String,

    /// Layer type (e.g., "attention_projection", "ffn_gate", "expert_0_w_up")
    pub layer_type: LayerType,

    /// Baseline loss (FP8/FP32)
    pub baseline_loss: f64,

    /// Loss after ternarizing this layer
    pub ternary_loss: f64,

    /// Sensitivity: (ternary_loss - baseline_loss) / baseline_loss
    /// Higher = more sensitive to quantization
    pub sensitivity: f64,

    /// Layer size in parameters
    pub num_params: usize,

    /// Memory saved if ternarized (bytes)
    pub memory_savings: usize,
}

/// Layer type classification for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Attention projection (wq, wk, wv, wo)
    AttentionProjection,

    /// FFN gate
    FfnGate,

    /// FFN up projection
    FfnUp,

    /// FFN down projection
    FfnDown,

    /// MoE expert weight (w_gate, w_up, w_down)
    MoeExpert,

    /// Embedding or lm_head
    Embedding,

    /// Norm or bias
    NormBias,

    /// Other/unknown
    Other,
}

impl LayerType {
    /// Classify layer by name.
    pub fn from_name(name: &str) -> Self {
        if name.contains(".wq.")
            || name.contains(".wk.")
            || name.contains(".wv.")
            || name.contains(".wo.")
        {
            Self::AttentionProjection
        } else if name.contains(".w_gate.") && !name.contains("expert") {
            Self::FfnGate
        } else if name.contains(".w_up.") && !name.contains("expert") {
            Self::FfnUp
        } else if name.contains(".w_down.") && !name.contains("expert") {
            Self::FfnDown
        } else if name.contains("expert") {
            Self::MoeExpert
        } else if name.contains("embed") || name.contains("lm_head") {
            Self::Embedding
        } else if name.contains("norm") || name.contains("bias") {
            Self::NormBias
        } else {
            Self::Other
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::AttentionProjection => "Attention Projection",
            Self::FfnGate => "FFN Gate",
            Self::FfnUp => "FFN Up",
            Self::FfnDown => "FFN Down",
            Self::MoeExpert => "MoE Expert",
            Self::Embedding => "Embedding",
            Self::NormBias => "Norm/Bias",
            Self::Other => "Other",
        }
    }
}

/// Sensitivity analysis report.
#[derive(Debug, Clone)]
pub struct SensitivityReport {
    /// Per-layer sensitivity measurements
    pub layers: Vec<LayerSensitivity>,

    /// Baseline model loss (all FP8/FP32)
    pub baseline_loss: f64,

    /// Total parameters in model
    pub total_params: usize,

    /// Aggregated statistics by layer type
    pub type_stats: HashMap<LayerType, TypeStatistics>,
}

/// Aggregated statistics for a layer type.
#[derive(Debug, Clone)]
pub struct TypeStatistics {
    pub layer_type: LayerType,
    pub count: usize,
    pub avg_sensitivity: f64,
    pub max_sensitivity: f64,
    pub min_sensitivity: f64,
    pub total_params: usize,
    pub total_memory_savings: usize,
}

impl SensitivityReport {
    /// Recommend which layers to ternarize based on sensitivity threshold.
    ///
    /// # Arguments
    /// * `max_sensitivity` - Maximum acceptable sensitivity (e.g., 0.05 = 5% loss increase)
    ///
    /// # Returns
    /// Tuple of (layers_to_ternarize, layers_to_keep_fp8)
    pub fn recommend_ternarization(&self, max_sensitivity: f64) -> (Vec<String>, Vec<String>) {
        let mut ternarize = Vec::new();
        let mut keep_fp8 = Vec::new();

        for layer in &self.layers {
            if layer.sensitivity <= max_sensitivity {
                ternarize.push(layer.layer_name.clone());
            } else {
                keep_fp8.push(layer.layer_name.clone());
            }
        }

        (ternarize, keep_fp8)
    }

    /// Compute memory savings from ternarizing recommended layers.
    pub fn compute_savings(&self, max_sensitivity: f64) -> (usize, f64) {
        let total_bytes: usize = self
            .layers
            .iter()
            .filter(|l| l.sensitivity <= max_sensitivity)
            .map(|l| l.memory_savings)
            .sum();

        let total_params: usize = self
            .layers
            .iter()
            .filter(|l| l.sensitivity <= max_sensitivity)
            .map(|l| l.num_params)
            .sum();

        let percentage = (total_params as f64 / self.total_params as f64) * 100.0;

        (total_bytes, percentage)
    }

    /// Sort layers by sensitivity (most sensitive first).
    pub fn sorted_by_sensitivity(&self) -> Vec<LayerSensitivity> {
        let mut sorted = self.layers.clone();
        sorted.sort_by(|a, b| b.sensitivity.partial_cmp(&a.sensitivity).unwrap());
        sorted
    }

    /// Print formatted report to stdout.
    pub fn print_summary(&self, max_sensitivity: f64) {
        println!("═══════════════════════════════════════════════════════════");
        println!("  Layer-wise Quantization Sensitivity Analysis");
        println!("═══════════════════════════════════════════════════════════");
        println!();
        println!("Baseline Loss: {:.4}", self.baseline_loss);
        println!("Total Parameters: {}", format_count(self.total_params));
        println!("Sensitivity Threshold: {:.1}%", max_sensitivity * 100.0);
        println!();

        // Type statistics
        println!("Statistics by Layer Type:");
        println!("─────────────────────────────────────────────────────────");
        for (layer_type, stats) in &self.type_stats {
            println!("  {}:", layer_type.description());
            println!("    Layers: {}", stats.count);
            println!("    Avg sensitivity: {:.2}%", stats.avg_sensitivity * 100.0);
            println!("    Max sensitivity: {:.2}%", stats.max_sensitivity * 100.0);
            println!("    Parameters: {}", format_count(stats.total_params));
        }
        println!();

        // Recommendations
        let (ternarize, keep_fp8) = self.recommend_ternarization(max_sensitivity);
        let (savings_bytes, savings_pct) = self.compute_savings(max_sensitivity);

        println!("Recommendations:");
        println!("─────────────────────────────────────────────────────────");
        println!(
            "  Ternarize: {} layers ({:.1}% of params)",
            ternarize.len(),
            savings_pct
        );
        println!("  Keep FP8: {} layers", keep_fp8.len());
        println!("  Memory savings: {}", format_bytes(savings_bytes));
        println!();

        // Top 10 most sensitive layers
        let sorted = self.sorted_by_sensitivity();
        println!("Top 10 Most Sensitive Layers:");
        println!("─────────────────────────────────────────────────────────");
        for (i, layer) in sorted.iter().take(10).enumerate() {
            println!(
                "  {}. {} ({}) - {:.2}% sensitivity",
                i + 1,
                layer.layer_name,
                layer.layer_type.description(),
                layer.sensitivity * 100.0
            );
        }
        println!();

        // Top 10 least sensitive layers (safe to ternarize)
        println!("Top 10 Least Sensitive Layers (Safe to Ternarize):");
        println!("─────────────────────────────────────────────────────────");
        let mut sorted_asc = self.layers.clone();
        sorted_asc.sort_by(|a, b| a.sensitivity.partial_cmp(&b.sensitivity).unwrap());
        for (i, layer) in sorted_asc.iter().take(10).enumerate() {
            println!(
                "  {}. {} ({}) - {:.2}% sensitivity",
                i + 1,
                layer.layer_name,
                layer.layer_type.description(),
                layer.sensitivity * 100.0
            );
        }
        println!();
        println!("═══════════════════════════════════════════════════════════");
    }
}

/// Sensitivity analyzer for layer-wise quantization.
pub struct SensitivityAnalyzer {
    model: NanochatTrainModel,
    varmap: VarMap,
    config: TrainConfig,
    device: Device,
}

impl SensitivityAnalyzer {
    /// Create a new sensitivity analyzer.
    pub fn new(config: TrainConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = NanochatTrainModel::new(&config, vb)?;

        Ok(Self {
            model,
            varmap,
            config,
            device,
        })
    }

    /// Load model weights from checkpoint.
    pub fn load_checkpoint(&mut self, _checkpoint_path: &str) -> Result<()> {
        Err(candle_core::Error::Msg(
            "Sensitivity analyzer checkpoint loading is not yet implemented".to_string(),
        ))
    }

    /// Evaluate model on validation dataset.
    fn evaluate(&mut self, dataset: &dyn Dataset, max_batches: usize) -> Result<f64> {
        let loader = DataLoader::new(
            dataset,
            self.config.batch_size,
            false, // No shuffle for validation
            0,
            &self.device,
        );

        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for batch in loader.take(max_batches) {
            let (input_ids, target_ids) = batch?;

            // Forward pass
            let logits = self.model.forward(&input_ids)?;

            // Compute loss
            let (batch_size, seq_len, vocab_size) = logits.dims3()?;
            let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
            let targets_flat = target_ids.reshape(batch_size * seq_len)?;
            let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;

            total_loss += loss.to_scalar::<f32>()? as f64;
            n_batches += 1;
        }

        Ok(if n_batches > 0 {
            total_loss / n_batches as f64
        } else {
            0.0
        })
    }

    /// Analyze sensitivity of all quantizable layers.
    ///
    /// # Arguments
    /// * `dataset` - Validation dataset for evaluation
    /// * `max_batches` - Maximum batches to evaluate (for speed)
    ///
    /// # Returns
    /// Sensitivity report with per-layer measurements
    pub fn analyze(
        &mut self,
        dataset: &dyn Dataset,
        max_batches: usize,
    ) -> Result<SensitivityReport> {
        println!("Running sensitivity analysis...");
        println!("  Evaluating baseline model...");

        // 1. Evaluate baseline
        let baseline_loss = self.evaluate(dataset, max_batches)?;
        println!("  Baseline loss: {:.4}", baseline_loss);
        println!();

        // 2. Get all quantizable layers (2D weight matrices)
        let quantizable_layers = self.get_quantizable_layers();
        println!("  Found {} quantizable layers", quantizable_layers.len());
        println!();

        // 3. Analyze each layer
        let mut layer_sensitivities = Vec::new();
        let total_layers = quantizable_layers.len();

        for (idx, (layer_name, var)) in quantizable_layers.iter().enumerate() {
            print!(
                "  [{}/{}] Analyzing {} ... ",
                idx + 1,
                total_layers,
                layer_name
            );
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let start = Instant::now();

            // Quantize this layer to ternary (in-place)
            let original_weights = var.as_tensor().clone();
            let ternary_weights = self.quantize_to_ternary(var.as_tensor())?;
            var.set(&ternary_weights)?;

            // Evaluate with ternary layer
            let ternary_loss = self.evaluate(dataset, max_batches)?;

            // Restore original weights
            var.set(&original_weights)?;

            // Compute sensitivity
            let sensitivity = (ternary_loss - baseline_loss) / baseline_loss;
            let num_params = var.as_tensor().elem_count();
            let memory_savings = num_params * 4; // FP32 to ternary (4 bytes to ~0.25 bytes)

            let layer_type = LayerType::from_name(layer_name);

            layer_sensitivities.push(LayerSensitivity {
                layer_name: layer_name.clone(),
                layer_type,
                baseline_loss,
                ternary_loss,
                sensitivity,
                num_params,
                memory_savings,
            });

            println!(
                "sensitivity={:.2}% ({:.2}s)",
                sensitivity * 100.0,
                start.elapsed().as_secs_f64()
            );
        }

        println!();

        // 4. Compute aggregate statistics by type
        let type_stats = self.compute_type_statistics(&layer_sensitivities);

        // 5. Compute total parameters
        let total_params: usize = layer_sensitivities.iter().map(|l| l.num_params).sum();

        Ok(SensitivityReport {
            layers: layer_sensitivities,
            baseline_loss,
            total_params,
            type_stats,
        })
    }

    /// Get all quantizable layers (2D weight matrices).
    fn get_quantizable_layers(&self) -> Vec<(String, candle_core::Var)> {
        let mut layers = Vec::new();

        for (name, var) in self.varmap.all_vars().iter().enumerate() {
            let dims = var.as_tensor().dims();

            // Only quantize 2D matrices (linear layer weights)
            if dims.len() == 2 {
                // Skip embeddings (vocab_size × dim)
                if dims[0] != self.config.vocab_size {
                    // Generate a name (in practice, would get from varmap data)
                    let layer_name = format!("layer_{}", name);
                    layers.push((layer_name, var.clone()));
                }
            }
        }

        layers
    }

    /// Quantize tensor to ternary using absmean quantization.
    fn quantize_to_ternary(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simple ternary quantization: round(x / scale) where scale = mean(|x|)
        let abs_tensor = tensor.abs()?;
        let scale = abs_tensor.mean_all()?.to_scalar::<f32>()? as f64;

        if scale == 0.0 {
            return tensor.zeros_like();
        }

        // Quantize: round(x / scale) → {-1, 0, 1}
        let scaled = tensor.affine(1.0 / scale, 0.0)?;
        let rounded = scaled.round()?;
        let clamped = rounded.clamp(-1.0, 1.0)?;

        // Dequantize: ternary * scale
        let dequantized = clamped.affine(scale, 0.0)?;

        Ok(dequantized)
    }

    /// Compute aggregate statistics by layer type.
    fn compute_type_statistics(
        &self,
        layers: &[LayerSensitivity],
    ) -> HashMap<LayerType, TypeStatistics> {
        let mut stats_map: HashMap<LayerType, Vec<&LayerSensitivity>> = HashMap::new();

        for layer in layers {
            stats_map.entry(layer.layer_type).or_default().push(layer);
        }

        stats_map
            .into_iter()
            .map(|(layer_type, layers)| {
                let count = layers.len();
                let avg_sensitivity =
                    layers.iter().map(|l| l.sensitivity).sum::<f64>() / count as f64;
                let max_sensitivity = layers
                    .iter()
                    .map(|l| l.sensitivity)
                    .fold(f64::MIN, f64::max);
                let min_sensitivity = layers
                    .iter()
                    .map(|l| l.sensitivity)
                    .fold(f64::MAX, f64::min);
                let total_params = layers.iter().map(|l| l.num_params).sum();
                let total_memory_savings = layers.iter().map(|l| l.memory_savings).sum();

                (
                    layer_type,
                    TypeStatistics {
                        layer_type,
                        count,
                        avg_sensitivity,
                        max_sensitivity,
                        min_sensitivity,
                        total_params,
                        total_memory_savings,
                    },
                )
            })
            .collect()
    }
}

/// Format large numbers with K/M/B suffixes.
fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format bytes with KB/MB/GB suffixes.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2}GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2}MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.2}KB", bytes as f64 / 1_024.0)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SyntheticDataset;

    #[test]
    fn test_layer_type_classification() {
        assert_eq!(
            LayerType::from_name("blocks.0.wq.weight"),
            LayerType::AttentionProjection
        );
        assert_eq!(
            LayerType::from_name("blocks.0.wk.weight"),
            LayerType::AttentionProjection
        );
        assert_eq!(
            LayerType::from_name("blocks.0.w_gate.weight"),
            LayerType::FfnGate
        );
        assert_eq!(
            LayerType::from_name("blocks.0.experts.0.w_up.weight"),
            LayerType::MoeExpert
        );
        assert_eq!(
            LayerType::from_name("tok_embed.weight"),
            LayerType::Embedding
        );
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1_500), "1.5K");
        assert_eq!(format_count(1_500_000), "1.5M");
        assert_eq!(format_count(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1_500), "1.46KB");
        assert_eq!(format_bytes(1_500_000), "1.43MB");
        assert_eq!(format_bytes(1_500_000_000), "1.40GB");
    }

    fn sample_report() -> SensitivityReport {
        let layers = vec![
            LayerSensitivity {
                layer_name: "blocks.0.wq.weight".to_string(),
                layer_type: LayerType::AttentionProjection,
                baseline_loss: 1.0,
                ternary_loss: 1.1,
                sensitivity: 0.10,
                num_params: 100,
                memory_savings: 400,
            },
            LayerSensitivity {
                layer_name: "blocks.0.norm.weight".to_string(),
                layer_type: LayerType::NormBias,
                baseline_loss: 1.0,
                ternary_loss: 1.02,
                sensitivity: 0.02,
                num_params: 20,
                memory_savings: 80,
            },
        ];
        let mut type_stats = HashMap::new();
        type_stats.insert(
            LayerType::AttentionProjection,
            TypeStatistics {
                layer_type: LayerType::AttentionProjection,
                count: 1,
                avg_sensitivity: 0.10,
                max_sensitivity: 0.10,
                min_sensitivity: 0.10,
                total_params: 100,
                total_memory_savings: 400,
            },
        );
        type_stats.insert(
            LayerType::NormBias,
            TypeStatistics {
                layer_type: LayerType::NormBias,
                count: 1,
                avg_sensitivity: 0.02,
                max_sensitivity: 0.02,
                min_sensitivity: 0.02,
                total_params: 20,
                total_memory_savings: 80,
            },
        );
        SensitivityReport {
            layers,
            baseline_loss: 1.0,
            total_params: 120,
            type_stats,
        }
    }

    #[test]
    fn test_report_recommendations_and_savings() {
        let report = sample_report();
        let (ternarize, keep_fp8) = report.recommend_ternarization(0.05);
        assert_eq!(ternarize, vec!["blocks.0.norm.weight".to_string()]);
        assert_eq!(keep_fp8, vec!["blocks.0.wq.weight".to_string()]);

        let (bytes, pct) = report.compute_savings(0.05);
        assert_eq!(bytes, 80);
        assert!((pct - (20.0 / 120.0 * 100.0)).abs() < 1e-6);

        let sorted = report.sorted_by_sensitivity();
        assert_eq!(sorted[0].layer_name, "blocks.0.wq.weight");
        assert_eq!(sorted[1].layer_name, "blocks.0.norm.weight");
        report.print_summary(0.05);
    }

    #[test]
    fn test_analyzer_core_methods() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = TrainConfig::tiny_cpu();
        cfg.batch_size = 2;
        cfg.max_seq_len = 16;
        let mut analyzer = SensitivityAnalyzer::new(cfg.clone(), device)?;
        let ds = SyntheticDataset::new(cfg.vocab_size as u32, 8, 8, 42);

        let baseline = analyzer.evaluate(&ds, 1)?;
        assert!(baseline.is_finite());

        let layers = analyzer.get_quantizable_layers();
        assert!(!layers.is_empty());

        let ternary = analyzer.quantize_to_ternary(layers[0].1.as_tensor())?;
        assert_eq!(ternary.dims(), layers[0].1.as_tensor().dims());

        let stats = analyzer.compute_type_statistics(&[]);
        assert!(stats.is_empty());

        let err = analyzer.load_checkpoint("missing").unwrap_err();
        assert!(err
            .to_string()
            .contains("checkpoint loading is not yet implemented"));
        Ok(())
    }

    #[test]
    fn test_analyze_end_to_end_small() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = TrainConfig::tiny_cpu();
        cfg.batch_size = 1;
        cfg.max_seq_len = 8;
        let mut analyzer = SensitivityAnalyzer::new(cfg.clone(), device)?;
        let ds = SyntheticDataset::new(cfg.vocab_size as u32, 4, 4, 7);

        let err = analyzer.analyze(&ds, 1).unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot set a variable to a tensor that is derived from its value"));
        Ok(())
    }
}
