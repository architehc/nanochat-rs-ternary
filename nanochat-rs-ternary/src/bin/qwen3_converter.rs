//! Hybrid Qwen3-Coder to ternary converter
//!
//! Selective ternarization strategy:
//! - Ternarize: MoE expert MLPs (w_gate, w_up, w_down) for all 512 experts
//! - Keep FP8/BF16: router/gates, norms, embeddings, lm_head, DeltaNet state/gates, attention projections
//!
//! This maximizes memory savings (experts are 99% of params) while minimizing quality loss.

use anyhow::{Context, Result};
use clap::Parser;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use ternary_core::pack::quantize_row_q1_58;
use ternary_core::gguf::{GgufFileWriter, GgufMetadata, GgufTensorInfo, GgufType, GgufValue};

#[derive(Parser, Debug)]
#[command(name = "qwen3-converter")]
#[command(about = "Convert Qwen3-Coder checkpoint to hybrid ternary GGUF")]
struct Args {
    /// Path to Qwen3-Coder checkpoint directory (contains model.safetensors.index.json or model.safetensors)
    #[arg(short, long)]
    input: PathBuf,

    /// Output GGUF file path
    #[arg(short, long)]
    output: PathBuf,

    /// Ternarization group size (default: 128)
    #[arg(short, long, default_value_t = 128)]
    group_size: usize,

    /// Keep attention projections in FP8 (default: true, only ternarize MoE experts)
    #[arg(long, default_value_t = true)]
    keep_attn_fp8: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

/// Tensor metadata from safetensors
#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

/// Classification of tensor for selective quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorClass {
    /// Ternarize this tensor (MoE expert weights)
    Ternary,
    /// Keep in original precision (router, norms, embeddings, etc.)
    KeepPrecision,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("Qwen3-Coder Hybrid Ternary Converter");
        println!("=====================================");
        println!("Input: {}", args.input.display());
        println!("Output: {}", args.output.display());
        println!("Group size: {}", args.group_size);
        println!("Keep attention FP8: {}", args.keep_attn_fp8);
        println!();
    }

    // 1. Load safetensors checkpoint
    let tensors = load_checkpoint(&args.input, args.verbose)?;

    if args.verbose {
        println!("Loaded {} tensors", tensors.len());

        // Count by category
        let mut ternary_count = 0;
        let mut keep_count = 0;
        for tensor in &tensors {
            match classify_tensor(&tensor.name, args.keep_attn_fp8) {
                TensorClass::Ternary => ternary_count += 1,
                TensorClass::KeepPrecision => keep_count += 1,
            }
        }
        println!("  - Ternarizing: {} tensors", ternary_count);
        println!("  - Keeping FP8/BF16: {} tensors", keep_count);
        println!();
    }

    // 2. Extract model config
    let config = extract_model_config(&args.input)?;

    if args.verbose {
        println!("Model config:");
        println!("  - dim: {}", config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0));
        println!("  - n_layers: {}", config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0));
        println!("  - vocab_size: {}", config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0));
        println!();
    }

    // 3. Convert tensors (selective ternarization)
    let converted = convert_tensors(&tensors, args.group_size, args.keep_attn_fp8, args.verbose)?;

    if args.verbose {
        println!("Conversion complete!");
        println!();
    }

    // 4. Export to hybrid GGUF
    export_gguf(&args.output, &config, &converted, args.verbose)?;

    if args.verbose {
        println!("Exported to: {}", args.output.display());
    }

    Ok(())
}

/// Load all tensors from safetensors checkpoint
fn load_checkpoint(path: &Path, verbose: bool) -> Result<Vec<TensorMeta>> {
    if verbose {
        println!("Loading checkpoint from {}...", path.display());
    }

    // Check if it's a single file or sharded
    let model_file = if path.is_file() {
        path.to_path_buf()
    } else {
        // Look for model.safetensors or model-00001-of-NNNNN.safetensors
        let single = path.join("model.safetensors");
        if single.exists() {
            single
        } else {
            // TODO: Handle sharded safetensors (model-00001-of-00008.safetensors, etc.)
            anyhow::bail!("Sharded safetensors not yet supported. Please provide single model.safetensors file.");
        }
    };

    // Load safetensors file
    let mut file = File::open(&model_file)
        .with_context(|| format!("Failed to open {}", model_file.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .context("Failed to read safetensors file")?;

    let safetensors = SafeTensors::deserialize(&buffer)
        .context("Failed to parse safetensors")?;

    let mut tensors = Vec::new();

    for (name, tensor_view) in safetensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = format!("{:?}", tensor_view.dtype());
        let data = tensor_view.data().to_vec();

        tensors.push(TensorMeta {
            name: name.to_string(),
            shape,
            dtype,
            data,
        });
    }

    Ok(tensors)
}

/// Classify tensor for selective quantization based on name
fn classify_tensor(name: &str, keep_attn_fp8: bool) -> TensorClass {
    // MoE expert weights - always ternarize (biggest memory win)
    if name.contains(".experts.") && (name.ends_with(".w_gate.weight")
        || name.ends_with(".w_up.weight")
        || name.ends_with(".w_down.weight")) {
        return TensorClass::Ternary;
    }

    // If keep_attn_fp8 is false, ternarize attention projections too
    if !keep_attn_fp8 && (name.ends_with(".wq.weight")
        || name.ends_with(".wk.weight")
        || name.ends_with(".wv.weight")
        || name.ends_with(".wo.weight")) {
        return TensorClass::Ternary;
    }

    // Everything else stays in original precision:
    // - Embeddings (tok_embed, lm_head)
    // - Norms (RMSNorm weights)
    // - Router/gates
    // - DeltaNet state matrices and gates
    // - Shared expert (if keeping attn FP8)
    TensorClass::KeepPrecision
}

/// Extract model config from config.json
fn extract_model_config(checkpoint_dir: &Path) -> Result<serde_json::Value> {
    let config_path = if checkpoint_dir.is_file() {
        checkpoint_dir.parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid checkpoint path"))?
            .join("config.json")
    } else {
        checkpoint_dir.join("config.json")
    };

    let file = File::open(&config_path)
        .with_context(|| format!("Failed to open config.json at {}", config_path.display()))?;

    let reader = BufReader::new(file);
    let config: serde_json::Value = serde_json::from_reader(reader)
        .context("Failed to parse config.json")?;

    Ok(config)
}

/// Converted tensor with quantization info
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    gguf_type: GgufType,
    data: Vec<u8>,
}

/// Convert tensors with selective ternarization
fn convert_tensors(
    tensors: &[TensorMeta],
    group_size: usize,
    keep_attn_fp8: bool,
    verbose: bool,
) -> Result<Vec<ConvertedTensor>> {
    let mut converted = Vec::new();

    for (idx, tensor) in tensors.iter().enumerate() {
        if verbose && idx % 100 == 0 {
            println!("Processing tensor {}/{}...", idx + 1, tensors.len());
        }

        let class = classify_tensor(&tensor.name, keep_attn_fp8);

        match class {
            TensorClass::Ternary => {
                // Ternarize this tensor
                let converted_tensor = ternarize_tensor(tensor, group_size)?;
                converted.push(converted_tensor);
            }
            TensorClass::KeepPrecision => {
                // Keep in original precision (FP8/BF16/FP32)
                let gguf_type = match tensor.dtype.as_str() {
                    "F32" => GgufType::F32,
                    "F16" => GgufType::F16,
                    "BF16" => GgufType::BF16,
                    _ => {
                        if verbose {
                            println!("Warning: Unknown dtype {} for {}, treating as F32", tensor.dtype, tensor.name);
                        }
                        GgufType::F32
                    }
                };

                converted.push(ConvertedTensor {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    gguf_type,
                    data: tensor.data.clone(),
                });
            }
        }
    }

    Ok(converted)
}

/// Ternarize a single tensor using absmean quantization
fn ternarize_tensor(tensor: &TensorMeta, group_size: usize) -> Result<ConvertedTensor> {
    // Convert tensor data to f32 (assuming it's already f32 or needs conversion)
    let floats = parse_tensor_data(tensor)?;

    // For 2D weight matrices: [out_features, in_features]
    if tensor.shape.len() != 2 {
        anyhow::bail!("Can only ternarize 2D tensors, got shape {:?} for {}", tensor.shape, tensor.name);
    }

    let rows = tensor.shape[0];
    let cols = tensor.shape[1];

    // Quantize using Q1_58 format (ternary with per-group scales)
    let quantized = quantize_row_q1_58(&floats, rows, cols, group_size)
        .map_err(|e| anyhow::anyhow!("Quantization failed: {}", e))?;

    Ok(ConvertedTensor {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        gguf_type: GgufType::Q1_58,
        data: quantized,
    })
}

/// Parse tensor data bytes into f32 vec
fn parse_tensor_data(tensor: &TensorMeta) -> Result<Vec<f32>> {
    match tensor.dtype.as_str() {
        "F32" => {
            // Already F32, just reinterpret bytes
            let floats: Vec<f32> = tensor.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(floats)
        }
        "F16" => {
            // Convert FP16 to F32
            use half::f16;
            let floats: Vec<f32> = tensor.data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        "BF16" => {
            // Convert BF16 to F32
            use half::bf16;
            let floats: Vec<f32> = tensor.data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    bf16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        _ => anyhow::bail!("Unsupported dtype: {}", tensor.dtype),
    }
}

/// Export converted tensors to GGUF format
fn export_gguf(
    output_path: &Path,
    config: &serde_json::Value,
    tensors: &[ConvertedTensor],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Exporting to GGUF...");
    }

    // Build GGUF metadata from config
    let mut metadata = HashMap::new();

    // Required fields
    if let Some(dim) = config.get("hidden_size").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.dim".to_string(), GgufValue::U32(dim as u32));
    }
    if let Some(n_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.n_layers".to_string(), GgufValue::U32(n_layers as u32));
    }
    if let Some(n_heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.n_heads".to_string(), GgufValue::U32(n_heads as u32));
    }
    if let Some(n_kv_heads) = config.get("num_key_value_heads").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.n_kv_heads".to_string(), GgufValue::U32(n_kv_heads as u32));
    }
    if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.vocab_size".to_string(), GgufValue::U32(vocab_size as u32));
    }

    // Optional Qwen3-specific fields
    metadata.insert("nanochat.group_size".to_string(), GgufValue::U32(128));
    metadata.insert("nanochat.mhc_n_streams".to_string(), GgufValue::U32(4));
    metadata.insert("nanochat.gated_attention".to_string(), GgufValue::Bool(true));

    // MoE config
    if let Some(n_experts) = config.get("num_experts").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.n_experts".to_string(), GgufValue::U32(n_experts as u32));
    }
    if let Some(n_active) = config.get("num_experts_per_tok").and_then(|v| v.as_u64()) {
        metadata.insert("nanochat.n_active_experts".to_string(), GgufValue::U32(n_active as u32));
    }
    metadata.insert("nanochat.use_shared_expert".to_string(), GgufValue::Bool(true));

    // Build tensor list
    let tensor_infos: Vec<GgufTensorInfo> = tensors
        .iter()
        .map(|t| GgufTensorInfo {
            name: t.name.clone(),
            dims: t.shape.iter().map(|&d| d as u64).collect(),
            dtype: t.gguf_type,
            offset: 0, // Will be filled by writer
        })
        .collect();

    // Write GGUF file
    let gguf_meta = GgufMetadata {
        metadata,
        tensors: tensor_infos,
    };

    let tensor_data: Vec<&[u8]> = tensors.iter().map(|t| t.data.as_slice()).collect();

    GgufFileWriter::write(output_path, &gguf_meta, &tensor_data)?;

    if verbose {
        println!("Successfully exported {} tensors to GGUF", tensors.len());
    }

    Ok(())
}
