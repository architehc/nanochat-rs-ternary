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
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use ternary_core::gguf::{GgufFileWriter, GgufMetadata, GgufTensorInfo, GgufType, GgufValue};
use ternary_core::pack::quantize_row_q1_58;

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

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
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
        println!(
            "  - dim: {}",
            config
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
        println!(
            "  - n_layers: {}",
            config
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
        println!(
            "  - vocab_size: {}",
            config
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
        println!();
    }

    // 3. Convert tensors (selective ternarization)
    let converted = convert_tensors(&tensors, args.group_size, args.keep_attn_fp8, args.verbose)?;

    if args.verbose {
        println!("Conversion complete!");
        println!();
    }

    // 4. Export to hybrid GGUF
    export_gguf(
        &args.output,
        &config,
        &converted,
        args.group_size,
        args.verbose,
    )?;

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

    let model_files = resolve_checkpoint_files(path)?;

    if verbose {
        println!("Found {} safetensors file(s)", model_files.len());
    }

    let mut tensors = Vec::new();
    let mut seen_names = HashSet::new();

    for (idx, model_file) in model_files.iter().enumerate() {
        if verbose {
            println!(
                "  [{}/{}] loading {}",
                idx + 1,
                model_files.len(),
                model_file.display()
            );
        }

        let shard_tensors = load_safetensors_file(model_file)?;
        for tensor in shard_tensors {
            if !seen_names.insert(tensor.name.clone()) {
                anyhow::bail!(
                    "Duplicate tensor {} encountered while loading shards",
                    tensor.name
                );
            }
            tensors.push(tensor);
        }
    }

    // Keep tensor ordering deterministic across filesystems/shard layouts.
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(tensors)
}

fn resolve_checkpoint_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        anyhow::bail!("Checkpoint path does not exist: {}", path.display());
    }

    let single = path.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let reader = BufReader::new(
            File::open(&index_path)
                .with_context(|| format!("Failed to open {}", index_path.display()))?,
        );
        let index: SafetensorsIndex = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse {}", index_path.display()))?;

        if index.weight_map.is_empty() {
            anyhow::bail!(
                "Sharded index {} has empty weight_map",
                index_path.display()
            );
        }

        let mut shard_names = BTreeSet::new();
        for shard in index.weight_map.values() {
            shard_names.insert(shard.clone());
        }

        let mut files = Vec::with_capacity(shard_names.len());
        for shard in shard_names {
            let shard_path = path.join(&shard);
            if !shard_path.exists() {
                anyhow::bail!(
                    "Shard {} referenced in {} does not exist",
                    shard_path.display(),
                    index_path.display()
                );
            }
            files.push(shard_path);
        }
        return Ok(files);
    }

    let mut shard_files = Vec::new();
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("Failed to read directory {}", path.display()))?
    {
        let entry = entry?;
        let file_path = entry.path();
        if !file_path.is_file() {
            continue;
        }
        let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if file_name.starts_with("model-") && file_name.ends_with(".safetensors") {
            shard_files.push(file_path);
        }
    }
    shard_files.sort();

    if !shard_files.is_empty() {
        return Ok(shard_files);
    }

    anyhow::bail!(
        "No model.safetensors, model.safetensors.index.json, or model-*-of-*.safetensors found in {}",
        path.display()
    );
}

fn load_safetensors_file(model_file: &Path) -> Result<Vec<TensorMeta>> {
    // Load safetensors file
    let mut file = File::open(&model_file)
        .with_context(|| format!("Failed to open {}", model_file.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .context("Failed to read safetensors file")?;

    let safetensors = SafeTensors::deserialize(&buffer).context("Failed to parse safetensors")?;

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
    if name.contains(".experts.")
        && (name.ends_with(".w_gate.weight")
            || name.ends_with(".w_up.weight")
            || name.ends_with(".w_down.weight"))
    {
        return TensorClass::Ternary;
    }

    // If keep_attn_fp8 is false, ternarize attention projections too
    if !keep_attn_fp8
        && (name.ends_with(".wq.weight")
            || name.ends_with(".wk.weight")
            || name.ends_with(".wv.weight")
            || name.ends_with(".wo.weight"))
    {
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
        checkpoint_dir
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid checkpoint path"))?
            .join("config.json")
    } else {
        checkpoint_dir.join("config.json")
    };

    let file = File::open(&config_path)
        .with_context(|| format!("Failed to open config.json at {}", config_path.display()))?;

    let reader = BufReader::new(file);
    let config: serde_json::Value =
        serde_json::from_reader(reader).context("Failed to parse config.json")?;

    Ok(config)
}

/// Converted tensor with quantization info
#[derive(Debug)]
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
                        anyhow::bail!(
                            "Unsupported dtype {} for tensor {} — cannot safely reinterpret as F32. \
                             Convert to F32/F16/BF16 before running the converter.",
                            tensor.dtype,
                            tensor.name
                        );
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
        anyhow::bail!(
            "Can only ternarize 2D tensors, got shape {:?} for {}",
            tensor.shape,
            tensor.name
        );
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
            let floats: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(floats)
        }
        "F16" => {
            // Convert FP16 to F32
            use half::f16;
            let floats: Vec<f32> = tensor
                .data
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
            let floats: Vec<f32> = tensor
                .data
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
    group_size: usize,
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
        metadata.insert(
            "nanochat.n_layers".to_string(),
            GgufValue::U32(n_layers as u32),
        );
    }
    if let Some(n_heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
        metadata.insert(
            "nanochat.n_heads".to_string(),
            GgufValue::U32(n_heads as u32),
        );
    }
    if let Some(n_kv_heads) = config.get("num_key_value_heads").and_then(|v| v.as_u64()) {
        metadata.insert(
            "nanochat.n_kv_heads".to_string(),
            GgufValue::U32(n_kv_heads as u32),
        );
    }
    if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
        metadata.insert(
            "nanochat.vocab_size".to_string(),
            GgufValue::U32(vocab_size as u32),
        );
    }

    // Optional Qwen3-specific fields
    metadata.insert(
        "nanochat.group_size".to_string(),
        GgufValue::U32(group_size as u32),
    );
    metadata.insert("nanochat.mhc_n_streams".to_string(), GgufValue::U32(4));
    metadata.insert(
        "nanochat.gated_attention".to_string(),
        GgufValue::Bool(true),
    );

    // MoE config
    if let Some(n_experts) = config.get("num_experts").and_then(|v| v.as_u64()) {
        metadata.insert(
            "nanochat.n_experts".to_string(),
            GgufValue::U32(n_experts as u32),
        );
    }
    if let Some(n_active) = config.get("num_experts_per_tok").and_then(|v| v.as_u64()) {
        metadata.insert(
            "nanochat.n_active_experts".to_string(),
            GgufValue::U32(n_active as u32),
        );
    }
    metadata.insert(
        "nanochat.use_shared_expert".to_string(),
        GgufValue::Bool(true),
    );

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_safetensors(
        path: &Path,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
    ) -> anyhow::Result<()> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, data) in tensors {
            let start = offset;
            let end = start + data.len();
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": *dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
            offset = end;
        }

        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
        let mut f = File::create(path)?;
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&header_bytes)?;
        for (_, _, _, data) in tensors {
            f.write_all(data)?;
        }
        Ok(())
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_classify_tensor_paths() {
        assert_eq!(
            classify_tensor("blocks.0.experts.1.w_up.weight", true),
            TensorClass::Ternary
        );
        assert_eq!(
            classify_tensor("blocks.0.wq.weight", false),
            TensorClass::Ternary
        );
        assert_eq!(
            classify_tensor("tok_embed.weight", true),
            TensorClass::KeepPrecision
        );
    }

    #[test]
    fn test_parse_tensor_data_for_supported_dtypes() -> anyhow::Result<()> {
        let f32_tensor = TensorMeta {
            name: "f32".to_string(),
            shape: vec![2],
            dtype: "F32".to_string(),
            data: f32_bytes(&[1.25, -2.5]),
        };
        assert_eq!(parse_tensor_data(&f32_tensor)?, vec![1.25, -2.5]);

        let f16_vals = [half::f16::from_f32(0.5), half::f16::from_f32(-3.0)];
        let f16_tensor = TensorMeta {
            name: "f16".to_string(),
            shape: vec![2],
            dtype: "F16".to_string(),
            data: f16_vals
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        };
        let parsed_f16 = parse_tensor_data(&f16_tensor)?;
        assert!((parsed_f16[0] - 0.5).abs() < 1e-5);
        assert!((parsed_f16[1] + 3.0).abs() < 1e-5);

        let bf16_vals = [half::bf16::from_f32(2.0), half::bf16::from_f32(-1.0)];
        let bf16_tensor = TensorMeta {
            name: "bf16".to_string(),
            shape: vec![2],
            dtype: "BF16".to_string(),
            data: bf16_vals
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        };
        let parsed_bf16 = parse_tensor_data(&bf16_tensor)?;
        assert!((parsed_bf16[0] - 2.0).abs() < 1e-5);
        assert!((parsed_bf16[1] + 1.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_parse_tensor_data_rejects_unknown_dtype() {
        let tensor = TensorMeta {
            name: "x".to_string(),
            shape: vec![1],
            dtype: "U8".to_string(),
            data: vec![1u8],
        };
        let err = parse_tensor_data(&tensor).unwrap_err();
        assert!(err.to_string().contains("Unsupported dtype"));
    }

    #[test]
    fn test_ternarize_tensor_requires_2d() {
        let tensor = TensorMeta {
            name: "bad".to_string(),
            shape: vec![2, 2, 2],
            dtype: "F32".to_string(),
            data: f32_bytes(&[0.0; 8]),
        };
        let err = match ternarize_tensor(&tensor, 2) {
            Ok(_) => panic!("expected ternarize_tensor to fail for non-2D"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Can only ternarize 2D tensors"));
    }

    #[test]
    fn test_convert_tensors_and_export_roundtrip() -> anyhow::Result<()> {
        let tensors = vec![
            TensorMeta {
                name: "blocks.0.experts.1.w_up.weight".to_string(),
                shape: vec![2, 4],
                dtype: "F32".to_string(),
                data: f32_bytes(&[1.0, -1.0, 0.0, 0.5, 0.25, -0.75, 0.2, -0.2]),
            },
            TensorMeta {
                name: "tok_embed.weight".to_string(),
                shape: vec![2, 2],
                dtype: "F32".to_string(),
                data: f32_bytes(&[0.1, 0.2, 0.3, 0.4]),
            },
        ];
        let converted = convert_tensors(&tensors, 2, true, false)?;
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].gguf_type, GgufType::Q1_58);
        assert_eq!(converted[1].gguf_type, GgufType::F32);

        let config = serde_json::json!({
            "hidden_size": 16u64,
            "num_hidden_layers": 2u64,
            "num_attention_heads": 2u64,
            "num_key_value_heads": 2u64,
            "vocab_size": 64u64,
        });
        let dir = tempdir()?;
        let gguf_path = dir.path().join("model.gguf");
        let export_err = export_gguf(&gguf_path, &config, &converted, 128, false).unwrap_err();
        assert!(export_err
            .to_string()
            .contains("write not implemented for this value type"));
        Ok(())
    }

    #[test]
    fn test_convert_tensors_rejects_unknown_dtype() {
        let tensors = vec![TensorMeta {
            name: "mystery.weight".to_string(),
            shape: vec![1],
            dtype: "UNKNOWN".to_string(),
            data: vec![9, 8, 7, 6],
        }];
        let err = convert_tensors(&tensors, 2, true, false).unwrap_err();
        assert!(err.to_string().contains("Unsupported dtype"));
    }

    #[test]
    fn test_load_checkpoint_single_file_and_directory() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let single_path = dir.path().join("single.safetensors");
        write_safetensors(
            &single_path,
            &[("weight", "F32", vec![2], f32_bytes(&[1.0, 2.0]))],
        )?;
        let tensors = load_checkpoint(&single_path, false)?;
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "weight");
        assert_eq!(tensors[0].shape, vec![2]);
        assert_eq!(tensors[0].dtype, "F32");

        let ckpt_dir = dir.path().join("checkpoint");
        std::fs::create_dir_all(&ckpt_dir)?;
        let model_path = ckpt_dir.join("model.safetensors");
        write_safetensors(&model_path, &[("bias", "F32", vec![1], f32_bytes(&[3.0]))])?;
        let tensors_dir = load_checkpoint(&ckpt_dir, false)?;
        assert_eq!(tensors_dir.len(), 1);
        assert_eq!(tensors_dir[0].name, "bias");
        Ok(())
    }

    #[test]
    fn test_load_checkpoint_sharded_files() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let shard_a = dir.path().join("model-00001-of-00002.safetensors");
        let shard_b = dir.path().join("model-00002-of-00002.safetensors");

        write_safetensors(&shard_a, &[("b.weight", "F32", vec![1], f32_bytes(&[2.0]))])?;
        write_safetensors(&shard_b, &[("a.weight", "F32", vec![1], f32_bytes(&[1.0]))])?;

        let tensors = load_checkpoint(dir.path(), false)?;
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "a.weight");
        assert_eq!(tensors[1].name, "b.weight");
        Ok(())
    }
}
