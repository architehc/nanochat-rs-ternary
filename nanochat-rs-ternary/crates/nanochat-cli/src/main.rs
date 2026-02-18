//! Interactive CLI for nanochat ternary models.
//!
//! Provides subcommands:
//! - chat: Interactive REPL for code generation
//! - generate: One-shot code generation
//! - benchmark: Performance benchmarking
//! - info: Model information

use clap::{Parser, Subcommand};
use colored::Colorize;
use rustyline::DefaultEditor;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanochat")]
#[command(about = "Ternary quantized code generation model", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat mode (REPL)
    Chat {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to mHC weights file
        #[arg(short = 'c', long)]
        mhc: PathBuf,

        /// Maximum tokens to generate
        #[arg(short = 't', long, default_value = "200")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Generate code from prompt
    Generate {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to mHC weights file
        #[arg(short = 'c', long)]
        mhc: PathBuf,

        /// Prompt for code generation
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 't', long, default_value = "200")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Run performance benchmark
    Benchmark {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to mHC weights file
        #[arg(short = 'c', long)]
        mhc: PathBuf,

        /// Number of samples to run
        #[arg(short, long, default_value = "10")]
        n_samples: usize,

        /// Sequence length for benchmark
        #[arg(short, long, default_value = "100")]
        seq_len: usize,
    },

    /// Display model information
    Info {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to mHC weights file
        #[arg(short = 'c', long)]
        mhc: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat {
            model,
            mhc,
            max_tokens,
            temperature,
        } => interactive_chat(&model, &mhc, max_tokens, temperature)?,

        Commands::Generate {
            model,
            mhc,
            prompt,
            max_tokens,
            temperature,
        } => generate_code(&model, &mhc, &prompt, max_tokens, temperature)?,

        Commands::Benchmark {
            model,
            mhc,
            n_samples,
            seq_len,
        } => run_benchmark(&model, &mhc, n_samples, seq_len)?,

        Commands::Info { model, mhc } => show_model_info(&model, &mhc)?,
    }

    Ok(())
}

fn load_model(
    model_path: &Path,
    mhc_path: &Path,
) -> anyhow::Result<nanochat_model::model::NanochatModel> {
    println!("{}", "Loading model...".yellow());
    let start = Instant::now();

    let model = nanochat_model::model::NanochatModel::from_gguf(
        model_path.to_str().unwrap(),
        mhc_path.to_str().unwrap(),
    )?;

    let elapsed = start.elapsed();
    println!("{} Loaded in {:.2}s", "✓".green(), elapsed.as_secs_f32());
    println!(
        "  {} dim={}, layers={}, heads={}, vocab={}",
        "Model:".cyan(),
        model.config.dim,
        model.config.n_layers,
        model.config.n_heads,
        model.config.vocab_size
    );
    println!();

    Ok(model)
}

fn interactive_chat(
    model_path: &Path,
    mhc_path: &Path,
    max_tokens: usize,
    _temperature: f32,
) -> anyhow::Result<()> {
    let mut model = load_model(model_path, mhc_path)?;

    println!("{}", "Interactive Chat Mode".bold().cyan());
    println!(
        "{}",
        "Type 'exit' or 'quit' to exit, 'clear' to reset context\n".dimmed()
    );

    let mut rl = DefaultEditor::new()?;
    let mut context: Vec<u32> = Vec::new();

    loop {
        let readline = rl.readline(&format!("{} ", ">>".green().bold()));

        match readline {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                if line == "exit" || line == "quit" {
                    println!("{}", "Goodbye!".yellow());
                    break;
                }

                if line == "clear" {
                    context.clear();
                    println!("{}", "Context cleared.".yellow());
                    continue;
                }

                // For simplicity, use token IDs directly (in real implementation, use tokenizer)
                // Here we just generate based on existing context
                let prompt_tokens: Vec<u32> = line
                    .chars()
                    .map(|c| (c as u32) % model.config.vocab_size as u32)
                    .collect();

                context.extend_from_slice(&prompt_tokens);

                // Limit context length
                if context.len() > model.config.max_seq_len {
                    let start = context.len() - model.config.max_seq_len;
                    context = context[start..].to_vec();
                }

                println!("{}", "--- Generating ---".cyan());
                let start = Instant::now();

                // Generate tokens
                let mut generated = 0;
                while generated < max_tokens && context.len() < model.config.max_seq_len {
                    let logits = model.forward_sequence(&context);

                    // Greedy sampling (take argmax)
                    let next_token = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u32)
                        .unwrap();

                    context.push(next_token);
                    generated += 1;

                    // Simple stop condition (in practice, use proper EOS handling)
                    if next_token == 0 {
                        break;
                    }
                }

                let elapsed = start.elapsed();
                let tokens_per_sec = generated as f32 / elapsed.as_secs_f32();

                println!(
                    "{} Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
                    "✓".green(),
                    generated,
                    elapsed.as_secs_f32(),
                    tokens_per_sec
                );
                println!();
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("{}", "^C".dimmed());
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("{}", "^D".dimmed());
                break;
            }
            Err(err) => {
                eprintln!("{} {}", "Error:".red(), err);
                break;
            }
        }
    }

    Ok(())
}

fn generate_code(
    model_path: &Path,
    mhc_path: &Path,
    prompt: &str,
    max_tokens: usize,
    _temperature: f32,
) -> anyhow::Result<()> {
    let mut model = load_model(model_path, mhc_path)?;

    println!("{} {}", "Prompt:".cyan(), prompt);
    println!("{}", "--- Generated Code ---".cyan());

    let start = Instant::now();

    // Simple tokenization (char-level for demo)
    let mut tokens: Vec<u32> = prompt
        .chars()
        .map(|c| (c as u32) % model.config.vocab_size as u32)
        .collect();

    let mut generated = 0;
    while generated < max_tokens && tokens.len() < model.config.max_seq_len {
        let logits = model.forward_sequence(&tokens);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        tokens.push(next_token);
        generated += 1;

        if next_token == 0 {
            break;
        }
    }

    let elapsed = start.elapsed();
    let tokens_per_sec = generated as f32 / elapsed.as_secs_f32();

    println!("{}", "----------------------".cyan());
    println!(
        "\n{} Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        "✓".green(),
        generated,
        elapsed.as_secs_f32(),
        tokens_per_sec
    );

    Ok(())
}

fn run_benchmark(
    model_path: &Path,
    mhc_path: &Path,
    n_samples: usize,
    seq_len: usize,
) -> anyhow::Result<()> {
    let mut model = load_model(model_path, mhc_path)?;

    println!("{}", "Running Benchmark".bold().cyan());
    println!("  Samples: {}", n_samples);
    println!("  Sequence length: {}", seq_len);
    println!();

    let mut total_time = 0.0;
    let mut total_tokens = 0;

    for i in 0..n_samples {
        // Create random input tokens
        let tokens: Vec<u32> = (0..seq_len)
            .map(|j| ((i * seq_len + j) % model.config.vocab_size as usize) as u32)
            .collect();

        let start = Instant::now();
        let _logits = model.forward_sequence(&tokens);
        let elapsed = start.elapsed();

        total_time += elapsed.as_secs_f64();
        total_tokens += seq_len;

        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {}/{}", i + 1, n_samples);
        }
    }
    println!();

    let avg_time = total_time / n_samples as f64;
    let tokens_per_sec = total_tokens as f64 / total_time;
    let time_per_token = total_time / total_tokens as f64 * 1000.0; // ms

    println!("{}", "\nBenchmark Results:".bold().green());
    println!("  Average latency: {:.2}ms per sample", avg_time * 1000.0);
    println!("  Throughput: {:.1} tokens/sec", tokens_per_sec);
    println!("  Time per token: {:.2}ms", time_per_token);

    Ok(())
}

fn show_model_info(model_path: &Path, mhc_path: &Path) -> anyhow::Result<()> {
    let model = load_model(model_path, mhc_path)?;

    println!("{}", "Model Information".bold().cyan());
    println!("{}", "─".repeat(50));
    println!("Architecture:");
    println!("  Dimension: {}", model.config.dim);
    println!("  Layers: {}", model.config.n_layers);
    println!("  Attention heads: {}", model.config.n_heads);
    println!("  KV heads: {}", model.config.n_kv_heads);
    println!("  FFN multiplier: {:.2}", model.config.ffn_mult);
    println!("  Vocabulary size: {}", model.config.vocab_size);
    println!("  Max sequence length: {}", model.config.max_seq_len);
    println!("  Group size: {}", model.config.group_size);
    println!("  mHC streams: {}", model.config.mhc_n_streams);
    println!("  RoPE theta: {}", model.config.rope_theta);
    println!("  Weight tied: {}", model.config.weight_tied);
    println!("{}", "─".repeat(50));

    // Verify mHC matrices
    match model.verify_mhc() {
        Ok(()) => println!("{} mHC matrices verified (doubly stochastic)", "✓".green()),
        Err(e) => println!("{} mHC verification failed: {}", "✗".red(), e),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn reference_model_paths() -> (PathBuf, PathBuf) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let gguf = root.join("training").join("ref.gguf");
        let mhc = root.join("training").join("ref.mhc");
        assert!(gguf.exists(), "missing test gguf at {}", gguf.display());
        assert!(mhc.exists(), "missing test mhc at {}", mhc.display());
        (gguf, mhc)
    }

    #[test]
    fn test_cli_parse_generate_defaults() {
        let cli = Cli::try_parse_from([
            "nanochat",
            "generate",
            "--model",
            "m.gguf",
            "--mhc",
            "m.mhc",
            "--prompt",
            "hello",
        ])
        .expect("parse generate");
        match cli.command {
            Commands::Generate {
                model,
                mhc,
                prompt,
                max_tokens,
                temperature,
            } => {
                assert_eq!(model, PathBuf::from("m.gguf"));
                assert_eq!(mhc, PathBuf::from("m.mhc"));
                assert_eq!(prompt, "hello");
                assert_eq!(max_tokens, 200);
                assert!((temperature - 0.7).abs() < f32::EPSILON);
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn test_cli_parse_benchmark_defaults() {
        let cli = Cli::try_parse_from([
            "nanochat",
            "benchmark",
            "--model",
            "m.gguf",
            "--mhc",
            "m.mhc",
        ])
        .expect("parse benchmark");
        match cli.command {
            Commands::Benchmark {
                n_samples, seq_len, ..
            } => {
                assert_eq!(n_samples, 10);
                assert_eq!(seq_len, 100);
            }
            _ => panic!("expected benchmark command"),
        }
    }

    #[test]
    fn test_load_model_and_info_path() -> anyhow::Result<()> {
        let (gguf, mhc) = reference_model_paths();
        let model = load_model(&gguf, &mhc)?;
        assert_eq!(model.config.vocab_size, 256);
        show_model_info(&gguf, &mhc)?;
        Ok(())
    }

    #[test]
    fn test_generate_code_executes() -> anyhow::Result<()> {
        let (gguf, mhc) = reference_model_paths();
        generate_code(&gguf, &mhc, "fn add(a:i32,b:i32)->i32{a+b}", 4, 0.7)?;
        Ok(())
    }

    #[test]
    fn test_run_benchmark_executes() -> anyhow::Result<()> {
        let (gguf, mhc) = reference_model_paths();
        run_benchmark(&gguf, &mhc, 2, 8)?;
        Ok(())
    }

    #[test]
    fn test_load_model_invalid_path_errors() {
        let err = load_model(Path::new("/does/not/exist.gguf"), Path::new("/does/not/exist.mhc"));
        assert!(err.is_err());
    }
}
