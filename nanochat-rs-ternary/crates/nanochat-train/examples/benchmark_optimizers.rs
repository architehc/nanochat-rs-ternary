use clap::Parser;
use nanochat_train::config::TrainConfig;
use nanochat_train::data::SyntheticDataset;
use nanochat_train::train::Trainer;
use serde::Serialize;
use std::fs;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "benchmark_optimizers")]
#[command(about = "Benchmark Muon optimizer variants (FP32, 8-bit, GaLore2, GaLore2+8-bit)")]
struct Args {
    /// Number of train steps per optimizer variant.
    #[arg(long, default_value_t = 100)]
    steps: usize,

    /// Output JSON report path.
    #[arg(long, default_value = "benchmark_optimizers.json")]
    output: String,

    /// Optional single variant name:
    /// muon | muon_8bit | galore2 | galore2_muon_8bit
    #[arg(long)]
    variant: Option<String>,
}

#[derive(Serialize)]
struct VariantReport {
    name: String,
    steps: usize,
    avg_loss: f64,
    avg_tokens_per_sec: f64,
    wall_seconds: f64,
    optimizer_variant: String,
    memory_reduction: f64,
    optimizer_details: String,
}

#[derive(Serialize)]
struct BenchmarkReport {
    steps_per_variant: usize,
    results: Vec<VariantReport>,
}

type VariantSpec = (&'static str, bool, bool);

fn build_config(use_8bit: bool, use_galore: bool) -> TrainConfig {
    let mut cfg = TrainConfig::d20();
    cfg.total_steps = 10_000;
    cfg.batch_size = 2;
    cfg.max_seq_len = 64;
    cfg.grad_accum_steps = 1;

    // Keep benchmark focused on optimizer behavior.
    cfg.use_mtp = false;
    cfg.use_collider = false;
    cfg.use_async_loader = false;

    cfg.use_8bit_optim = use_8bit;
    cfg.use_galore = use_galore;
    if use_galore {
        cfg.galore_rank = 32;
        cfg.galore_update_freq = 10;
    }
    cfg
}

fn benchmark_variant(
    name: &str,
    steps: usize,
    use_8bit: bool,
    use_galore: bool,
) -> Result<VariantReport, Box<dyn std::error::Error>> {
    let device = candle_core::Device::Cpu;
    let cfg = build_config(use_8bit, use_galore);
    let mut trainer = Trainer::new(cfg.clone(), device)?;

    let seq_len = cfg.max_seq_len / 2;
    let n_samples = steps.saturating_mul(cfg.batch_size).max(cfg.batch_size);
    let ds = SyntheticDataset::new(cfg.vocab_size as u32, seq_len, n_samples, 42);
    let loader =
        nanochat_train::data::DataLoader::new(&ds, cfg.batch_size, false, 42, &trainer.device);

    let mut loss_sum = 0.0f64;
    let mut tok_sum = 0.0f64;
    let mut ran_steps = 0usize;
    let wall_start = Instant::now();

    for batch in loader {
        if ran_steps >= steps {
            break;
        }
        let (input_ids, target_ids) = batch?;
        let stats = trainer.train_step(&input_ids, &target_ids)?;
        loss_sum += stats.loss;
        tok_sum += stats.tokens_per_sec;
        ran_steps += 1;
    }
    let mem = trainer.optimizer_memory_stats();

    let denom = ran_steps.max(1) as f64;
    Ok(VariantReport {
        name: name.to_string(),
        steps: ran_steps,
        avg_loss: loss_sum / denom,
        avg_tokens_per_sec: tok_sum / denom,
        wall_seconds: wall_start.elapsed().as_secs_f64(),
        optimizer_variant: mem.variant.to_string(),
        memory_reduction: mem.memory_reduction,
        optimizer_details: mem.details,
    })
}

fn selected_variants(single: Option<&str>) -> Result<Vec<VariantSpec>, Box<dyn std::error::Error>> {
    let all = vec![
        ("muon", false, false),
        ("muon_8bit", true, false),
        ("galore2", false, true),
        ("galore2_muon_8bit", true, true),
    ];
    if let Some(s) = single {
        let one = all
            .iter()
            .copied()
            .find(|(name, _, _)| *name == s)
            .ok_or_else(|| format!("unknown variant '{}'", s))?;
        Ok(vec![one])
    } else {
        Ok(all)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let variants = selected_variants(args.variant.as_deref())?;

    let mut reports = Vec::with_capacity(variants.len());
    for (name, use_8bit, use_galore) in variants {
        println!(
            "Benchmarking {} (8bit={}, galore={})",
            name, use_8bit, use_galore
        );
        let report = benchmark_variant(name, args.steps, use_8bit, use_galore)?;
        println!(
            "  steps={} loss={:.4} tok/s={:.1} mem_red={:.1}% ({})",
            report.steps,
            report.avg_loss,
            report.avg_tokens_per_sec,
            report.memory_reduction * 100.0,
            report.optimizer_variant
        );
        reports.push(report);
    }

    let output = BenchmarkReport {
        steps_per_variant: args.steps,
        results: reports,
    };
    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&args.output, json)?;
    println!("Wrote benchmark report to {}", args.output);
    Ok(())
}
