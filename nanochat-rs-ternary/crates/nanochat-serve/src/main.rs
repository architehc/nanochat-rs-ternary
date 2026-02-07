//! nanochat-serve — OpenAI-compatible inference server for nanochat ternary models.

use std::sync::Arc;
use std::time::Instant;

use nanochat_model::model::NanochatModel;
use nanochat_serve::engine::InferenceEngine;
use nanochat_serve::server::{AppState, build_router};

struct Args {
    model: String,
    mhc: String,
    host: String,
    port: u16,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut mhc = String::new();
    let mut host = "0.0.0.0".to_string();
    let mut port = 8080u16;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).cloned().unwrap_or_default();
            }
            "--mhc" => {
                i += 1;
                mhc = args.get(i).cloned().unwrap_or_default();
            }
            "--host" => {
                i += 1;
                host = args.get(i).cloned().unwrap_or_default();
            }
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8080);
            }
            "--help" | "-h" => {
                eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> [--port 8080] [--host 0.0.0.0]");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> [--port 8080] [--host 0.0.0.0]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model.is_empty() || mhc.is_empty() {
        eprintln!("Error: --model and --mhc are required");
        eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> [--port 8080] [--host 0.0.0.0]");
        std::process::exit(1);
    }

    Args { model, mhc, host, port }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = parse_args();

    // Load model
    tracing::info!("Loading model from {} + {}", args.model, args.mhc);
    let start = Instant::now();
    let model = NanochatModel::from_gguf(&args.model, &args.mhc).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

    let elapsed = start.elapsed();
    let config = &model.config;
    tracing::info!(
        "Model loaded in {:.2}s: dim={}, layers={}, heads={}, vocab={}, params={}",
        elapsed.as_secs_f32(),
        config.dim,
        config.n_layers,
        config.n_heads,
        config.vocab_size,
        model.param_count().total,
    );

    // Verify mHC integrity
    model.verify_mhc().unwrap_or_else(|e| {
        eprintln!("mHC verification failed: {e}");
        std::process::exit(1);
    });
    tracing::info!("mHC doubly-stochastic verification passed");

    // Initialize tokenizer
    let tokenizer = tiktoken_rs::r50k_base().expect("Failed to load GPT-2 tokenizer");
    tracing::info!("Tokenizer loaded (GPT-2 r50k_base, 50257 tokens)");

    let model_name = format!("nanochat-{}m", model.param_count().total / 1_000_000);
    let engine = InferenceEngine::new(model);

    let state = Arc::new(AppState {
        engine: std::sync::Mutex::new(engine),
        tokenizer,
        model_name,
    });

    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
        eprintln!("Failed to bind to {addr}: {e}");
        std::process::exit(1);
    });
    tracing::info!("Listening on http://{addr}");
    tracing::info!("Endpoints:");
    tracing::info!("  GET  /                    (chat UI)");
    tracing::info!("  POST /v1/chat/completions");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /health");

    axum::serve(listener, app).await.unwrap();
}
