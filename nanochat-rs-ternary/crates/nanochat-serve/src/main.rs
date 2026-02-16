//! nanochat-serve — OpenAI-compatible inference server for nanochat ternary models.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nanochat_model::model::NanochatModel;
use nanochat_serve::api::ChatTemplate;
use nanochat_serve::engine::{EngineHandle, InferenceEngine, NumaInferenceEngine};
use nanochat_serve::server::{build_router, AppState};

struct Args {
    model: String,
    mhc: String,
    tokenizer: String,
    host: String,
    port: u16,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut mhc = String::new();
    let mut tokenizer_path = String::new();
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
            "--tokenizer" => {
                i += 1;
                tokenizer_path = args.get(i).cloned().unwrap_or_default();
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
                eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> --tokenizer <path.json> [--port 8080] [--host 0.0.0.0]");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> --tokenizer <path.json> [--port 8080] [--host 0.0.0.0]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model.is_empty() || mhc.is_empty() || tokenizer_path.is_empty() {
        eprintln!("Error: --model, --mhc, and --tokenizer are required");
        eprintln!("Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> --tokenizer <path.json> [--port 8080] [--host 0.0.0.0]");
        std::process::exit(1);
    }

    Args {
        model,
        mhc,
        tokenizer: tokenizer_path,
        host,
        port,
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Register Prometheus metrics
    nanochat_serve::metrics::register_metrics();

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

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer).unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer from {}: {e}", args.tokenizer);
        std::process::exit(1);
    });
    let tok_vocab = tokenizer.get_vocab_size(true);
    tracing::info!(
        "Tokenizer loaded from {} ({} tokens)",
        args.tokenizer,
        tok_vocab
    );

    let model_name = format!("nanochat-{}m", model.param_count().total / 1_000_000);
    let vocab_size = model.config.vocab_size as u32;
    let max_seq_len = model.config.max_seq_len;
    let api_key = std::env::var("NANOCHAT_API_KEY")
        .ok()
        .filter(|v| !v.is_empty());
    let request_timeout = Duration::from_secs(
        std::env::var("NANOCHAT_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&secs| secs > 0)
            .unwrap_or(120),
    );
    let cors_allowed_origin = std::env::var("NANOCHAT_CORS_ORIGIN")
        .ok()
        .filter(|v| !v.is_empty());
    let chat_template = std::env::var("NANOCHAT_CHAT_TEMPLATE")
        .ok()
        .and_then(|v| ChatTemplate::parse(&v))
        .unwrap_or(ChatTemplate::RoleTagged);
    let numa_mode_raw = std::env::var("NANOCHAT_NUMA_MODE").unwrap_or_else(|_| "off".to_string());
    let use_numa_engine = match numa_mode_raw.trim().to_ascii_lowercase().as_str() {
        "off" | "false" | "0" => false,
        "threadpool" | "on" | "true" | "1" => true,
        other => {
            tracing::warn!("Unknown NANOCHAT_NUMA_MODE='{}', defaulting to off", other);
            false
        }
    };

    if api_key.is_some() {
        tracing::info!("API key auth enabled for /v1 endpoints");
    } else {
        tracing::warn!("API key auth disabled (set NANOCHAT_API_KEY for production)");
    }
    tracing::info!(
        "Request timeout configured: {}s (NANOCHAT_REQUEST_TIMEOUT_SECS)",
        request_timeout.as_secs()
    );
    if let Some(origin) = cors_allowed_origin.as_deref() {
        tracing::info!("CORS restricted to origin: {}", origin);
    } else {
        tracing::warn!("CORS origin not set; cross-origin browser access is disabled by default");
    }
    tracing::info!("Chat template: {:?}", chat_template);
    tracing::info!(
        "NUMA engine mode: {}",
        if use_numa_engine { "threadpool" } else { "off" }
    );

    let mut engines = Vec::new();
    if use_numa_engine {
        let numa_engine = NumaInferenceEngine::new(model);
        tracing::info!("{}", numa_engine.numa_status());
        engines.push(std::sync::Mutex::new(EngineHandle::Numa(numa_engine)));
    } else {
        engines.push(std::sync::Mutex::new(EngineHandle::Standard(
            InferenceEngine::new(model),
        )));
    }

    // Optional engine replication for concurrent streaming requests.
    let replicas = std::env::var("NANOCHAT_ENGINE_REPLICAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    if replicas > 1 {
        tracing::info!("Initializing {} inference engine replicas", replicas);
        for replica_idx in 1..replicas {
            let replica_model =
                NanochatModel::from_gguf(&args.model, &args.mhc).unwrap_or_else(|e| {
                    eprintln!("Failed to load replica {} model: {e}", replica_idx);
                    std::process::exit(1);
                });
            if use_numa_engine {
                let replica = NumaInferenceEngine::new(replica_model);
                tracing::info!("replica {}: {}", replica_idx, replica.numa_status());
                engines.push(std::sync::Mutex::new(EngineHandle::Numa(replica)));
            } else {
                engines.push(std::sync::Mutex::new(EngineHandle::Standard(
                    InferenceEngine::new(replica_model),
                )));
            }
        }
    }

    let state = Arc::new(AppState {
        engines,
        next_engine: AtomicUsize::new(0),
        tokenizer,
        model_name,
        vocab_size,
        max_seq_len,
        chat_template,
        api_key,
        request_timeout,
        cors_allowed_origin,
    });

    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
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
