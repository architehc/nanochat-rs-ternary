//! nanochat-serve — OpenAI-compatible inference server for nanochat ternary models.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nanochat_model::model::NanochatModel;
use nanochat_serve::api::ChatTemplate;
use nanochat_serve::engine::{EngineHandle, InferenceEngine, NumaInferenceEngine};
use nanochat_serve::server::{build_router, AppState};

const USAGE: &str = "Usage: nanochat-serve --model <path.gguf> --mhc <path.mhc> --tokenizer <path.json> [--port 8080] [--host 0.0.0.0]";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    model: String,
    mhc: String,
    tokenizer: String,
    host: String,
    port: u16,
}

#[derive(Debug, PartialEq, Eq)]
enum ParseArgsError {
    Help,
    Message(String),
}

fn parse_args_from<I, S>(args: I) -> Result<Args, ParseArgsError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
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
                return Err(ParseArgsError::Help);
            }
            other => {
                return Err(ParseArgsError::Message(format!(
                    "Unknown argument: {}",
                    other
                )));
            }
        }
        i += 1;
    }

    if model.is_empty() || mhc.is_empty() || tokenizer_path.is_empty() {
        return Err(ParseArgsError::Message(
            "Error: --model, --mhc, and --tokenizer are required".to_string(),
        ));
    }

    Ok(Args {
        model,
        mhc,
        tokenizer: tokenizer_path,
        host,
        port,
    })
}

fn parse_args() -> Args {
    match parse_args_from(std::env::args()) {
        Ok(args) => args,
        Err(ParseArgsError::Help) => {
            eprintln!("{}", USAGE);
            std::process::exit(0);
        }
        Err(ParseArgsError::Message(message)) => {
            eprintln!("{}", message);
            eprintln!("{}", USAGE);
            std::process::exit(1);
        }
    }
}

fn parse_numa_mode(raw: &str) -> (NumaMode, bool) {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" | "false" | "0" => (NumaMode::Off, true),
        "threadpool" | "on" | "true" | "1" => (NumaMode::ThreadPool, true),
        "local" => (NumaMode::Local, true),
        _ => (NumaMode::Off, false),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumaMode {
    Off,
    ThreadPool, // One engine uses all nodes
    Local,      // Replicas distributed across nodes, each node-local
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
    nanochat_serve::metrics::MODEL_LOAD_TIME.set(elapsed.as_secs_f64());
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
    let stream_channel_capacity = std::env::var("NANOCHAT_STREAM_CHANNEL_CAPACITY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64);
    let numa_mode_raw = std::env::var("NANOCHAT_NUMA_MODE").unwrap_or_else(|_| "off".to_string());
    let (numa_mode, known_numa_mode) = parse_numa_mode(&numa_mode_raw);
    if !known_numa_mode {
        tracing::warn!(
            "Unknown NANOCHAT_NUMA_MODE='{}', defaulting to off",
            numa_mode_raw
        );
    }

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
    tracing::info!("Stream channel capacity: {}", stream_channel_capacity);
    tracing::info!("NUMA engine mode: {:?}", numa_mode);

    let numa_config = nanochat_serve::engine::NumaConfig::detect();
    let mut engines = Vec::new();
    
    let replicas = std::env::var("NANOCHAT_ENGINE_REPLICAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);

    for replica_idx in 0..replicas {
        let node_idx = replica_idx % numa_config.num_nodes;
        
        // Clone model to the assigned node if using NUMA
        let replica_model = if numa_mode != NumaMode::Off {
            tracing::info!("Cloning model for replica {} to NUMA node {}", replica_idx, node_idx);
            model.clone_to_node(node_idx)
        } else {
            model.clone()
        };

        match numa_mode {
            NumaMode::ThreadPool => {
                let engine = NumaInferenceEngine::new(replica_model);
                tracing::info!("Replica {} (ThreadPool): {}", replica_idx, engine.numa_status());
                engines.push(std::sync::Mutex::new(EngineHandle::Numa(engine)));
            }
            NumaMode::Local | NumaMode::Off => {
                if numa_mode == NumaMode::Local {
                    let engine = NumaInferenceEngine::new(replica_model).with_preferred_node(node_idx);
                    tracing::info!("Replica {} (Local): Node {}", replica_idx, node_idx);
                    engines.push(std::sync::Mutex::new(EngineHandle::Numa(engine)));
                } else {
                    engines.push(std::sync::Mutex::new(EngineHandle::Standard(
                        InferenceEngine::new(replica_model),
                    )));
                }
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
        stream_channel_capacity,
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

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("Shutting down gracefully...");
        })
        .await
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_from_success_defaults() {
        let parsed = parse_args_from([
            "nanochat-serve",
            "--model",
            "m.gguf",
            "--mhc",
            "m.mhc",
            "--tokenizer",
            "tok.json",
        ])
        .expect("parse success");
        assert_eq!(parsed.model, "m.gguf");
        assert_eq!(parsed.mhc, "m.mhc");
        assert_eq!(parsed.tokenizer, "tok.json");
        assert_eq!(parsed.host, "0.0.0.0");
        assert_eq!(parsed.port, 8080);
    }

    #[test]
    fn test_parse_args_from_success_overrides() {
        let parsed = parse_args_from([
            "nanochat-serve",
            "--model",
            "m.gguf",
            "--mhc",
            "m.mhc",
            "--tokenizer",
            "tok.json",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
        ])
        .expect("parse success");
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.port, 9001);
    }

    #[test]
    fn test_parse_args_from_invalid_port_defaults() {
        let parsed = parse_args_from([
            "nanochat-serve",
            "--model",
            "m.gguf",
            "--mhc",
            "m.mhc",
            "--tokenizer",
            "tok.json",
            "--port",
            "not-a-number",
        ])
        .expect("parse success");
        assert_eq!(parsed.port, 8080);
    }

    #[test]
    fn test_parse_args_from_missing_required_errors() {
        let err = parse_args_from(["nanochat-serve", "--model", "m.gguf"])
            .expect_err("missing required args should fail");
        assert!(matches!(err, ParseArgsError::Message(_)));
        assert_eq!(
            err,
            ParseArgsError::Message(
                "Error: --model, --mhc, and --tokenizer are required".to_string()
            )
        );
    }

    #[test]
    fn test_parse_args_from_unknown_argument_errors() {
        let err = parse_args_from(["nanochat-serve", "--wat", "x"])
            .expect_err("unknown arg should fail");
        assert_eq!(
            err,
            ParseArgsError::Message("Unknown argument: --wat".to_string())
        );
    }

    #[test]
    fn test_parse_args_from_help() {
        let err = parse_args_from(["nanochat-serve", "--help"]).expect_err("help should return");
        assert_eq!(err, ParseArgsError::Help);
    }

    #[test]
    fn test_parse_numa_mode_variants() {
        assert_eq!(parse_numa_mode("off"), (NumaMode::Off, true));
        assert_eq!(parse_numa_mode("0"), (NumaMode::Off, true));
        assert_eq!(parse_numa_mode("threadpool"), (NumaMode::ThreadPool, true));
        assert_eq!(parse_numa_mode("ON"), (NumaMode::ThreadPool, true));
        assert_eq!(parse_numa_mode("local"), (NumaMode::Local, true));
        assert_eq!(parse_numa_mode("weird"), (NumaMode::Off, false));
    }
}
