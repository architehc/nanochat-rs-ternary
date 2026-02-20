//! Axum HTTP server with OpenAI-compatible endpoints.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio_stream::wrappers::ReceiverStream;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::cors::CorsLayer;

use crate::api::*;
use crate::engine::EngineHandle;
use crate::metrics;

/// Shared application state.
pub struct AppState {
    pub engines: Vec<std::sync::Mutex<EngineHandle>>,
    pub next_engine: AtomicUsize,
    pub tokenizer: tokenizers::Tokenizer,
    pub model_name: String,
    pub vocab_size: u32,
    pub max_seq_len: usize,
    pub chat_template: ChatTemplate,
    pub api_key: Option<String>,
    pub request_timeout: Duration,
    pub cors_allowed_origin: Option<String>,
    pub stream_channel_capacity: usize,
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> Router {
    let mut cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE]);
    if let Some(origin) = state.cors_allowed_origin.as_deref() {
        match HeaderValue::from_str(origin) {
            Ok(value) => {
                cors = cors.allow_origin(value);
            }
            Err(err) => {
                eprintln!(
                    "WARNING: ignoring invalid CORS origin '{}': {}",
                    origin, err
                );
            }
        }
    }

    Router::new()
        .route("/", get(chat_ui))
        .route("/health", get(health))
        .route("/metrics", get(prometheus_metrics_authed))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .layer(ConcurrencyLimitLayer::new(128))
        .layer(cors)
        .with_state(state)
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Create a JSON error response with proper HTTP status code.
fn error_json(status: StatusCode, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": null
            }
        })),
    )
}

/// Constant-time string comparison to prevent timing attacks on API keys.
/// Iterates over max(a.len(), b.len()) to avoid leaking length via timing.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let mut result = (a.len() ^ b.len()) as u8;
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let ab = a.as_bytes().get(i).copied().unwrap_or(0);
        let bb = b.as_bytes().get(i).copied().unwrap_or(0);
        result |= ab ^ bb;
    }
    result == 0
}

fn unauthorized_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "unauthorized",
            "type": "authentication_error"
        }
    });
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

fn is_authorized(headers: &HeaderMap, api_key: Option<&str>) -> bool {
    let Some(expected) = api_key else {
        return true;
    };

    let bearer = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    if let Some(auth) = bearer {
        let expected_bearer = format!("Bearer {}", expected);
        if constant_time_eq(auth, &expected_bearer) {
            return true;
        }
    }

    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .is_some_and(|provided| constant_time_eq(provided, expected))
}

async fn chat_ui() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'",
        ),
    );
    (headers, Html(CHAT_HTML))
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.engines.is_empty() {
        return (StatusCode::SERVICE_UNAVAILABLE, "no engines available").into_response();
    }
    // Probe each engine mutex to detect poisoning (prior panic).
    for (i, engine_mutex) in state.engines.iter().enumerate() {
        if engine_mutex.lock().is_err() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("engine {} mutex poisoned", i),
            )
                .into_response();
        }
    }
    (StatusCode::OK, "ok").into_response()
}

async fn prometheus_metrics_authed(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !is_authorized(&headers, state.api_key.as_deref()) {
        return unauthorized_response();
    }
    metrics::render_metrics().into_response()
}

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !is_authorized(&headers, state.api_key.as_deref()) {
        return unauthorized_response();
    }
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "nanochat"
        }]
    }))
    .into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if !is_authorized(&headers, state.api_key.as_deref()) {
        return unauthorized_response();
    }

    let stream = req.stream.unwrap_or(false);

    if stream {
        stream_completion(state, req).await.into_response()
    } else {
        non_stream_completion(state, req).await.into_response()
    }
}

async fn non_stream_completion(state: Arc<AppState>, req: ChatCompletionRequest) -> Response {
    metrics::INFERENCE_REQUESTS.inc();

    let model_name = state.model_name.clone();
    let prompt = match req.to_prompt_string_with_template(state.chat_template) {
        Ok(prompt) => prompt,
        Err(err) => {
            metrics::INFERENCE_ERRORS.inc();
            return error_json(StatusCode::BAD_REQUEST, &err).into_response();
        }
    };
    let params = req.to_sampling_params();
    let max_seq_len = state.max_seq_len;
    let request_timeout = state.request_timeout;

    // Cancellation flag: set when timeout fires so the blocking task stops generating.
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_inner = cancel.clone();

    let result = tokio::time::timeout(
        request_timeout,
        tokio::task::spawn_blocking(move || {
            // Metrics guards live inside spawn_blocking so they measure
            // the actual generation duration and stay alive until completion.
            let _active_guard = metrics::ActiveRequestGuard::new();
            let _timer = metrics::LatencyTimer::new();

            let gen_start = Instant::now();

            let prompt_ids = state
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|err| {
                    eprintln!("ERROR: tokenizer encode failed: {}", err);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to tokenize prompt".to_string(),
                    )
                })?
                .get_ids()
                .to_vec();
            let prompt_len = prompt_ids.len();

            // Validate prompt length to prevent RoPE assert panics
            if prompt_len >= max_seq_len {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Prompt too long: {} tokens exceeds model limit of {}",
                        prompt_len, max_seq_len
                    ),
                ));
            }

            let vs = state.vocab_size;
            if prompt_ids.iter().any(|&t| t >= vs) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "tokenizer produced token id outside model vocab (vocab_size={})",
                        vs
                    ),
                ));
            }
            let token_ids = prompt_ids;
            if state.engines.is_empty() {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "no inference engines are available".to_string(),
                ));
            }
            let engine_idx =
                state.next_engine.fetch_add(1, Ordering::Relaxed) % state.engines.len();
            let engine_lock = state.engines.get(engine_idx).ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "engine index out of range".to_string(),
                )
            })?;

            let mut engine = engine_lock.lock().unwrap_or_else(|poisoned| {
                eprintln!("WARNING: engine mutex was poisoned (previous task panicked), recovering");
                poisoned.into_inner()
            });

            // Use generate_streaming with cancellation check so the task
            // stops promptly when the request timeout fires.
            let mut output_ids = Vec::new();
            let mut finish_reason = crate::engine::FinishReason::Stop;
            engine.generate_streaming(&token_ids, &params, |tok| {
                if cancel_inner.load(Ordering::Relaxed) {
                    return false; // Timeout fired — stop generating
                }
                output_ids.push(tok.token_id);
                if let Some(ref fr) = tok.finish_reason {
                    if fr == "length" {
                        finish_reason = crate::engine::FinishReason::Length;
                    }
                }
                tok.finish_reason.is_none()
            });
            if output_ids.len() >= params.max_tokens
                && finish_reason == crate::engine::FinishReason::Stop
            {
                finish_reason = crate::engine::FinishReason::Length;
            }
            let output_len = output_ids.len();

            // Record tokens per second
            let gen_elapsed = gen_start.elapsed();
            if gen_elapsed.as_secs_f64() > 0.0 && output_len > 0 {
                let tps = output_len as f64 / gen_elapsed.as_secs_f64();
                metrics::TOKENS_PER_SECOND.set(tps);
            }

            // Check for degraded state after generation
            let degraded = engine.last_forward_was_degraded();
            let degraded_reason = engine.last_forward_error_message();
            if degraded {
                if let Some(reason) = degraded_reason.as_deref() {
                    eprintln!(
                        "WARNING: Non-streaming generation had degraded outputs: {}",
                        reason
                    );
                } else {
                    eprintln!(
                        "WARNING: Non-streaming generation had degraded outputs (LoopLM errors)"
                    );
                }
            }

            let text = state.tokenizer.decode(&output_ids, true).map_err(|err| {
                eprintln!("ERROR: tokenizer decode failed: {}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to decode generated tokens".to_string(),
                )
            })?;

            Ok((
                text,
                prompt_len,
                output_len,
                finish_reason,
                degraded,
                degraded_reason,
            ))
        }),
    )
    .await;

    let (
        generated_text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        degraded,
        degraded_reason,
    ) = match result {
        Ok(Ok(Ok(data))) => data,
        Ok(Ok(Err((status, err)))) => {
            metrics::INFERENCE_ERRORS.inc();
            return error_json(status, &err).into_response();
        }
        Ok(Err(join_err)) => {
            metrics::INFERENCE_ERRORS.inc();
            eprintln!("ERROR: non_stream spawn_blocking join failed: {}", join_err);
            return error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal request failure",
            )
            .into_response();
        }
        Err(_) => {
            // Signal the blocking task to stop generating tokens.
            cancel.store(true, Ordering::Relaxed);
            metrics::INFERENCE_ERRORS.inc();
            return error_json(
                StatusCode::REQUEST_TIMEOUT,
                &format!("request timed out after {}s", request_timeout.as_secs()),
            )
            .into_response();
        }
    };

    let now = unix_timestamp_secs();

    // Track generated tokens
    metrics::TOKENS_GENERATED.inc_by(completion_tokens as f64);

    let warning = if degraded {
        let reason = degraded_reason.unwrap_or_else(|| "LoopLM forward pass error".to_string());
        eprintln!(
            "WARNING: non-stream generation degraded: {}",
            reason
        );
        Some(format!("Model output may be unreliable: {}", reason))
    } else {
        None
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: now,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: Role::Assistant,
                content: generated_text,
            },
            finish_reason: finish_reason.as_str().to_string(),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
        warning,
    })
    .into_response()
}

async fn stream_completion(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    metrics::INFERENCE_REQUESTS.inc();

    let model_name = state.model_name.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let max_seq_len = state.max_seq_len;
    let request_timeout = state.request_timeout;
    let params = req.to_sampling_params();

    let stream_capacity = state.stream_channel_capacity.max(1);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(stream_capacity);
    let prompt = match req.to_prompt_string_with_template(state.chat_template) {
        Ok(prompt) => prompt,
        Err(err) => {
            metrics::INFERENCE_ERRORS.inc();
            let now = unix_timestamp_secs();
            let error_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some(Role::Assistant),
                        content: Some(format!("Error: {}", err)),
                    },
                    finish_reason: Some("error".to_string()),
                }],
            };
            if let Ok(json) = serde_json::to_string(&error_chunk) {
                let _ = tx.try_send(Ok(Event::default().data(json)));
            }
            let _ = tx.try_send(Ok(Event::default().data("[DONE]")));
            return Sse::new(ReceiverStream::new(rx));
        }
    };

    let req_id = request_id.clone();
    let model = model_name.clone();
    tokio::task::spawn_blocking(move || {
        // Metrics guards live inside spawn_blocking so they measure
        // the actual generation duration and stay alive until completion.
        let _active_guard = metrics::ActiveRequestGuard::new();
        let _timer = metrics::LatencyTimer::new();

        let try_send = |tx: &mpsc::Sender<Result<Event, Infallible>>, data: String| -> bool {
            match tx.try_send(Ok(Event::default().data(data))) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    metrics::INFERENCE_ERRORS.inc();
                    eprintln!("ERROR: stream channel backpressure (buffer full)");
                    false
                }
                Err(TrySendError::Closed(_)) => false,
            }
        };

        let send_error = |message: String| {
            let now = unix_timestamp_secs();
            let error_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some(Role::Assistant),
                        content: Some(format!("Error: {}", message)),
                    },
                    finish_reason: Some("error".to_string()),
                }],
            };
            if let Ok(json) = serde_json::to_string(&error_chunk) {
                let _ = try_send(&tx, json);
            } else {
                eprintln!("ERROR: Failed to serialize streaming error chunk");
            }
            let _ = try_send(&tx, "[DONE]".to_string());
        };

        let prompt_ids = match state.tokenizer.encode(prompt.as_str(), false) {
            Ok(encoded) => encoded.get_ids().to_vec(),
            Err(err) => {
                metrics::INFERENCE_ERRORS.inc();
                eprintln!("ERROR: tokenizer encode failed: {}", err);
                send_error("failed to tokenize prompt".to_string());
                return;
            }
        };

        // Validate prompt length before streaming
        if prompt_ids.len() >= max_seq_len {
            metrics::INFERENCE_ERRORS.inc();
            send_error(format!(
                "prompt too long ({} tokens exceeds limit of {})",
                prompt_ids.len(),
                max_seq_len
            ));
            return;
        }

        let vs = state.vocab_size;
        if prompt_ids.iter().any(|&t| t >= vs) {
            metrics::INFERENCE_ERRORS.inc();
            send_error(format!(
                "tokenizer produced token id outside model vocab (vocab_size={})",
                vs
            ));
            return;
        }
        let token_ids = prompt_ids;
        if state.engines.is_empty() {
            metrics::INFERENCE_ERRORS.inc();
            send_error("no inference engines are available".to_string());
            return;
        }
        let engine_idx = state.next_engine.fetch_add(1, Ordering::Relaxed) % state.engines.len();
        let Some(engine_lock) = state.engines.get(engine_idx) else {
            metrics::INFERENCE_ERRORS.inc();
            send_error("engine index out of range".to_string());
            return;
        };

        let now = unix_timestamp_secs();
        let started = Instant::now();

        // Send initial chunk with role
        let initial = ChatCompletionChunk {
            id: req_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some(Role::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(json) = serde_json::to_string(&initial) {
            if !try_send(&tx, json) {
                return;
            }
        } else {
            metrics::INFERENCE_ERRORS.inc();
            eprintln!("ERROR: Failed to serialize initial streaming chunk");
            let _ = try_send(&tx, "[DONE]".to_string());
            return;
        }

        let mut engine = engine_lock.lock().unwrap_or_else(|poisoned| {
            eprintln!("WARNING: engine mutex was poisoned (previous task panicked), recovering");
            poisoned.into_inner()
        });

        let mut token_count = 0usize;
        let mut had_degraded = false;
        let mut timed_out = false;
        let mut backpressured = false;
        engine.generate_streaming(&token_ids, &params, |tok| {
            if started.elapsed() >= request_timeout {
                timed_out = true;
                metrics::INFERENCE_ERRORS.inc();
                let timeout_chunk = ChatCompletionChunk {
                    id: req_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: now,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(format!(
                                "Error: request timed out after {}s",
                                request_timeout.as_secs()
                            )),
                        },
                        finish_reason: Some("error".to_string()),
                    }],
                };
                if let Ok(json) = serde_json::to_string(&timeout_chunk) {
                    let _ = try_send(&tx, json);
                }
                let _ = try_send(&tx, "[DONE]".to_string());
                return false;
            }

            token_count += 1;

            // Track degraded state
            if tok.degraded {
                had_degraded = true;
            }

            // NOTE: Per-token BPE decode can produce garbled output for multi-byte
            // UTF-8 characters that span multiple BPE tokens. A future improvement
            // would use a token accumulator buffer that collects tokens until a
            // complete UTF-8 codepoint boundary is reached before decoding, similar
            // to how HuggingFace text-generation-inference handles this.
            let text = state
                .tokenizer
                .decode(&[tok.token_id], true)
                .unwrap_or_else(|err| {
                    metrics::INFERENCE_ERRORS.inc();
                    eprintln!("ERROR: tokenizer decode failed in stream: {}", err);
                    String::new()
                });

            let chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: if tok.finish_reason.is_some() {
                            None
                        } else {
                            Some(text)
                        },
                    },
                    finish_reason: tok.finish_reason.clone(),
                }],
            };

            match serde_json::to_string(&chunk) {
                Ok(json) => {
                    let sent = try_send(&tx, json);
                    if !sent {
                        backpressured = true;
                    }
                    sent
                }
                Err(err) => {
                    metrics::INFERENCE_ERRORS.inc();
                    eprintln!("ERROR: Failed to serialize streaming chunk: {}", err);
                    false
                }
            }
        });
        let degraded_reason = if had_degraded {
            engine.last_forward_error_message()
        } else {
            None
        };

        // Log if any degraded outputs occurred
        if had_degraded {
            if let Some(reason) = degraded_reason.as_deref() {
                eprintln!(
                    "WARNING: Streaming response had degraded outputs: {}",
                    reason
                );
            } else {
                eprintln!("WARNING: Streaming response had degraded outputs (LoopLM errors)");
            }
        }
        if timed_out {
            return;
        }
        if backpressured {
            return;
        }

        // Track generated tokens
        metrics::TOKENS_GENERATED.inc_by(token_count as f64);

        // Record tokens per second
        let gen_elapsed = started.elapsed();
        if gen_elapsed.as_secs_f64() > 0.0 && token_count > 0 {
            let tps = token_count as f64 / gen_elapsed.as_secs_f64();
            metrics::TOKENS_PER_SECOND.set(tps);
        }

        // Send [DONE]
        let _ = try_send(&tx, "[DONE]".to_string());
    });

    Sse::new(ReceiverStream::new(rx))
}

const CHAT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanochat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
header{background:#16213e;padding:12px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #0f3460}
header h1{font-size:18px;color:#e94560;font-weight:600}
#model-name{font-size:13px;color:#888;margin-left:auto}
#messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:10px 14px;border-radius:12px;line-height:1.5;font-size:14px;white-space:pre-wrap;word-break:break-word}
.msg.user{align-self:flex-end;background:#0f3460;color:#e0e0e0;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#16213e;color:#e0e0e0;border-bottom-left-radius:4px}
.msg.system{align-self:center;color:#888;font-size:12px;font-style:italic}
#input-area{background:#16213e;padding:12px 20px;border-top:1px solid #0f3460;display:flex;gap:10px;align-items:flex-end}
#prompt{flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;border-radius:8px;padding:10px 14px;font-size:14px;font-family:inherit;resize:none;min-height:44px;max-height:120px;outline:none}
#prompt:focus{border-color:#e94560}
#send-btn{background:#e94560;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:600;white-space:nowrap}
#send-btn:hover{background:#c73e54}
#send-btn:disabled{background:#555;cursor:not-allowed}
#controls{display:flex;gap:16px;padding:6px 20px;background:#16213e;font-size:12px;color:#888;align-items:center}
#controls label{display:flex;align-items:center;gap:4px}
#controls input[type=range]{width:80px;accent-color:#e94560}
#controls select,#controls input[type=number]{background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;border-radius:4px;padding:2px 6px;font-size:12px}
.typing{opacity:0.6}
</style>
</head>
<body>
<header>
<h1>nanochat</h1>
<span id="model-name">loading...</span>
</header>
<div id="messages"></div>
<div id="controls">
<label>Temp <input type="range" id="temp" min="0" max="2" step="0.1" value="0.8"><span id="temp-val">0.8</span></label>
<label>Max tokens <input type="number" id="max-tok" value="256" min="1" max="2048" style="width:60px"></label>
<label>Top-p <input type="range" id="top-p" min="0" max="1" step="0.05" value="0.9"><span id="top-p-val">0.9</span></label>
</div>
<div id="input-area">
<textarea id="prompt" rows="1" placeholder="Type a message... (Enter to send, Shift+Enter for newline)"></textarea>
<button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
const msgs=document.getElementById('messages');
const prompt=document.getElementById('prompt');
const sendBtn=document.getElementById('send-btn');
const tempSlider=document.getElementById('temp');
const tempVal=document.getElementById('temp-val');
const topPSlider=document.getElementById('top-p');
const topPVal=document.getElementById('top-p-val');
let chatHistory=[];
let generating=false;
let apiKey=null;

tempSlider.oninput=()=>tempVal.textContent=tempSlider.value;
topPSlider.oninput=()=>topPVal.textContent=topPSlider.value;

function apiHeaders(){
  const h={'Content-Type':'application/json'};
  if(apiKey)h['Authorization']='Bearer '+apiKey;
  return h;
}

async function initModels(){
  const h=apiKey?{Authorization:'Bearer '+apiKey}:{};
  const r=await fetch('/v1/models',{headers:h});
  if(r.status===401){
    const k=window.prompt('This server requires an API key:');
    if(k){apiKey=k.trim();return initModels();}
    document.getElementById('model-name').textContent='(auth required)';
    return;
  }
  const d=await r.json();
  document.getElementById('model-name').textContent=d.data?.[0]?.id||'unknown';
}
initModels().catch(()=>{document.getElementById('model-name').textContent='(connection error)';});

prompt.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}
});
prompt.addEventListener('input',()=>{
  prompt.style.height='auto';
  prompt.style.height=Math.min(prompt.scrollHeight,120)+'px';
});

function addMsg(role,content,cls){
  const d=document.createElement('div');
  d.className='msg '+cls;
  d.textContent=content;
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
  return d;
}

async function sendMessage(){
  if(generating)return;
  const text=prompt.value.trim();
  if(!text)return;
  prompt.value='';
  prompt.style.height='auto';

  addMsg('user',text,'user');
  chatHistory.push({role:'user',content:text});

  const assistantDiv=addMsg('assistant','','assistant typing');
  generating=true;
  sendBtn.disabled=true;

  try{
    const res=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:apiHeaders(),
      body:JSON.stringify({
        messages:chatHistory,
        temperature:parseFloat(tempSlider.value),
        top_p:parseFloat(topPSlider.value),
        max_tokens:parseInt(document.getElementById('max-tok').value)||256,
        stream:true
      })
    });

    const reader=res.body.getReader();
    const decoder=new TextDecoder();
    let fullText='';
    let buf='';

    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n');
      buf=lines.pop()||'';
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const data=line.slice(6).trim();
        if(data==='[DONE]')break;
        try{
          const chunk=JSON.parse(data);
          const c=chunk.choices?.[0]?.delta?.content;
          if(c){fullText+=c;assistantDiv.textContent=fullText;}
        }catch{}
      }
    }

    assistantDiv.classList.remove('typing');
    if(!fullText)fullText='(empty response)';
    assistantDiv.textContent=fullText;
    chatHistory.push({role:'assistant',content:fullText});
  }catch(err){
    assistantDiv.textContent='Error: '+err.message;
    assistantDiv.classList.remove('typing');
  }

  generating=false;
  sendBtn.disabled=false;
  msgs.scrollTop=msgs.scrollHeight;
  prompt.focus();
}
</script>
</body>
</html>"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use nanochat_model::config::ModelConfig;
    use tower::ServiceExt;

    /// Build a byte-level BPE tokenizer with 256 single-byte tokens (no merges).
    /// Uses GPT-2's byte_to_unicode mapping for ByteLevel pre/post-processing.
    fn make_test_tokenizer() -> tokenizers::Tokenizer {
        let mut vocab = serde_json::Map::new();
        let mut n = 256u32;
        for b in 0u32..256 {
            let is_direct =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let ch = if is_direct {
                char::from_u32(b).unwrap()
            } else {
                let c = char::from_u32(n).unwrap();
                n += 1;
                c
            };
            vocab.insert(ch.to_string(), serde_json::json!(b));
        }
        let json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            }
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        tokenizers::Tokenizer::from_bytes(&bytes).unwrap()
    }

    fn make_test_state() -> Arc<AppState> {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let max_seq_len = config.max_seq_len;
        let engine = EngineHandle::Standard(InferenceEngine::new_random(config));
        let tokenizer = make_test_tokenizer();

        Arc::new(AppState {
            engines: vec![std::sync::Mutex::new(engine)],
            next_engine: AtomicUsize::new(0),
            tokenizer,
            model_name: "nanochat-test".to_string(),
            vocab_size: 256,
            max_seq_len,
            chat_template: ChatTemplate::RoleTagged,
            api_key: None,
            request_timeout: std::time::Duration::from_secs(30),
            cors_allowed_origin: None,
            stream_channel_capacity: 64,
        })
    }

    fn make_auth_state(api_key: &str) -> Arc<AppState> {
        let mut state = make_test_state();
        Arc::get_mut(&mut state)
            .expect("state should be uniquely owned in tests")
            .api_key = Some(api_key.to_string());
        state
    }

    #[tokio::test]
    async fn test_chat_ui() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 100000)
            .await
            .unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        assert!(html.contains("nanochat"));
        assert!(html.contains("/v1/chat/completions"));
        assert!(html.contains("<!DOCTYPE html>"));
    }

    #[tokio::test]
    async fn test_health() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_models() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 10000).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"][0]["id"], "nanochat-test");
    }

    #[tokio::test]
    async fn test_auth_rejects_missing_api_key() {
        let state = make_auth_state("secret-key");
        let app = build_router(state);

        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_accepts_bearer_api_key() {
        let state = make_auth_state("secret-key");
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::get("/v1/models")
                    .header("authorization", "Bearer secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let state = make_test_state();
        let app = build_router(state);

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0.0
        });

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 100000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["choices"][0]["message"]["content"].is_string());
        assert!(json["usage"]["total_tokens"].is_number());
    }

    #[tokio::test]
    async fn test_chat_completions_streaming() {
        let state = make_test_state();
        let app = build_router(state);

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": true
        });

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/event-stream"),
            "Expected text/event-stream, got: {content_type}"
        );

        let body = axum::body::to_bytes(resp.into_body(), 100000)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        // SSE format: lines starting with "data: "
        let data_lines: Vec<&str> = text.lines().filter(|l| l.starts_with("data: ")).collect();
        assert!(
            !data_lines.is_empty(),
            "Expected SSE data lines, got none. Full body:\n{text}"
        );

        // First data chunk should have role
        let first: serde_json::Value =
            serde_json::from_str(data_lines[0].strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

        // Last data line should be [DONE]
        let last_data = data_lines.last().unwrap();
        assert_eq!(
            last_data.strip_prefix("data: ").unwrap().trim(),
            "[DONE]",
            "Expected [DONE] terminator"
        );

        // Intermediate chunks should be valid JSON with content deltas
        let mut got_content = false;
        for line in &data_lines[1..data_lines.len() - 1] {
            let json_str = line.strip_prefix("data: ").unwrap();
            let chunk: serde_json::Value = serde_json::from_str(json_str).unwrap();
            assert_eq!(chunk["object"], "chat.completion.chunk");
            if chunk["choices"][0]["delta"]["content"].is_string() {
                got_content = true;
            }
        }
        assert!(got_content, "Expected at least one content delta chunk");
    }

    #[tokio::test]
    async fn test_malformed_json_body() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("not valid json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Axum returns 422 for deserialization failures
        assert!(
            resp.status().is_client_error(),
            "Expected 4xx for malformed JSON, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_missing_messages_field() {
        let state = make_test_state();
        let app = build_router(state);

        let body = serde_json::json!({
            "temperature": 0.5
        });

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error(),
            "Expected 4xx for missing messages, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_empty_body() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error(),
            "Expected 4xx for empty body, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_wrong_content_type() {
        let state = make_test_state();
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "text/plain")
                    .body(Body::from(
                        "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}",
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error(),
            "Expected 4xx for wrong content-type, got {}",
            resp.status()
        );
    }
}
