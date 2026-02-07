//! Axum HTTP server with OpenAI-compatible endpoints.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::api::*;
use crate::engine::InferenceEngine;

/// Shared application state.
pub struct AppState {
    pub engine: std::sync::Mutex<InferenceEngine>,
    pub tokenizer: tiktoken_rs::CoreBPE,
    pub model_name: String,
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "nanochat"
        }]
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);

    if stream {
        stream_completion(state, req).await.into_response()
    } else {
        non_stream_completion(state, req).await.into_response()
    }
}

async fn non_stream_completion(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Json<ChatCompletionResponse> {
    let prompt = req.to_prompt_string();
    let params = req.to_sampling_params();
    let model_name = state.model_name.clone();

    let (generated_text, prompt_tokens, completion_tokens) =
        tokio::task::spawn_blocking(move || {
            let prompt_ids = state.tokenizer.encode_ordinary(&prompt);
            let prompt_len = prompt_ids.len();
            let token_ids: Vec<u32> = prompt_ids.into_iter().map(|t| t as u32).collect();

            let mut engine = state.engine.lock().unwrap();
            let output_ids = engine.generate(&token_ids, &params);
            let output_len = output_ids.len();

            let text = state.tokenizer.decode(output_ids).unwrap_or_default();

            (text, prompt_len, output_len)
        })
        .await
        .unwrap();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

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
            finish_reason: "stop".to_string(),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    })
}

async fn stream_completion(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let prompt = req.to_prompt_string();
    let params = req.to_sampling_params();
    let model_name = state.model_name.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    let req_id = request_id.clone();
    let model = model_name.clone();
    tokio::task::spawn_blocking(move || {
        let prompt_ids = state.tokenizer.encode_ordinary(&prompt);
        let token_ids: Vec<u32> = prompt_ids.into_iter().map(|t| t as u32).collect();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

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
        let json = serde_json::to_string(&initial).unwrap();
        let _ = tx.blocking_send(Ok(Event::default().data(json)));

        let mut engine = state.engine.lock().unwrap();

        engine.generate_streaming(&token_ids, &params, |tok| {
            let text = state.tokenizer.decode(vec![tok.token_id]).unwrap_or_default();

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

            let json = serde_json::to_string(&chunk).unwrap();
            tx.blocking_send(Ok(Event::default().data(json))).is_ok()
        });

        // Send [DONE]
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use nanochat_model::config::ModelConfig;
    use tower::ServiceExt;

    fn make_test_state() -> Arc<AppState> {
        let config = ModelConfig {
            dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            ffn_mult: 2.667,
            vocab_size: 50257, // Must match tiktoken GPT-2 vocab
            max_seq_len: 64,
            group_size: 128,
            mhc_n_streams: 2,
            rope_theta: 10000.0,
            n_experts: None,
            n_active_experts: None,
            deltanet_ratio: None,
            weight_tied: false,
        };
        let engine = InferenceEngine::new_random(config);

        let tokenizer = tiktoken_rs::r50k_base().unwrap();

        Arc::new(AppState {
            engine: std::sync::Mutex::new(engine),
            tokenizer,
            model_name: "nanochat-test".to_string(),
        })
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
        let body = axum::body::to_bytes(resp.into_body(), 10000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"][0]["id"], "nanochat-test");
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
}
