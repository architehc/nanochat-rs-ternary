//! Axum HTTP server with OpenAI-compatible endpoints.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
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
    pub tokenizer: tokenizers::Tokenizer,
    pub model_name: String,
    pub vocab_size: u32,
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(chat_ui))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn chat_ui() -> Html<&'static str> {
    Html(CHAT_HTML)
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
            let prompt_ids = state.tokenizer.encode(prompt.as_str(), false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            let prompt_len = prompt_ids.len();
            let vs = state.vocab_size;
            let token_ids: Vec<u32> = prompt_ids.into_iter().map(|t| t % vs).collect();

            let mut engine = state.engine.lock().unwrap();
            let output_ids = engine.generate(&token_ids, &params);
            let output_len = output_ids.len();

            let text = state.tokenizer.decode(&output_ids, true).unwrap_or_default();

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
        let prompt_ids = state.tokenizer.encode(prompt.as_str(), false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default();
        let vs = state.vocab_size;
        let token_ids: Vec<u32> = prompt_ids.into_iter().map(|t| t % vs).collect();

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
            let text = state.tokenizer.decode(&[tok.token_id], true).unwrap_or_default();

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

tempSlider.oninput=()=>tempVal.textContent=tempSlider.value;
topPSlider.oninput=()=>topPVal.textContent=topPSlider.value;

fetch('/v1/models').then(r=>r.json()).then(d=>{
  document.getElementById('model-name').textContent=d.data?.[0]?.id||'unknown';
}).catch(()=>{});

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
      headers:{'Content-Type':'application/json'},
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
            let is_direct = (33..=126).contains(&b)
                || (161..=172).contains(&b)
                || (174..=255).contains(&b);
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
        let engine = InferenceEngine::new_random(config);
        let tokenizer = make_test_tokenizer();

        Arc::new(AppState {
            engine: std::sync::Mutex::new(engine),
            tokenizer,
            model_name: "nanochat-test".to_string(),
            vocab_size: 256,
        })
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
        let data_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("data: "))
            .collect();
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
                    .body(Body::from("{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}"))
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
