//! OpenAI-compatible API types for /v1/chat/completions.

use crate::engine::SamplingParams;
use serde::{Deserialize, Serialize};

pub const MAX_MESSAGES: usize = 256;
pub const MAX_MESSAGE_CHARS: usize = 16 * 1024;
pub const MAX_PROMPT_CHARS: usize = 256 * 1024;

/// Prompt template used to flatten chat messages into a model prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<role>: <content>\n...assistant: `
    RoleTagged,
    /// ChatML-style framing: `<|im_start|>role\ncontent<|im_end|>\n`
    ChatMl,
}

impl ChatTemplate {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "role" | "role_tagged" | "role-tagged" | "default" => Some(Self::RoleTagged),
            "chatml" | "chat_ml" | "chat-ml" => Some(Self::ChatMl),
            _ => None,
        }
    }
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self::RoleTagged
    }
}

/// Chat message role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Chat completion request (OpenAI-compatible subset).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl ChatCompletionRequest {
    pub fn to_sampling_params(&self) -> SamplingParams {
        let temperature = self.temperature.unwrap_or(1.0).clamp(0.0, 2.0);
        let top_p = self.top_p.unwrap_or(0.9).clamp(0.0, 1.0);
        let top_k = self.top_k.unwrap_or(50).clamp(1, 200);
        let max_tokens = self.max_tokens.unwrap_or(256).clamp(1, 4096);

        SamplingParams {
            temperature,
            top_p,
            top_k,
            max_tokens,
            seed: self.seed,
        }
    }

    pub fn validate_messages(&self) -> Result<(), String> {
        if self.messages.is_empty() {
            return Err("messages must contain at least one item".to_string());
        }
        if self.messages.len() > MAX_MESSAGES {
            return Err(format!(
                "messages exceeds limit ({} > {})",
                self.messages.len(),
                MAX_MESSAGES
            ));
        }
        for (idx, msg) in self.messages.iter().enumerate() {
            let len = msg.content.chars().count();
            if len > MAX_MESSAGE_CHARS {
                return Err(format!(
                    "messages[{}].content exceeds limit ({} > {})",
                    idx, len, MAX_MESSAGE_CHARS
                ));
            }
        }
        Ok(())
    }

    /// Convert chat messages to a flat prompt string.
    /// Simple format: "<role>: <content>\n" per message.
    pub fn to_prompt_string(&self) -> Result<String, String> {
        self.to_prompt_string_with_template(ChatTemplate::default())
    }

    pub fn to_prompt_string_with_template(&self, template: ChatTemplate) -> Result<String, String> {
        self.validate_messages()?;

        let mut prompt = String::new();
        for (idx, msg) in self.messages.iter().enumerate() {
            match template {
                ChatTemplate::RoleTagged => {
                    prompt.push_str(msg.role.as_str());
                    prompt.push_str(": ");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                ChatTemplate::ChatMl => {
                    prompt.push_str("<|im_start|>");
                    prompt.push_str(msg.role.as_str());
                    prompt.push('\n');
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
            }
            if prompt.chars().count() > MAX_PROMPT_CHARS {
                return Err(format!(
                    "prompt exceeds limit while processing message {} (>{})",
                    idx, MAX_PROMPT_CHARS
                ));
            }
        }
        match template {
            ChatTemplate::RoleTagged => prompt.push_str("assistant: "),
            ChatTemplate::ChatMl => prompt.push_str("<|im_start|>assistant\n"),
        }
        if prompt.chars().count() > MAX_PROMPT_CHARS {
            return Err(format!("prompt exceeds limit (>{})", MAX_PROMPT_CHARS));
        }
        Ok(prompt)
    }
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single choice in the response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage counts.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

// ============================================================
// Streaming types (SSE chunks)
// ============================================================

/// Streaming chunk response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant] {
            let s = role.as_str();
            let parsed = Role::parse(s).unwrap();
            assert_eq!(parsed, role);
        }
    }

    #[test]
    fn test_role_from_str_invalid() {
        assert!(Role::parse("unknown").is_none());
    }

    #[test]
    fn test_role_serde() {
        let json = serde_json::to_string(&Role::User).unwrap();
        assert_eq!(json, "\"user\"");
        let parsed: Role = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(parsed, Role::Assistant);
    }

    #[test]
    fn test_chat_request_deserialize() {
        let json = r#"{
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn test_chat_request_to_prompt() {
        let req = ChatCompletionRequest {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: "You are helpful.".to_string(),
                },
                ChatMessage {
                    role: Role::User,
                    content: "Hello!".to_string(),
                },
            ],
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            seed: None,
        };

        let prompt = req.to_prompt_string().unwrap();
        assert!(prompt.contains("system: You are helpful."));
        assert!(prompt.contains("user: Hello!"));
        assert!(prompt.ends_with("assistant: "));
    }

    #[test]
    fn test_chat_request_to_prompt_chatml() {
        let req = ChatCompletionRequest {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: "You are helpful.".to_string(),
                },
                ChatMessage {
                    role: Role::User,
                    content: "Hello!".to_string(),
                },
            ],
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            seed: None,
        };

        let prompt = req
            .to_prompt_string_with_template(ChatTemplate::ChatMl)
            .unwrap();
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nHello!<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chat_template_parse() {
        assert_eq!(
            ChatTemplate::parse("default"),
            Some(ChatTemplate::RoleTagged)
        );
        assert_eq!(ChatTemplate::parse("chatml"), Some(ChatTemplate::ChatMl));
        assert_eq!(ChatTemplate::parse("unknown"), None);
    }

    #[test]
    fn test_chat_request_to_prompt_rejects_empty_messages() {
        let req = ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            seed: None,
        };

        assert!(req.to_prompt_string().is_err());
    }

    #[test]
    fn test_chat_request_to_prompt_rejects_oversize_message() {
        let req = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: "x".repeat(MAX_MESSAGE_CHARS + 1),
            }],
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            seed: None,
        };

        assert!(req.to_prompt_string().is_err());
    }

    #[test]
    fn test_chat_request_to_sampling_params() {
        let req = ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: Some(0.5),
            top_p: Some(0.8),
            top_k: Some(40),
            max_tokens: Some(100),
            stream: None,
            seed: None,
        };

        let params = req.to_sampling_params();
        assert!((params.temperature - 0.5).abs() < 1e-6);
        assert!((params.top_p - 0.8).abs() < 1e-6);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.max_tokens, 100);
    }

    #[test]
    fn test_chat_request_defaults() {
        let req = ChatCompletionRequest {
            messages: vec![],
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            seed: None,
        };

        let params = req.to_sampling_params();
        assert!((params.temperature - 1.0).abs() < 1e-6);
        assert!((params.top_p - 0.9).abs() < 1e-6);
        assert_eq!(params.max_tokens, 256);
    }

    #[test]
    fn test_usage() {
        let u = Usage::new(10, 20);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn test_chat_completion_response_serialize() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "nanochat-125m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage::new(5, 3),
        };

        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chatcmpl-test"));
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_streaming_chunk_serialize() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "nanochat-125m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("Hello"));
        // role should be absent when None
        assert!(!json.contains("role"));
    }

    #[test]
    fn test_streaming_chunk_with_role() {
        let chunk = ChatCompletionChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some(Role::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(!json.contains("content"));
    }
}
