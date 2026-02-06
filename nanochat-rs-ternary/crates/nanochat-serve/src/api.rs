//! OpenAI-compatible API types for /v1/chat/completions.
//!
//! Minimal implementation — just the request/response types and conversion logic.
//! The actual HTTP server would use Axum or similar.

use crate::engine::SamplingParams;

/// Chat message role.
#[derive(Debug, Clone, PartialEq)]
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

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// A single chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Chat completion request (OpenAI-compatible subset).
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl ChatCompletionRequest {
    pub fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(0.9),
            max_tokens: self.max_tokens.unwrap_or(256),
            ..Default::default()
        }
    }

    /// Convert chat messages to a flat prompt string.
    /// Simple format: "<role>: <content>\n" per message.
    pub fn to_prompt_string(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            prompt.push_str(msg.role.as_str());
            prompt.push_str(": ");
            prompt.push_str(&msg.content);
            prompt.push('\n');
        }
        prompt.push_str("assistant: ");
        prompt
    }
}

/// Chat completion response.
#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// A single choice in the response.
#[derive(Debug, Clone)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage counts.
#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant] {
            let s = role.as_str();
            let parsed = Role::from_str(s).unwrap();
            assert_eq!(parsed, role);
        }
    }

    #[test]
    fn test_role_from_str_invalid() {
        assert!(Role::from_str("unknown").is_none());
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
            temperature: None,
            top_p: None,
            max_tokens: None,
        };

        let prompt = req.to_prompt_string();
        assert!(prompt.contains("system: You are helpful."));
        assert!(prompt.contains("user: Hello!"));
        assert!(prompt.ends_with("assistant: "));
    }

    #[test]
    fn test_chat_request_to_sampling_params() {
        let req = ChatCompletionRequest {
            messages: vec![],
            temperature: Some(0.5),
            top_p: Some(0.8),
            max_tokens: Some(100),
        };

        let params = req.to_sampling_params();
        assert!((params.temperature - 0.5).abs() < 1e-6);
        assert!((params.top_p - 0.8).abs() < 1e-6);
        assert_eq!(params.max_tokens, 100);
    }

    #[test]
    fn test_chat_request_defaults() {
        let req = ChatCompletionRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
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
    fn test_chat_completion_response() {
        let resp = ChatCompletionResponse {
            id: "test-123".to_string(),
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

        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "Hello!");
        assert_eq!(resp.usage.total_tokens, 8);
    }
}
