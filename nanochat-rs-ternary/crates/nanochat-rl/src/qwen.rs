//! Qwen3 Coder endpoint integration for external code evaluation
//!
//! Provides an additional signal for code quality by querying a larger,
//! more capable model (Qwen3 Coder) to evaluate generated code.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Qwen3 evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenEvaluation {
    /// Quality score (0-10)
    pub quality_score: f64,

    /// Correctness score (0-10)
    pub correctness_score: f64,

    /// Idiomatic Rust score (0-10)
    pub idiomaticity_score: f64,

    /// Overall recommendation
    pub recommendation: String,

    /// Detailed feedback
    pub feedback: String,
}

/// Qwen3 API client
pub struct QwenClient {
    client: Client,
    endpoint: String,
    api_key: Option<String>,
}

impl QwenClient {
    /// Create a new Qwen3 client
    pub fn new(endpoint: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            api_key,
        }
    }

    /// Evaluate a code sample using Qwen3 Coder
    pub async fn evaluate_code(&self, code: &str, context: &str) -> Result<QwenEvaluation> {
        let prompt = format!(
            r#"You are a Rust code quality evaluator. Evaluate the following Rust code and provide scores.

Context: {}

Code:
```rust
{}
```

Provide a JSON response with the following structure:
{{
  "quality_score": <0-10>,
  "correctness_score": <0-10>,
  "idiomaticity_score": <0-10>,
  "recommendation": "<accept/revise/reject>",
  "feedback": "<detailed feedback>"
}}

Scoring guidelines:
- quality_score: Overall code quality (readability, maintainability, organization)
- correctness_score: Likely correctness (logic, error handling, edge cases)
- idiomaticity_score: How idiomatic and Rusty the code is
- recommendation: accept (good), revise (needs work), reject (fundamentally flawed)
- feedback: Specific suggestions for improvement

Respond with ONLY the JSON, no additional text."#,
            context, code
        );

        #[derive(Serialize)]
        struct QwenRequest {
            model: String,
            messages: Vec<Message>,
            temperature: f64,
            max_tokens: usize,
        }

        #[derive(Serialize)]
        struct Message {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct QwenResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }

        #[derive(Deserialize)]
        struct ResponseMessage {
            content: String,
        }

        let request = QwenRequest {
            model: "qwen-coder-plus".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt,
            }],
            temperature: 0.3,
            max_tokens: 1000,
        };

        let mut req = self.client.post(&self.endpoint).json(&request);

        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = req
            .send()
            .await
            .context("Failed to send request to Qwen3")?
            .json::<QwenResponse>()
            .await
            .context("Failed to parse Qwen3 response")?;

        let content = &response.choices[0].message.content;

        // Try to parse JSON from response
        // The model might wrap it in markdown code blocks, so extract JSON
        let json_str = if content.contains("```json") {
            content
                .split("```json")
                .nth(1)
                .and_then(|s| s.split("```").next())
                .unwrap_or(content)
                .trim()
        } else if content.contains("```") {
            content.split("```").nth(1).unwrap_or(content).trim()
        } else {
            content.trim()
        };

        serde_json::from_str(json_str)
            .context("Failed to parse evaluation JSON from Qwen3 response")
    }

    /// Evaluate multiple code samples in parallel
    pub async fn evaluate_batch(
        &self,
        samples: &[(String, String)],
    ) -> Result<Vec<QwenEvaluation>> {
        let mut tasks = Vec::new();

        for (code, context) in samples {
            let client = self.clone();
            let code = code.clone();
            let context = context.clone();

            tasks.push(tokio::spawn(async move {
                client.evaluate_code(&code, &context).await
            }));
        }

        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(eval)) => results.push(eval),
                Ok(Err(e)) => {
                    eprintln!("Qwen3 evaluation failed: {}", e);
                    // Return default evaluation on error
                    results.push(QwenEvaluation::default());
                }
                Err(e) => {
                    eprintln!("Task join error: {}", e);
                    results.push(QwenEvaluation::default());
                }
            }
        }

        Ok(results)
    }
}

impl Clone for QwenClient {
    fn clone(&self) -> Self {
        Self {
            client: Client::new(),
            endpoint: self.endpoint.clone(),
            api_key: self.api_key.clone(),
        }
    }
}

impl Default for QwenEvaluation {
    fn default() -> Self {
        Self {
            quality_score: 5.0,
            correctness_score: 5.0,
            idiomaticity_score: 5.0,
            recommendation: "unknown".to_string(),
            feedback: "Evaluation unavailable".to_string(),
        }
    }
}

/// Convert Qwen evaluation to a reward contribution
pub fn qwen_to_reward(eval: &QwenEvaluation, weight: f64) -> f64 {
    // Average the three scores and normalize to [0, 1]
    let avg_score = (eval.quality_score + eval.correctness_score + eval.idiomaticity_score) / 30.0;

    // Apply weight
    avg_score * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_to_reward() {
        let eval = QwenEvaluation {
            quality_score: 8.0,
            correctness_score: 9.0,
            idiomaticity_score: 7.0,
            recommendation: "accept".to_string(),
            feedback: "Good code".to_string(),
        };

        let reward = qwen_to_reward(&eval, 5.0);

        // Average is 24/30 = 0.8, reward = 0.8 * 5.0 = 4.0
        assert!((reward - 4.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_qwen_client_creation() {
        let client = QwenClient::new(
            "https://api.example.com/v1/chat/completions".to_string(),
            Some("test_key".to_string()),
        );

        assert_eq!(
            client.endpoint,
            "https://api.example.com/v1/chat/completions"
        );
    }
}
