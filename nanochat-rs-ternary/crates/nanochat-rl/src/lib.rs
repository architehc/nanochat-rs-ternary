//! nanochat-rl: Reinforcement Learning for Rust Code Generation
//!
//! This crate implements GRPO/GSPO (Group Relative Policy Optimization) for training
//! Rust code generation models with compiler feedback and AST analysis.
//!
//! ## Architecture
//!
//! 1. **Compiler Feedback**: Integrates with rustc to get compilation results
//! 2. **AST Analysis**: Uses syn crate to parse and analyze code structure
//! 3. **Reward Function**: Combines compiler success, AST quality metrics
//! 4. **GRPO Algorithm**: Group-based policy optimization with relative rewards
//! 5. **External Evaluation**: Optional Qwen3 coder endpoint for additional feedback

pub mod ast_analysis;
pub mod compiler;
pub mod grpo;
pub mod maxrl;
pub mod qwen;
pub mod reward;
pub mod trainer;
pub mod training_free_grpo;

pub use ast_analysis::{analyze_ast, AstMetrics};
pub use compiler::{CompileResult, CompilerFeedback};
pub use grpo::{GrpoConfig, GrpoTrainer};
pub use maxrl::{MaxRLConfig, MaxRLStats, MaxRLTrainer};
pub use reward::{compute_reward, RewardConfig};
pub use trainer::RLTrainer;
pub use training_free_grpo::{Experience, GRPOConfig, GRPOStats, TrainingFreeGRPO};

/// Configuration for RL training
#[derive(Debug, Clone)]
pub struct RLConfig {
    /// Base model checkpoint to start from
    pub base_checkpoint: String,

    /// GRPO algorithm configuration
    pub grpo: GrpoConfig,

    /// Reward function configuration
    pub reward: RewardConfig,

    /// Number of code samples to generate per prompt
    pub n_samples: usize,

    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Temperature for sampling
    pub temperature: f64,

    /// Number of RL iterations
    pub n_iterations: usize,

    /// Batch size for policy updates
    pub batch_size: usize,

    /// Device for training (cpu, cuda:0, etc.)
    pub device: String,

    /// Optional Qwen3 endpoint URL
    pub qwen_endpoint: Option<String>,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            base_checkpoint: "checkpoints/rust-6hour/step_2000".to_string(),
            grpo: GrpoConfig::default(),
            reward: RewardConfig::default(),
            n_samples: 4,
            max_tokens: 256,
            temperature: 0.8,
            n_iterations: 1000,
            batch_size: 2,
            device: "cuda:0".to_string(),
            qwen_endpoint: None,
        }
    }
}
