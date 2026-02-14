//! Code generation evaluation framework for HumanEval and MBPP.
//!
//! Provides:
//! - Dataset loaders for HumanEval and MBPP
//! - Code execution sandbox
//! - Pass@k metric calculation
//! - Model comparison tools

pub mod datasets;
pub mod eval;
pub mod executor;
pub mod metrics;

pub use datasets::{CodeProblem, HumanEvalDataset, MBPPDataset};
pub use eval::{EvalReport, EvaluationConfig, Evaluator};
pub use executor::{CodeExecutor, ExecutionResult};
pub use metrics::{EvalMetrics, PassAtK};
