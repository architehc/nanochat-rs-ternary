//! Code generation evaluation framework for HumanEval and MBPP.
//!
//! Provides:
//! - Dataset loaders for HumanEval and MBPP
//! - Code execution sandbox
//! - Pass@k metric calculation
//! - Model comparison tools

pub mod datasets;
pub mod executor;
pub mod metrics;
pub mod eval;

pub use datasets::{HumanEvalDataset, MBPPDataset, CodeProblem};
pub use executor::{CodeExecutor, ExecutionResult};
pub use metrics::{EvalMetrics, PassAtK};
pub use eval::{EvaluationConfig, Evaluator, EvalReport};
