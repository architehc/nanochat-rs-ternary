//! Structured logging for training with tracing.
//!
//! Provides structured logging with JSON output, automatic warnings for
//! anomalies (low entropy, gradient explosion), and per-step metrics tracking.

use tracing::{debug, error, info, span, warn, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize structured logging.
///
/// Reads log level from RUST_LOG environment variable (defaults to "info").
/// Outputs JSON-formatted logs for production monitoring.
pub fn init_logging() {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // Default: info for our crates, warn for dependencies
                "info,nanochat_train=info,nanochat_model=info,nanochat_core=info".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    info!("Structured logging initialized");
}

/// Initialize simple console logging (for examples/debugging).
pub fn init_console_logging() {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,nanochat_train=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().pretty())
        .init();
}

/// Training metrics for structured logging.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total loss (CE + entropy regularization).
    pub loss: f64,
    /// Cross-entropy loss.
    pub ce_loss: f64,
    /// Entropy regularization term.
    pub entropy: f64,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Gradient norm (L2).
    pub grad_norm: f64,
    /// Throughput in tokens per second.
    pub tokens_per_sec: f64,
    /// Optional: mHC composite gain.
    pub mhc_gain: Option<f64>,
}

impl TrainingMetrics {
    /// Create metrics from training step outputs.
    pub fn new(
        loss: f64,
        ce_loss: f64,
        entropy: f64,
        learning_rate: f64,
        grad_norm: f64,
        tokens_per_sec: f64,
    ) -> Self {
        Self {
            loss,
            ce_loss,
            entropy,
            learning_rate,
            grad_norm,
            tokens_per_sec,
            mhc_gain: None,
        }
    }

    /// Set mHC composite gain.
    pub fn with_mhc_gain(mut self, gain: f64) -> Self {
        self.mhc_gain = Some(gain);
        self
    }
}

/// Log a training step with structured metrics.
///
/// Automatically emits warnings for:
/// - Low entropy (< 5.0): potential model collapse
/// - High gradient norm (> 10.0): potential instability
/// - Divergence (NaN/infinite loss)
pub fn log_training_step(step: usize, metrics: &TrainingMetrics) {
    let span = span!(Level::INFO, "training_step", step = step);
    let _enter = span.enter();

    // Check for divergence
    if !metrics.loss.is_finite() || !metrics.ce_loss.is_finite() {
        error!(
            loss = metrics.loss,
            ce_loss = metrics.ce_loss,
            step = step,
            "Training diverged! NaN or infinite loss detected"
        );
        return;
    }

    // Log metrics with structured fields
    info!(
        loss = metrics.loss,
        ce_loss = metrics.ce_loss,
        entropy = metrics.entropy,
        lr = metrics.learning_rate,
        grad_norm = metrics.grad_norm,
        tokens_per_sec = metrics.tokens_per_sec,
        mhc_gain = metrics.mhc_gain,
        "Training step completed"
    );

    // Automatic warnings for anomalies
    if metrics.entropy < 5.0 {
        warn!(
            entropy = metrics.entropy,
            step = step,
            threshold = 5.0,
            "Low entropy detected - model may be collapsing. Consider: \
             (1) increasing entropy weight, (2) reducing learning rate, \
             (3) checking for mode collapse in outputs"
        );
    }

    if metrics.grad_norm > 10.0 {
        warn!(
            grad_norm = metrics.grad_norm,
            step = step,
            threshold = 10.0,
            "High gradient norm detected - potential instability. Consider: \
             (1) reducing learning rate, (2) increasing gradient clipping, \
             (3) checking for NaN in activations"
        );
    }

    // Debug-level additional info
    debug!(
        step = step,
        loss_breakdown = format!("CE: {:.4}, Entropy: {:.4}", metrics.ce_loss, metrics.entropy),
        "Training diagnostics"
    );
}

/// Log checkpoint save event.
pub fn log_checkpoint_save(step: usize, path: &str, loss: f64) {
    info!(
        step = step,
        path = path,
        loss = loss,
        event = "checkpoint_saved",
        "Checkpoint saved successfully"
    );
}

/// Log evaluation results.
pub fn log_evaluation(step: usize, eval_loss: f64, eval_ppl: f64) {
    info!(
        step = step,
        eval_loss = eval_loss,
        eval_perplexity = eval_ppl,
        event = "evaluation",
        "Evaluation completed"
    );
}

/// Log model collapse detection.
pub fn log_model_collapse(step: usize, entropy: f64) {
    error!(
        step = step,
        entropy = entropy,
        threshold = 5.0,
        event = "model_collapse",
        "Model collapse detected! Entropy fell below threshold. Training should stop."
    );
}

/// Log gradient explosion.
pub fn log_gradient_explosion(step: usize, grad_norm: f64) {
    error!(
        step = step,
        grad_norm = grad_norm,
        threshold = 10.0,
        event = "gradient_explosion",
        "Gradient explosion detected! Training unstable."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = TrainingMetrics::new(2.5, 2.3, 6.5, 0.001, 1.2, 5000.0)
            .with_mhc_gain(0.98);

        assert_eq!(metrics.loss, 2.5);
        assert_eq!(metrics.entropy, 6.5);
        assert_eq!(metrics.mhc_gain, Some(0.98));
    }

    #[test]
    fn test_logging_does_not_panic() {
        // Just ensure logging functions don't panic
        let metrics = TrainingMetrics::new(2.5, 2.3, 3.0, 0.001, 15.0, 5000.0);

        // These should emit warnings internally but not panic
        log_training_step(100, &metrics);
        log_checkpoint_save(100, "/tmp/checkpoint", 2.5);
        log_evaluation(100, 2.3, 10.0);
        log_model_collapse(100, 3.0);
        log_gradient_explosion(100, 15.0);
    }
}
