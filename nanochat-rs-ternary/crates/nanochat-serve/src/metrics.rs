//! Prometheus metrics for inference server.
//!
//! Tracks:
//! - Request counts and latencies
//! - Active requests (gauge)
//! - Model load time
//! - Tokens generated
//!
//! Metrics are exposed via GET /metrics endpoint.

use lazy_static::lazy_static;
use prometheus::{
    register_counter, register_gauge, register_histogram, Counter, Encoder, Gauge, Histogram,
    Registry, TextEncoder,
};

lazy_static! {
    /// Global metrics registry.
    pub static ref REGISTRY: Registry = Registry::new();

    /// Total number of inference requests.
    pub static ref INFERENCE_REQUESTS: Counter = register_counter!(
        "nanochat_inference_requests_total",
        "Total number of inference requests"
    )
    .unwrap();

    /// Total number of failed requests.
    pub static ref INFERENCE_ERRORS: Counter = register_counter!(
        "nanochat_inference_errors_total",
        "Total number of failed inference requests"
    )
    .unwrap();

    /// Inference latency histogram in seconds.
    pub static ref INFERENCE_LATENCY: Histogram = register_histogram!(
        "nanochat_inference_latency_seconds",
        "Inference latency in seconds",
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .unwrap();

    /// Number of currently active requests.
    pub static ref ACTIVE_REQUESTS: Gauge = register_gauge!(
        "nanochat_active_requests",
        "Number of currently active inference requests"
    )
    .unwrap();

    /// Model load time in seconds.
    pub static ref MODEL_LOAD_TIME: Gauge = register_gauge!(
        "nanochat_model_load_time_seconds",
        "Time taken to load model in seconds"
    )
    .unwrap();

    /// Total tokens generated.
    pub static ref TOKENS_GENERATED: Counter = register_counter!(
        "nanochat_tokens_generated_total",
        "Total number of tokens generated"
    )
    .unwrap();

    /// Tokens per second (throughput).
    pub static ref TOKENS_PER_SECOND: Gauge = register_gauge!(
        "nanochat_tokens_per_second",
        "Current tokens per second throughput"
    )
    .unwrap();
}

/// Register all metrics with the global registry.
pub fn register_metrics() {
    REGISTRY
        .register(Box::new(INFERENCE_REQUESTS.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(INFERENCE_ERRORS.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(INFERENCE_LATENCY.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(ACTIVE_REQUESTS.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(MODEL_LOAD_TIME.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(TOKENS_GENERATED.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(TOKENS_PER_SECOND.clone()))
        .unwrap();
}

/// Render metrics in Prometheus text format.
pub fn render_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// RAII guard for tracking active requests.
///
/// Increments ACTIVE_REQUESTS on creation, decrements on drop.
pub struct ActiveRequestGuard;

impl ActiveRequestGuard {
    pub fn new() -> Self {
        ACTIVE_REQUESTS.inc();
        Self
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

impl Default for ActiveRequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for timing requests.
///
/// Records latency to INFERENCE_LATENCY histogram on drop.
pub struct LatencyTimer {
    start: std::time::Instant,
}

impl LatencyTimer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        INFERENCE_LATENCY.observe(elapsed);
    }
}

impl Default for LatencyTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registration() {
        // Ensure metrics can be registered without panic
        register_metrics();

        // Verify metrics are accessible
        INFERENCE_REQUESTS.inc();
        ACTIVE_REQUESTS.set(5.0);
        MODEL_LOAD_TIME.set(2.5);

        // Render metrics
        let output = render_metrics();
        assert!(output.contains("nanochat_inference_requests_total"));
        assert!(output.contains("nanochat_active_requests"));
    }

    #[test]
    fn test_active_request_guard() {
        let initial = ACTIVE_REQUESTS.get();
        {
            let _guard = ActiveRequestGuard::new();
            assert_eq!(ACTIVE_REQUESTS.get(), initial + 1.0);
        }
        assert_eq!(ACTIVE_REQUESTS.get(), initial);
    }

    #[test]
    fn test_latency_timer() {
        let initial_count = INFERENCE_LATENCY.get_sample_count();
        {
            let _timer = LatencyTimer::new();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(INFERENCE_LATENCY.get_sample_count(), initial_count + 1);
    }
}
