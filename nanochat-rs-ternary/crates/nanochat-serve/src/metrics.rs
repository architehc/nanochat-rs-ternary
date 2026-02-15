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
use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Opts, Registry, TextEncoder};
use std::sync::Once;

static METRICS_INIT: Once = Once::new();

#[derive(Clone)]
pub struct SafeCounter {
    inner: Option<Counter>,
}

impl SafeCounter {
    fn new(name: &str, help: &str) -> Self {
        let counter = Counter::with_opts(Opts::new(name, help))
            .unwrap_or_else(|err| panic!("failed to create counter {}: {}", name, err));
        Self {
            inner: Some(counter),
        }
    }

    fn register(&self, registry: &Registry) {
        if let Some(metric) = &self.inner {
            registry
                .register(Box::new(metric.clone()))
                .unwrap_or_else(|err| panic!("failed to register counter metric: {}", err));
        }
    }

    pub fn inc(&self) {
        if let Some(metric) = &self.inner {
            metric.inc();
        }
    }

    pub fn inc_by(&self, value: f64) {
        if let Some(metric) = &self.inner {
            metric.inc_by(value);
        }
    }

    pub fn get(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, Counter::get)
    }
}

#[derive(Clone)]
pub struct SafeGauge {
    inner: Option<Gauge>,
}

impl SafeGauge {
    fn new(name: &str, help: &str) -> Self {
        let gauge = Gauge::with_opts(Opts::new(name, help))
            .unwrap_or_else(|err| panic!("failed to create gauge {}: {}", name, err));
        Self { inner: Some(gauge) }
    }

    fn register(&self, registry: &Registry) {
        if let Some(metric) = &self.inner {
            registry
                .register(Box::new(metric.clone()))
                .unwrap_or_else(|err| panic!("failed to register gauge metric: {}", err));
        }
    }

    pub fn inc(&self) {
        if let Some(metric) = &self.inner {
            metric.inc();
        }
    }

    pub fn dec(&self) {
        if let Some(metric) = &self.inner {
            metric.dec();
        }
    }

    pub fn set(&self, value: f64) {
        if let Some(metric) = &self.inner {
            metric.set(value);
        }
    }

    pub fn get(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, Gauge::get)
    }
}

#[derive(Clone)]
pub struct SafeHistogram {
    inner: Option<Histogram>,
}

impl SafeHistogram {
    fn new(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let opts = HistogramOpts::new(name, help).buckets(buckets);
        let histogram = Histogram::with_opts(opts)
            .unwrap_or_else(|err| panic!("failed to create histogram {}: {}", name, err));
        Self {
            inner: Some(histogram),
        }
    }

    fn register(&self, registry: &Registry) {
        if let Some(metric) = &self.inner {
            registry
                .register(Box::new(metric.clone()))
                .unwrap_or_else(|err| panic!("failed to register histogram metric: {}", err));
        }
    }

    pub fn observe(&self, value: f64) {
        if let Some(metric) = &self.inner {
            metric.observe(value);
        }
    }

    pub fn get_sample_count(&self) -> u64 {
        self.inner.as_ref().map_or(0, Histogram::get_sample_count)
    }
}

lazy_static! {
    /// Global metrics registry.
    pub static ref REGISTRY: Registry = Registry::new();

    /// Total number of inference requests.
    pub static ref INFERENCE_REQUESTS: SafeCounter = SafeCounter::new(
        "nanochat_inference_requests_total",
        "Total number of inference requests"
    );

    /// Total number of failed requests.
    pub static ref INFERENCE_ERRORS: SafeCounter = SafeCounter::new(
        "nanochat_inference_errors_total",
        "Total number of failed inference requests"
    );

    /// Inference latency histogram in seconds.
    pub static ref INFERENCE_LATENCY: SafeHistogram = SafeHistogram::new(
        "nanochat_inference_latency_seconds",
        "Inference latency in seconds",
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    );

    /// Number of currently active requests.
    pub static ref ACTIVE_REQUESTS: SafeGauge = SafeGauge::new(
        "nanochat_active_requests",
        "Number of currently active inference requests"
    );

    /// Model load time in seconds.
    pub static ref MODEL_LOAD_TIME: SafeGauge = SafeGauge::new(
        "nanochat_model_load_time_seconds",
        "Time taken to load model in seconds"
    );

    /// Total tokens generated.
    pub static ref TOKENS_GENERATED: SafeCounter = SafeCounter::new(
        "nanochat_tokens_generated_total",
        "Total number of tokens generated"
    );

    /// Tokens per second (throughput).
    pub static ref TOKENS_PER_SECOND: SafeGauge = SafeGauge::new(
        "nanochat_tokens_per_second",
        "Current tokens per second throughput"
    );
}

/// Register all metrics with the global registry.
pub fn register_metrics() {
    METRICS_INIT.call_once(|| {
        INFERENCE_REQUESTS.register(&REGISTRY);
        INFERENCE_ERRORS.register(&REGISTRY);
        INFERENCE_LATENCY.register(&REGISTRY);
        ACTIVE_REQUESTS.register(&REGISTRY);
        MODEL_LOAD_TIME.register(&REGISTRY);
        TOKENS_GENERATED.register(&REGISTRY);
        TOKENS_PER_SECOND.register(&REGISTRY);
    });
}

/// Render metrics in Prometheus text format.
pub fn render_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        eprintln!("ERROR: failed to encode metrics: {}", err);
        return "# metrics_encode_error 1\n".to_string();
    }
    String::from_utf8(buffer).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to render metrics as UTF-8: {}", err);
        "# metrics_utf8_error 1\n".to_string()
    })
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
