//! Inference engine: KV-cache management, sampling, text generation.
//!
//! Includes both a standard `InferenceEngine` and a NUMA-aware `NumaInferenceEngine`
//! for dual-socket systems (e.g. dual AMD EPYC 9654). The NUMA engine provides
//! per-request/thread-pool dispatch for prefill/decode on detected node pools.
//! It does not currently hard-pin worker threads to NUMA nodes.

use nanochat_model::config::ModelConfig;
use nanochat_model::model::NanochatModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ternary_core::planar;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            max_tokens: 256,
            seed: None,
        }
    }
}

/// Reason why generation finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Token generated during streaming.
pub struct GeneratedToken {
    pub token_id: u32,
    pub finish_reason: Option<String>,
    /// Whether the forward pass was degraded (LoopLM errors)
    pub degraded: bool,
}

/// Inference engine wrapping the model with generation logic.
pub struct InferenceEngine {
    pub model: NanochatModel,
    pub eot_token: u32,
}

impl InferenceEngine {
    pub fn new(model: NanochatModel) -> Self {
        let vocab = model.config.vocab_size;
        // Use vocab-1 as EOT for small vocabs, GPT-2 standard for full vocab
        let eot = if vocab <= 50256 {
            (vocab - 1) as u32
        } else {
            50256
        };
        Self {
            model,
            eot_token: eot,
        }
    }

    /// Create engine with random weights for testing.
    pub fn new_random(config: ModelConfig) -> Self {
        let vocab = config.vocab_size;
        let eot = if vocab <= 50256 {
            (vocab - 1) as u32
        } else {
            50256
        };
        Self {
            model: NanochatModel::new_random(config),
            eot_token: eot,
        }
    }

    /// Generate tokens autoregressively.
    ///
    /// prompt_ids: input token sequence
    /// params: sampling parameters
    ///
    /// Returns: (generated token IDs not including prompt, finish reason).
    /// FinishReason::Length when max_tokens is reached, FinishReason::Stop otherwise.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
    ) -> (Vec<u32>, FinishReason) {
        let mut tokens = Vec::new();
        let mut reason = FinishReason::Stop;
        self.generate_streaming(prompt_ids, params, |tok| {
            tokens.push(tok.token_id);
            if let Some(ref fr) = tok.finish_reason {
                if fr == "length" {
                    reason = FinishReason::Length;
                }
            }
            tok.finish_reason.is_none()
        });
        // If we generated max_tokens without hitting EOS or explicit finish,
        // that's a length stop
        if tokens.len() >= params.max_tokens && reason == FinishReason::Stop {
            // Check: if the last token didn't have a "stop" finish_reason from EOS,
            // then we stopped due to length
            reason = FinishReason::Length;
        }
        (tokens, reason)
    }

    /// Generate tokens one at a time, calling `on_token` for each.
    /// Returns when generation is complete or `on_token` returns false.
    pub fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
        mut on_token: F,
    ) where
        F: FnMut(GeneratedToken) -> bool,
    {
        // Validate prompt length to prevent RoPE assert panics
        let max_seq_len = self.model.config.max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            eprintln!(
                "ERROR: Prompt length {} exceeds model max_seq_len {}, returning empty generation",
                prompt_ids.len(),
                max_seq_len
            );
            // Return early without generating anything
            return;
        }

        self.model.reset_caches();

        let mut rng: Box<dyn RngCore> = match params.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(StdRng::from_entropy()),
        };

        // Prefill: process all prompt tokens (batched for efficiency)
        let mut logits = if prompt_ids.len() > 1 {
            self.model.forward_sequence_batched(prompt_ids)
        } else if prompt_ids.len() == 1 {
            self.model.forward_token(prompt_ids[0], 0)
        } else {
            vec![]
        };

        // Check for degraded state after prefill
        if self.model.last_forward_was_degraded() {
            eprintln!("WARNING: Forward pass degraded during prefill");
        }

        // Decode: generate one token at a time
        let mut pos = prompt_ids.len();

        for _ in 0..params.max_tokens {
            if logits.is_empty() {
                break;
            }

            let next_token = sample_token(&logits, params, &mut *rng);

            let is_eot = next_token == self.eot_token;
            let at_limit = pos + 1 >= self.model.config.max_seq_len;

            let finish_reason = if is_eot {
                Some("stop".to_string())
            } else if at_limit {
                Some("length".to_string())
            } else {
                None
            };

            let degraded = self.model.last_forward_was_degraded();
            let should_continue = on_token(GeneratedToken {
                token_id: next_token,
                finish_reason: finish_reason.clone(),
                degraded,
            });

            if !should_continue || is_eot || at_limit {
                break;
            }

            logits = self.model.forward_token(next_token, pos);
            pos += 1;

            // Check for degraded state after each token
            if self.model.last_forward_was_degraded() {
                eprintln!("WARNING: Forward pass degraded at position {}", pos);
            }
        }
    }
}

/// Runtime-selectable engine variant used by the HTTP server.
pub enum EngineHandle {
    Standard(InferenceEngine),
    Numa(NumaInferenceEngine),
}

impl EngineHandle {
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
    ) -> (Vec<u32>, FinishReason) {
        match self {
            EngineHandle::Standard(engine) => engine.generate(prompt_ids, params),
            EngineHandle::Numa(engine) => engine.generate(prompt_ids, params),
        }
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
        on_token: F,
    ) where
        F: FnMut(GeneratedToken) -> bool,
    {
        match self {
            EngineHandle::Standard(engine) => {
                engine.generate_streaming(prompt_ids, params, on_token)
            }
            EngineHandle::Numa(engine) => engine.generate_streaming(prompt_ids, params, on_token),
        }
    }

    pub fn last_forward_was_degraded(&self) -> bool {
        match self {
            EngineHandle::Standard(engine) => engine.model.last_forward_was_degraded(),
            EngineHandle::Numa(engine) => engine.model.last_forward_was_degraded(),
        }
    }

    pub fn last_forward_error_message(&self) -> Option<String> {
        match self {
            EngineHandle::Standard(engine) => engine.model.last_forward_error_message(),
            EngineHandle::Numa(engine) => engine.model.last_forward_error_message(),
        }
    }
}

// ============================================================
// NUMA-aware Inference Engine
// ============================================================

/// NUMA configuration for a dual-socket inference engine.
#[derive(Debug)]
pub struct NumaConfig {
    /// Number of NUMA nodes detected
    pub num_nodes: usize,
    /// Number of threads per node pool
    pub threads_per_node: usize,
    /// Whether NUMA is actually active (runtime check)
    pub numa_active: bool,
}

impl NumaConfig {
    /// Detect NUMA topology and create configuration.
    ///
    /// On a dual-socket EPYC 9654 (224 threads total), this would produce:
    /// - num_nodes: 2
    /// - threads_per_node: 112
    /// - numa_active: true
    ///
    /// Falls back gracefully on single-socket or non-NUMA systems.
    pub fn detect() -> Self {
        let numa_active = planar::numa_is_available();
        let max_node = planar::numa_max_node_id();
        let num_nodes = if numa_active { max_node + 1 } else { 1 };

        // Divide available threads across nodes
        let total_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads_per_node = total_threads / num_nodes;

        Self {
            num_nodes,
            threads_per_node: threads_per_node.max(1),
            numa_active,
        }
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(spec: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in spec.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let Ok(start) = start.trim().parse::<usize>() else {
                continue;
            };
            let Ok(end) = end.trim().parse::<usize>() else {
                continue;
            };
            if end < start {
                continue;
            }
            cpus.extend(start..=end);
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

#[cfg(target_os = "linux")]
fn node_cpu_ids(node: usize) -> Vec<usize> {
    let path = format!("/sys/devices/system/node/node{}/cpulist", node);
    std::fs::read_to_string(path)
        .map(|s| parse_linux_cpu_list(&s))
        .unwrap_or_default()
}

#[cfg(not(target_os = "linux"))]
fn node_cpu_ids(_node: usize) -> Vec<usize> {
    Vec::new()
}

fn pin_current_thread(cpu: usize) {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        return;
    };
    if let Some(core) = core_ids.into_iter().find(|c| c.id == cpu) {
        let _ = core_affinity::set_for_current(core);
    }
}

/// NUMA-aware inference engine for dual-socket systems.
///
/// Current behavior:
/// - Detects NUMA topology and builds per-node thread pools.
/// - Uses NUMA-aware weight allocation where available.
/// - Dispatches prefill/decode work onto per-node thread pools.
/// - Attempts CPU affinity pinning for pools when node CPU topology is available.
/// - Does not yet split a single forward pass per layer across nodes.
pub struct NumaInferenceEngine {
    pub model: NanochatModel,
    pub eot_token: u32,
    pub numa_config: NumaConfig,
    /// Per-node rayon thread pools. On non-NUMA systems, contains a single pool.
    pub thread_pools: Vec<rayon::ThreadPool>,
    /// Whether worker threads were pinned to node-local CPU sets.
    pub thread_pinning_active: bool,
    /// Layer split point: layers 0..split on node 0, split..N on node 1
    pub layer_split: usize,
}

impl NumaInferenceEngine {
    /// Create a NUMA-aware engine, detecting topology automatically.
    pub fn new(model: NanochatModel) -> Self {
        let numa_config = NumaConfig::detect();
        let n_layers = model.config.n_layers;
        let layer_split = n_layers / 2;

        // Create per-node thread pools
        let mut thread_pools = Vec::new();
        let mut thread_pinning_active = false;
        for node in 0..numa_config.num_nodes {
            let node_cpus = node_cpu_ids(node);
            let pinning_for_node = !node_cpus.is_empty();
            if pinning_for_node {
                thread_pinning_active = true;
            }
            let n_threads = if pinning_for_node {
                node_cpus.len().min(numa_config.threads_per_node).max(1)
            } else {
                numa_config.threads_per_node
            };

            let node_cpus_for_handler = node_cpus.clone();
            let mut builder = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .thread_name(move |idx| format!("numa-{}-worker-{}", node, idx));
            if pinning_for_node {
                builder = builder.start_handler(move |idx| {
                    if !node_cpus_for_handler.is_empty() {
                        let cpu = node_cpus_for_handler[idx % node_cpus_for_handler.len()];
                        pin_current_thread(cpu);
                    }
                });
            }

            let pool = builder.build().unwrap_or_else(|_| {
                // Fallback: build with default settings
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("failed to create fallback thread pool")
            });
            thread_pools.push(pool);
        }

        // Use model vocab for eot_token (same logic as standard engine)
        let vocab = model.config.vocab_size;
        let eot_token = if vocab <= 50256 {
            (vocab - 1) as u32
        } else {
            50256 // GPT-2 standard
        };

        Self {
            model,
            eot_token,
            numa_config,
            thread_pools,
            thread_pinning_active,
            layer_split,
        }
    }

    /// Create a NUMA engine with random weights for testing.
    pub fn new_random(config: ModelConfig) -> Self {
        let model = NanochatModel::new_random(config);
        Self::new(model)
    }

    /// Returns which NUMA node a given layer index should run on.
    pub fn node_for_layer(&self, layer_idx: usize) -> usize {
        if self.numa_config.num_nodes <= 1 {
            return 0;
        }
        if layer_idx < self.layer_split {
            0
        } else {
            1.min(self.numa_config.num_nodes - 1)
        }
    }

    fn forward_prefill_on_node(&mut self, prompt_ids: &[u32]) -> Vec<f32> {
        let node_idx = 0;
        let pool = &self.thread_pools[node_idx];
        let model = &mut self.model;
        pool.install(|| {
            if prompt_ids.len() > 1 {
                model.forward_sequence_batched(prompt_ids)
            } else if prompt_ids.len() == 1 {
                model.forward_token(prompt_ids[0], 0)
            } else {
                vec![]
            }
        })
    }

    fn forward_decode_on_node(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let node_idx = if self.numa_config.num_nodes <= 1 {
            0
        } else {
            // Round-robin token decode work across detected NUMA pools.
            pos % self.numa_config.num_nodes
        };
        let pool = &self.thread_pools[node_idx];
        let model = &mut self.model;
        pool.install(|| model.forward_token(token, pos))
    }

    /// Generate tokens autoregressively with NUMA thread-pool dispatch.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
    ) -> (Vec<u32>, FinishReason) {
        let mut tokens = Vec::new();
        let mut reason = FinishReason::Stop;
        self.generate_streaming(prompt_ids, params, |tok| {
            tokens.push(tok.token_id);
            if let Some(ref fr) = tok.finish_reason {
                if fr == "length" {
                    reason = FinishReason::Length;
                }
            }
            tok.finish_reason.is_none()
        });
        if tokens.len() >= params.max_tokens && reason == FinishReason::Stop {
            reason = FinishReason::Length;
        }
        (tokens, reason)
    }

    /// Generate tokens one at a time with NUMA thread-pool dispatch.
    pub fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[u32],
        params: &SamplingParams,
        mut on_token: F,
    ) where
        F: FnMut(GeneratedToken) -> bool,
    {
        // Validate prompt length to prevent RoPE assert panics
        let max_seq_len = self.model.config.max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            eprintln!(
                "ERROR: Prompt length {} exceeds model max_seq_len {}, returning empty generation",
                prompt_ids.len(),
                max_seq_len
            );
            return;
        }

        self.model.reset_caches();
        let mut rng: Box<dyn RngCore> = match params.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(StdRng::from_entropy()),
        };

        let mut logits = self.forward_prefill_on_node(prompt_ids);
        if self.model.last_forward_was_degraded() {
            eprintln!("WARNING: NUMA forward pass degraded during prefill");
        }

        let mut pos = prompt_ids.len();

        for _ in 0..params.max_tokens {
            if logits.is_empty() {
                break;
            }

            let next_token = sample_token(&logits, params, &mut *rng);
            let is_eot = next_token == self.eot_token;
            let at_limit = pos + 1 >= self.model.config.max_seq_len;
            let finish_reason = if is_eot {
                Some("stop".to_string())
            } else if at_limit {
                Some("length".to_string())
            } else {
                None
            };

            let degraded = self.model.last_forward_was_degraded();
            let should_continue = on_token(GeneratedToken {
                token_id: next_token,
                finish_reason: finish_reason.clone(),
                degraded,
            });

            if !should_continue || is_eot || at_limit {
                break;
            }

            logits = self.forward_decode_on_node(next_token, pos);
            pos += 1;

            if self.model.last_forward_was_degraded() {
                eprintln!("WARNING: NUMA forward pass degraded at position {}", pos);
            }
        }
    }

    /// Report NUMA status for logging.
    pub fn numa_status(&self) -> String {
        if self.numa_config.numa_active {
            let pinning = if self.thread_pinning_active {
                "affinity pinning enabled"
            } else {
                "no affinity pinning"
            };
            format!(
                "NUMA active: {} nodes, {} threads/node, decode dispatch via pools ({}), layer split metadata {}/{}",
                self.numa_config.num_nodes,
                self.numa_config.threads_per_node,
                pinning,
                self.layer_split,
                self.model.config.n_layers,
            )
        } else {
            format!(
                "NUMA not available, single-node mode ({} threads)",
                self.numa_config.threads_per_node * self.numa_config.num_nodes,
            )
        }
    }
}

// Trait alias for RNG used in sampling
use rand::RngCore;

/// Sample a token from logits given sampling parameters.
pub fn sample_token(logits: &[f32], params: &SamplingParams, rng: &mut dyn RngCore) -> u32 {
    if params.temperature < 1e-6 {
        return argmax(logits);
    }

    // Replace non-finite logits with large negative value
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| {
            if l.is_finite() {
                l / params.temperature
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();

    // Top-k filtering
    if params.top_k > 0 && params.top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        // Handle NaN by treating it as less than any finite value
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Less)
        });
        let threshold = probs[indices[params.top_k - 1]];
        for p in probs.iter_mut() {
            if *p < threshold {
                *p = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (*p - max_val).exp();
        if !p.is_finite() {
            *p = 0.0; // Clamp non-finite results to 0
        }
        sum += *p;
    }
    // Handle edge case where all probabilities are 0 or non-finite
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Top-p (nucleus) filtering
    if params.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        // Safe comparison handling NaN
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted.len();
        for (i, &(_, p)) in sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> =
            sorted[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *p = 0.0;
            }
        }
        // Re-normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv;
            }
        }
    }

    // Categorical sampling
    categorical_sample(&probs, rng)
}

/// Argmax over a slice.
fn argmax(x: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Sample from a categorical distribution using the inverse CDF method.
fn categorical_sample(probs: &[f32], rng: &mut dyn RngCore) -> u32 {
    let u: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    // Fallback: return last non-zero probability token
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(sample_token(&logits, &params, &mut rng), 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let params = SamplingParams {
            temperature: 0.01, // very low temp -> nearly greedy
            top_k: 0,
            top_p: 1.0,
            max_tokens: 10,
            seed: Some(42),
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 2); // Should pick max with very low temp
    }

    #[test]
    fn test_categorical_sample_uniform() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let t = categorical_sample(&probs, &mut rng);
            counts[t as usize] += 1;
        }
        // Each should be roughly 250 ± 50
        for &c in &counts {
            assert!(c > 150 && c < 350, "count {} out of range", c);
        }
    }

    #[test]
    fn test_categorical_sample_deterministic_seed() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);
        for _ in 0..100 {
            assert_eq!(
                categorical_sample(&probs, &mut rng1),
                categorical_sample(&probs, &mut rng2),
            );
        }
    }

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_k, 50);
        assert!((p.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_engine_generate() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let mut engine = InferenceEngine::new_random(config);
        engine.eot_token = 0; // Use 0 as EOT for small vocab

        let prompt = vec![1u32, 5, 10];
        let params = SamplingParams {
            temperature: 0.0, // greedy for determinism
            max_tokens: 5,
            ..Default::default()
        };

        let (output, finish_reason) = engine.generate(&prompt, &params);
        assert!(!output.is_empty(), "generate produced no tokens");
        assert!(output.len() <= 5, "exceeded max_tokens");
        // finish_reason should be Stop or Length
        assert!(
            finish_reason == FinishReason::Stop || finish_reason == FinishReason::Length,
            "unexpected finish reason: {:?}",
            finish_reason
        );

        for &t in &output {
            assert!((t as usize) < 256, "invalid token: {}", t);
        }
    }

    #[test]
    fn test_sample_with_top_k() {
        let logits = vec![1.0, 2.0, 10.0, 3.0, 0.5, 0.1, 0.2, 0.3];
        let params = SamplingParams {
            temperature: 0.01, // near-greedy
            top_k: 3,
            top_p: 1.0,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 2); // Should pick max
    }

    #[test]
    fn test_sample_with_top_p() {
        let logits = vec![10.0, 0.0, -10.0, -10.0, -10.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert_eq!(token, 0); // Token 0 has overwhelming probability
    }

    #[test]
    fn test_sample_with_top_k_and_top_p() {
        let logits = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let params = SamplingParams {
            temperature: 0.5,
            top_k: 4,
            top_p: 0.8,
            max_tokens: 10,
            seed: None,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let token = sample_token(&logits, &params, &mut rng);
        assert!(token < 4, "token {} should be in top-4", token);
    }

    #[test]
    fn test_engine_new() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let model = NanochatModel::new_random(config);
        let engine = InferenceEngine::new(model);
        assert_eq!(engine.model.config.dim, 128);
    }

    #[test]
    fn test_engine_empty_prompt() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let mut engine = InferenceEngine::new_random(config);

        let params = SamplingParams::default();
        let (output, _finish_reason) = engine.generate(&[], &params);
        assert!(output.is_empty());
    }

    // ============================================================
    // NUMA engine tests
    // ============================================================

    #[test]
    fn test_numa_config_detect() {
        let config = NumaConfig::detect();
        println!(
            "NUMA config: num_nodes={}, threads_per_node={}, active={}",
            config.num_nodes, config.threads_per_node, config.numa_active
        );
        assert!(config.num_nodes >= 1);
        assert!(config.threads_per_node >= 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_linux_cpu_list() {
        let cpus = parse_linux_cpu_list("0-3,8,10-11");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 10, 11]);
    }

    #[test]
    fn test_numa_engine_forward() {
        let config = ModelConfig::test_config(128, 4, 4, 256); // Use 4 layers to test split at 2
        let mut engine = NumaInferenceEngine::new_random(config);
        engine.eot_token = 0; // Use 0 as EOT for small vocab

        // Verify layer split
        assert_eq!(engine.layer_split, 2);
        assert_eq!(engine.node_for_layer(0), 0);
        assert_eq!(engine.node_for_layer(1), 0);
        if engine.numa_config.num_nodes > 1 {
            assert_eq!(engine.node_for_layer(2), 1);
            assert_eq!(engine.node_for_layer(3), 1);
        }

        let prompt = vec![1u32, 5, 10];
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        let (output, _finish_reason) = engine.generate(&prompt, &params);
        // Should produce valid tokens (may be empty if first token is EOT)
        for &t in &output {
            assert!((t as usize) < 256, "invalid token: {}", t);
        }

        // NUMA status should be reportable
        let status = engine.numa_status();
        assert!(!status.is_empty());
        println!("NUMA status: {}", status);
    }

    #[test]
    fn test_numa_engine_thread_pools() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let engine = NumaInferenceEngine::new_random(config);

        // Should have at least 1 thread pool
        assert!(!engine.thread_pools.is_empty());
        assert_eq!(engine.thread_pools.len(), engine.numa_config.num_nodes);

        // Each pool should be functional
        for (i, pool) in engine.thread_pools.iter().enumerate() {
            let result = pool.install(|| 42 + i);
            assert_eq!(result, 42 + i);
        }
    }

    #[test]
    fn test_generate_streaming() {
        let config = ModelConfig::test_config(128, 2, 4, 256);
        let mut engine = InferenceEngine::new_random(config);
        engine.eot_token = 0;

        let prompt = vec![1u32, 5];
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };

        let mut tokens = Vec::new();
        let mut had_finish = false;
        engine.generate_streaming(&prompt, &params, |tok| {
            tokens.push(tok.token_id);
            if tok.finish_reason.is_some() {
                had_finish = true;
            }
            tok.finish_reason.is_none()
        });

        assert!(!tokens.is_empty());
    }
}
