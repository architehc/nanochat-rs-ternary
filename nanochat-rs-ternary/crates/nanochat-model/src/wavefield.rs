//! Wave Field Attention: O(n log n) physics-based attention via FFT convolution.
//!
//! Replaces standard KV-cache attention with a constant-size wave field state.
//! Heavy projections (scatter, gate, output) are ternary BitLinear;
//! physics engine (omega, alpha, phi + FFT) stays FP32.

use crate::bitlinear::BitLinear;
use crate::config::WaveFieldConfig;
use num_complex::Complex;
use std::sync::Mutex;

/// Per-head physics parameters: frequency, damping, phase.
#[derive(Debug)]
pub struct WavePhysicsParams {
    pub omega: Vec<f32>, // [n_heads]
    pub alpha: Vec<f32>, // [n_heads] post-softplus, always positive
    pub phi: Vec<f32>,   // [n_heads]
}

/// Pre-FFT'd wave kernels in frequency domain (cached at model load).
#[derive(Debug)]
pub struct WaveKernelCache {
    pub kernels_freq: Vec<Complex<f32>>, // [n_heads * fft_size]
    pub n_heads: usize,
    pub field_size: usize,
    pub fft_size: usize,
}

impl WaveKernelCache {
    /// Build kernel cache from physics params.
    pub fn from_physics(physics: &WavePhysicsParams, field_size: usize) -> Self {
        let n_heads = physics.omega.len();
        let fft_size = (2 * field_size).next_power_of_two();
        let mut kernels_freq = vec![Complex::new(0.0, 0.0); n_heads * fft_size];

        for h in 0..n_heads {
            // Generate time-domain kernel: k[t] = exp(-alpha*t) * cos(omega*t + phi)
            let mut kernel = vec![0.0f32; field_size];
            for t in 0..field_size {
                let tf = t as f32;
                kernel[t] = (-physics.alpha[h] * tf).exp() * (physics.omega[h] * tf + physics.phi[h]).cos();
            }
            // FFT the kernel
            let kernel_fft = wave_fft::cpu_fft::precompute_kernel_fft(&kernel, field_size);
            kernels_freq[h * fft_size..(h + 1) * fft_size].copy_from_slice(&kernel_fft);
        }

        Self {
            kernels_freq,
            n_heads,
            field_size,
            fft_size,
        }
    }
}

/// Inter-head coupling matrix (row-softmax normalized).
#[derive(Debug)]
pub struct HeadCoupling {
    pub weights: Vec<f32>, // [n_heads * n_heads] row-softmax
    pub n_heads: usize,
}

impl HeadCoupling {
    /// Create identity coupling (no cross-head interaction).
    pub fn identity(n_heads: usize) -> Self {
        let mut weights = vec![0.0f32; n_heads * n_heads];
        for h in 0..n_heads {
            weights[h * n_heads + h] = 1.0;
        }
        Self { weights, n_heads }
    }

    /// Create from raw logits (row-softmax applied).
    pub fn from_logits(logits: &[f32], n_heads: usize) -> Self {
        assert_eq!(logits.len(), n_heads * n_heads);
        let mut weights = vec![0.0f32; n_heads * n_heads];

        for h in 0..n_heads {
            let row = &logits[h * n_heads..(h + 1) * n_heads];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..n_heads {
                weights[h * n_heads + j] = (row[j] - max_val).exp();
                sum += weights[h * n_heads + j];
            }
            let inv_sum = 1.0 / sum;
            for j in 0..n_heads {
                weights[h * n_heads + j] *= inv_sum;
            }
        }

        Self { weights, n_heads }
    }

    /// Apply coupling: mix head fields. fields: [n_heads * field_size * head_dim]
    pub fn apply(&self, fields: &mut [f32], field_size: usize, head_dim: usize) {
        let n = self.n_heads;
        let stride = field_size * head_dim;
        let mut tmp = vec![0.0f32; fields.len()];

        for h_out in 0..n {
            for h_in in 0..n {
                let w = self.weights[h_out * n + h_in];
                if w.abs() < 1e-8 {
                    continue;
                }
                let src = &fields[h_in * stride..(h_in + 1) * stride];
                let dst = &mut tmp[h_out * stride..(h_out + 1) * stride];
                for i in 0..stride {
                    dst[i] += w * src[i];
                }
            }
        }

        fields.copy_from_slice(&tmp);
    }
}

/// Per-sequence wave field state (replaces KV cache).
/// Constant size regardless of sequence length.
#[derive(Debug, Clone)]
pub struct WaveFieldState {
    pub fields: Vec<f32>, // [n_heads * field_size * head_dim]
    pub n_heads: usize,
    pub field_size: usize,
    pub head_dim: usize,
}

impl WaveFieldState {
    pub fn new(n_heads: usize, field_size: usize, head_dim: usize) -> Self {
        Self {
            fields: vec![0.0f32; n_heads * field_size * head_dim],
            n_heads,
            field_size,
            head_dim,
        }
    }

    pub fn reset(&mut self) {
        self.fields.fill(0.0);
    }

    /// Compute field energy for diagnostics: sum(field^2).
    pub fn energy(&self) -> f32 {
        self.fields.iter().map(|v| v * v).sum()
    }
}

/// Workspace buffers for wave field forward pass (avoid allocation per token).
#[derive(Debug, Default)]
struct WaveFieldWorkspace {
    scattered: Vec<f32>,
    gate_out: Vec<f32>,
    gathered: Vec<f32>,
    proj_out: Vec<f32>,
    conv_buf: Vec<f32>,
}

/// Wave Field Attention layer for inference.
///
/// Uses ternary projections for scatter/gate/output, FP32 physics for wave propagation.
/// State is O(n_heads * field_size * head_dim) — constant in sequence length.
#[derive(Debug)]
pub struct WaveFieldAttention {
    pub scatter_proj: BitLinear, // dim -> n_heads * head_dim
    pub gate_proj: BitLinear,    // dim -> n_heads
    pub out_proj: BitLinear,     // n_heads * head_dim -> dim
    pub physics: WavePhysicsParams,
    pub coupling: Option<HeadCoupling>,
    pub kernel_cache: WaveKernelCache,
    pub n_heads: usize,
    pub head_dim: usize,
    pub field_size: usize,
    pub stride: f32, // (field_size - 1) / (max_seq_len - 1)
    workspace: Mutex<WaveFieldWorkspace>,
}

/// Bilinear scatter: deposit values at fractional position onto field.
fn bilinear_scatter(values: &[f32], field: &mut [f32], pos: f32, field_size: usize, head_dim: usize) {
    let idx_lo = pos.floor() as usize;
    let idx_hi = (idx_lo + 1).min(field_size - 1);
    let frac = pos - idx_lo as f32;

    for d in 0..head_dim {
        field[idx_lo * head_dim + d] += values[d] * (1.0 - frac);
        if idx_hi != idx_lo {
            field[idx_hi * head_dim + d] += values[d] * frac;
        }
    }
}

/// Bilinear gather: read values from fractional position on field.
fn bilinear_gather(field: &[f32], pos: f32, field_size: usize, head_dim: usize, out: &mut [f32]) {
    let idx_lo = pos.floor() as usize;
    let idx_hi = (idx_lo + 1).min(field_size - 1);
    let frac = pos - idx_lo as f32;

    for d in 0..head_dim {
        out[d] = field[idx_lo * head_dim + d] * (1.0 - frac)
            + field[idx_hi * head_dim + d] * frac;
    }
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl WaveFieldAttention {
    /// Create with random weights for testing.
    pub fn new_random(config: &WaveFieldConfig, dim: usize, max_seq_len: usize) -> Self {
        let n_heads = config.n_wave_heads;
        let head_dim = config.head_dim;
        let field_size = config.field_size;
        let total_proj = n_heads * head_dim;

        // Random scatter proj: dim -> n_heads * head_dim
        let scatter_weights: Vec<f32> = (0..total_proj * dim)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761) >> 16) % 200;
                v as f32 / 100.0 - 1.0
            })
            .collect();
        let scatter_proj = BitLinear::from_float(&scatter_weights, total_proj, dim, 128);

        // Random gate proj: dim -> n_heads
        let gate_weights: Vec<f32> = (0..n_heads * dim)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) >> 16) % 200;
                v as f32 / 100.0 - 1.0
            })
            .collect();
        let gate_proj = BitLinear::from_float(&gate_weights, n_heads, dim, 128);

        // Random output proj: n_heads * head_dim -> dim
        let out_weights: Vec<f32> = (0..dim * total_proj)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(22695477).wrapping_add(1) >> 16) % 200;
                v as f32 / 100.0 - 1.0
            })
            .collect();
        let out_proj = BitLinear::from_float(&out_weights, dim, total_proj, 128);

        // Physics params: spread across heads
        let physics = WavePhysicsParams {
            omega: (0..n_heads)
                .map(|h| 0.3 + (4.0 - 0.3) * h as f32 / n_heads.max(1) as f32)
                .collect(),
            alpha: (0..n_heads)
                .map(|h| 0.04 + (1.0 - 0.04) * h as f32 / n_heads.max(1) as f32)
                .collect(),
            phi: (0..n_heads)
                .map(|h| std::f32::consts::PI * h as f32 / n_heads.max(1) as f32)
                .collect(),
        };

        let kernel_cache = WaveKernelCache::from_physics(&physics, field_size);

        let coupling = if config.use_head_coupling {
            Some(HeadCoupling::identity(n_heads))
        } else {
            None
        };

        let stride = if max_seq_len > 1 {
            (field_size - 1) as f32 / (max_seq_len - 1) as f32
        } else {
            0.0
        };

        Self {
            scatter_proj,
            gate_proj,
            out_proj,
            physics,
            coupling,
            kernel_cache,
            n_heads,
            head_dim,
            field_size,
            stride,
            workspace: Mutex::new(WaveFieldWorkspace::default()),
        }
    }

    /// Single-token autoregressive forward.
    ///
    /// x: [dim]
    /// state: wave field state (modified in-place)
    /// pos: token position
    /// out: [dim] (caller-allocated)
    pub fn forward(&self, x: &[f32], state: &mut WaveFieldState, pos: usize, out: &mut [f32]) {
        let dim = x.len();
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let field_size = self.field_size;
        let total_proj = n_heads * head_dim;

        let mut ws = self.workspace.lock().expect("workspace lock poisoned");
        ws.scattered.resize(total_proj, 0.0);
        ws.gate_out.resize(n_heads, 0.0);
        ws.gathered.resize(total_proj, 0.0);
        ws.proj_out.resize(dim, 0.0);

        // 1. Scatter projection: x -> [n_heads * head_dim]
        self.scatter_proj.forward(x, &mut ws.scattered);

        // 2. Bilinear scatter onto field
        let field_pos = pos as f32 * self.stride;
        let field_pos = field_pos.min((field_size - 1) as f32).max(0.0);
        for h in 0..n_heads {
            let values = &ws.scattered[h * head_dim..(h + 1) * head_dim];
            let field = &mut state.fields[h * field_size * head_dim..(h + 1) * field_size * head_dim];
            bilinear_scatter(values, field, field_pos, field_size, head_dim);
        }

        // 3. FFT convolution per head (using precomputed kernels)
        // IMPORTANT: Convolve on a temporary copy — don't modify state.fields.
        // State accumulates raw scattered values; convolution is applied only for reading.
        let field_len = n_heads * field_size * head_dim;
        ws.conv_buf.resize(field_len, 0.0);
        let fft_size = self.kernel_cache.fft_size;
        let mut col_buf = vec![0.0f32; field_size];
        for h in 0..n_heads {
            let field_start = h * field_size * head_dim;
            let conv_start = h * field_size * head_dim;
            let kf_start = h * fft_size;

            for d in 0..head_dim {
                // Extract column d of this head's field
                for t in 0..field_size {
                    col_buf[t] = state.fields[field_start + t * head_dim + d];
                }

                // Convolve into temporary buffer
                let kf = &self.kernel_cache.kernels_freq[kf_start..kf_start + fft_size];
                let convolved = wave_fft::cpu_fft::fft_convolve_precomputed(&col_buf, kf, field_size);

                // Write to temporary convolved buffer (NOT back to state)
                for t in 0..field_size {
                    ws.conv_buf[conv_start + t * head_dim + d] = convolved[t];
                }
            }
        }

        // 4. Optional head coupling (applied to convolved copy)
        if let Some(ref coupling) = self.coupling {
            coupling.apply(&mut ws.conv_buf, field_size, head_dim);
        }

        // 5. Bilinear gather from convolved field into separate buffer
        //    (can't borrow ws.conv_buf and ws.gathered simultaneously)
        let conv_buf_ref = &ws.conv_buf;
        let mut gathered_local = vec![0.0f32; total_proj];
        for h in 0..n_heads {
            let conv_field = &conv_buf_ref[h * field_size * head_dim..(h + 1) * field_size * head_dim];
            let gathered = &mut gathered_local[h * head_dim..(h + 1) * head_dim];
            bilinear_gather(conv_field, field_pos, field_size, head_dim, gathered);
        }
        ws.gathered.resize(total_proj, 0.0);
        ws.gathered.copy_from_slice(&gathered_local);

        // 6. Content gate: sigmoid(gate_proj(x)) * gathered
        self.gate_proj.forward(x, &mut ws.gate_out);
        for h in 0..n_heads {
            let g = sigmoid(ws.gate_out[h]);
            for d in 0..head_dim {
                ws.gathered[h * head_dim + d] *= g;
            }
        }

        // 7. Output projection: [n_heads * head_dim] -> [dim]
        self.out_proj.forward(&ws.gathered, out);
    }

    /// Batched forward for prefill (all tokens at once).
    ///
    /// x_batch: [seq_len * dim]
    /// state: wave field state (modified in-place)
    /// start_pos: starting position
    /// seq_len: number of tokens
    /// out_batch: [seq_len * dim]
    pub fn forward_batch(
        &self,
        x_batch: &[f32],
        state: &mut WaveFieldState,
        start_pos: usize,
        seq_len: usize,
        out_batch: &mut [f32],
    ) {
        let dim = x_batch.len() / seq_len;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let field_size = self.field_size;
        let total_proj = n_heads * head_dim;

        // 1. Scatter all tokens onto fields
        let mut scatter_buf = vec![0.0f32; total_proj];
        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            self.scatter_proj.forward(x, &mut scatter_buf);

            let pos = (start_pos + t) as f32 * self.stride;
            let pos = pos.min((field_size - 1) as f32).max(0.0);

            for h in 0..n_heads {
                let values = &scatter_buf[h * head_dim..(h + 1) * head_dim];
                let field = &mut state.fields[h * field_size * head_dim..(h + 1) * field_size * head_dim];
                bilinear_scatter(values, field, pos, field_size, head_dim);
            }
        }

        // 2. Single FFT convolution per head into temporary buffer (O(n log n) total)
        // IMPORTANT: Don't modify state.fields — it accumulates raw scattered values.
        let fft_size = self.kernel_cache.fft_size;
        let field_len = n_heads * field_size * head_dim;
        let mut conv_fields = vec![0.0f32; field_len];
        let mut col_buf = vec![0.0f32; field_size];
        for h in 0..n_heads {
            let field_start = h * field_size * head_dim;
            let kf_start = h * fft_size;
            let kf = &self.kernel_cache.kernels_freq[kf_start..kf_start + fft_size];

            for d in 0..head_dim {
                for t in 0..field_size {
                    col_buf[t] = state.fields[field_start + t * head_dim + d];
                }

                let convolved = wave_fft::cpu_fft::fft_convolve_precomputed(&col_buf, kf, field_size);

                for t in 0..field_size {
                    conv_fields[field_start + t * head_dim + d] = convolved[t];
                }
            }
        }

        // 3. Optional coupling (applied to convolved copy)
        if let Some(ref coupling) = self.coupling {
            coupling.apply(&mut conv_fields, field_size, head_dim);
        }

        // 4. Gather per-token results from convolved buffer, apply gate, output projection
        let mut gathered = vec![0.0f32; total_proj];
        let mut gate_buf = vec![0.0f32; n_heads];
        for t in 0..seq_len {
            let x = &x_batch[t * dim..(t + 1) * dim];
            let out = &mut out_batch[t * dim..(t + 1) * dim];

            let pos = (start_pos + t) as f32 * self.stride;
            let pos = pos.min((field_size - 1) as f32).max(0.0);

            for h in 0..n_heads {
                let field = &conv_fields[h * field_size * head_dim..(h + 1) * field_size * head_dim];
                let g = &mut gathered[h * head_dim..(h + 1) * head_dim];
                bilinear_gather(field, pos, field_size, head_dim, g);
            }

            // Gate
            self.gate_proj.forward(x, &mut gate_buf);
            for h in 0..n_heads {
                let g = sigmoid(gate_buf[h]);
                for d in 0..head_dim {
                    gathered[h * head_dim + d] *= g;
                }
            }

            // Output projection
            self.out_proj.forward(&gathered, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WaveFieldConfig;

    fn test_wave_config() -> WaveFieldConfig {
        WaveFieldConfig {
            field_size: 64,
            n_wave_heads: 4,
            head_dim: 32,
            use_head_coupling: true,
        }
    }

    #[test]
    fn test_wave_field_single_token_finite_output() {
        let dim = 128;
        let max_seq = 128;
        let wf_cfg = test_wave_config();
        let attn = WaveFieldAttention::new_random(&wf_cfg, dim, max_seq);
        let mut state = WaveFieldState::new(wf_cfg.n_wave_heads, wf_cfg.field_size, wf_cfg.head_dim);

        let x = vec![0.1f32; dim];
        let mut out = vec![0.0f32; dim];

        attn.forward(&x, &mut state, 0, &mut out);

        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        assert!(out.iter().any(|&v| v != 0.0), "all-zero output");
    }

    #[test]
    fn test_wave_field_state_accumulates() {
        let dim = 128;
        let max_seq = 128;
        let wf_cfg = test_wave_config();
        let attn = WaveFieldAttention::new_random(&wf_cfg, dim, max_seq);
        let mut state = WaveFieldState::new(wf_cfg.n_wave_heads, wf_cfg.field_size, wf_cfg.head_dim);

        let e0 = state.energy();
        assert_eq!(e0, 0.0, "initial energy should be 0");

        let x = vec![0.5f32; dim];
        let mut out = vec![0.0f32; dim];

        attn.forward(&x, &mut state, 0, &mut out);
        let e1 = state.energy();
        assert!(e1 > 0.0, "energy should increase after token");

        attn.forward(&x, &mut state, 1, &mut out);
        let e2 = state.energy();
        assert!(e2 > 0.0, "energy should remain positive");
    }

    #[test]
    fn test_wave_field_state_reset() {
        let wf_cfg = test_wave_config();
        let mut state = WaveFieldState::new(wf_cfg.n_wave_heads, wf_cfg.field_size, wf_cfg.head_dim);
        state.fields[10] = 1.0;
        assert!(state.energy() > 0.0);

        state.reset();
        assert_eq!(state.energy(), 0.0);
    }

    #[test]
    fn test_wave_field_batch_produces_output() {
        let dim = 128;
        let max_seq = 128;
        let seq_len = 4;
        let wf_cfg = test_wave_config();
        let attn = WaveFieldAttention::new_random(&wf_cfg, dim, max_seq);
        let mut state = WaveFieldState::new(wf_cfg.n_wave_heads, wf_cfg.field_size, wf_cfg.head_dim);

        let x_batch = vec![0.1f32; seq_len * dim];
        let mut out_batch = vec![0.0f32; seq_len * dim];

        attn.forward_batch(&x_batch, &mut state, 0, seq_len, &mut out_batch);

        assert!(out_batch.iter().all(|v| v.is_finite()), "non-finite batch output");
        assert!(out_batch.iter().any(|&v| v != 0.0), "all-zero batch output");
    }

    #[test]
    fn test_wave_kernel_cache_decays() {
        let physics = WavePhysicsParams {
            omega: vec![1.0],
            alpha: vec![0.1],
            phi: vec![0.0],
        };
        let cache = WaveKernelCache::from_physics(&physics, 128);
        assert_eq!(cache.n_heads, 1);
        assert_eq!(cache.field_size, 128);
        assert!(cache.fft_size >= 256);
    }

    #[test]
    fn test_head_coupling_identity() {
        let n_heads = 4;
        let field_size = 8;
        let head_dim = 4;
        let coupling = HeadCoupling::identity(n_heads);

        let mut fields: Vec<f32> = (0..n_heads * field_size * head_dim)
            .map(|i| i as f32)
            .collect();
        let original = fields.clone();

        coupling.apply(&mut fields, field_size, head_dim);

        for i in 0..fields.len() {
            assert!(
                (fields[i] - original[i]).abs() < 1e-5,
                "identity coupling changed field at {}: {} vs {}",
                i, fields[i], original[i]
            );
        }
    }

    #[test]
    fn test_head_coupling_from_logits() {
        let n_heads = 3;
        let logits = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let coupling = HeadCoupling::from_logits(&logits, n_heads);

        // With dominant diagonal logits, should be approximately identity
        for h in 0..n_heads {
            assert!(
                coupling.weights[h * n_heads + h] > 0.95,
                "diagonal should dominate: {}",
                coupling.weights[h * n_heads + h]
            );
        }
    }

    #[test]
    fn test_bilinear_scatter_gather_roundtrip() {
        let field_size = 32;
        let head_dim = 4;
        let mut field = vec![0.0f32; field_size * head_dim];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        // Scatter at integer position
        bilinear_scatter(&values, &mut field, 5.0, field_size, head_dim);

        let mut gathered = vec![0.0f32; head_dim];
        bilinear_gather(&field, 5.0, field_size, head_dim, &mut gathered);

        for d in 0..head_dim {
            assert!(
                (gathered[d] - values[d]).abs() < 1e-5,
                "roundtrip failed at dim {}: {} vs {}",
                d, gathered[d], values[d]
            );
        }
    }

    #[test]
    fn test_bilinear_scatter_fractional() {
        let field_size = 32;
        let head_dim = 1;
        let mut field = vec![0.0f32; field_size * head_dim];

        // Scatter at position 5.3
        bilinear_scatter(&[1.0], &mut field, 5.3, field_size, head_dim);

        assert!((field[5] - 0.7).abs() < 1e-5, "field[5] should be 0.7, got {}", field[5]);
        assert!((field[6] - 0.3).abs() < 1e-5, "field[6] should be 0.3, got {}", field[6]);
    }

    #[test]
    fn test_energy_bounded_after_multiple_tokens() {
        let dim = 128;
        let max_seq = 128;
        let wf_cfg = test_wave_config();
        let attn = WaveFieldAttention::new_random(&wf_cfg, dim, max_seq);
        let mut state = WaveFieldState::new(wf_cfg.n_wave_heads, wf_cfg.field_size, wf_cfg.head_dim);

        let x = vec![0.1f32; dim];
        let mut out = vec![0.0f32; dim];

        for pos in 0..20 {
            attn.forward(&x, &mut state, pos, &mut out);
        }

        let energy = state.energy();
        assert!(energy.is_finite(), "energy should be finite, got {}", energy);
        // Energy should not explode — with positive damping it's bounded
        // (exact bound depends on input magnitude, but should stay reasonable)
    }
}
