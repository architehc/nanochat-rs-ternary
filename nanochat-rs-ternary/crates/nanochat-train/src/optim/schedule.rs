//! Warmup-Stable-Decay (WSD) learning rate schedule.

/// WSD learning rate multiplier.
///
/// Returns a value in [min_lr_frac, 1.0]:
/// - Warmup phase: linear ramp from 0 to 1 over `warmup_steps`
/// - Stable phase: constant 1.0
/// - Decay phase: cosine anneal from 1.0 to `min_lr_frac`
pub fn wsd_schedule(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    decay_start_frac: f64,
    min_lr_frac: f64,
) -> f64 {
    if step < warmup_steps {
        return step as f64 / warmup_steps.max(1) as f64;
    }

    let decay_start = (total_steps as f64 * decay_start_frac) as usize;

    if step < decay_start {
        return 1.0;
    }

    let decay_steps = total_steps.saturating_sub(decay_start).max(1);
    let progress = ((step - decay_start) as f64 / decay_steps as f64).min(1.0);  // Clamp at 1.0
    min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + (std::f64::consts::PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wsd_warmup_phase() {
        // Linear ramp from 0 to 1
        let mult_0 = wsd_schedule(0, 100, 1000, 0.8, 0.1);
        assert!((mult_0 - 0.0).abs() < 1e-10, "Step 0 should be 0.0, got {}", mult_0);

        let mult_50 = wsd_schedule(50, 100, 1000, 0.8, 0.1);
        assert!((mult_50 - 0.5).abs() < 1e-10, "Step 50 should be 0.5, got {}", mult_50);

        let mult_99 = wsd_schedule(99, 100, 1000, 0.8, 0.1);
        assert!((mult_99 - 0.99).abs() < 1e-10, "Step 99 should be 0.99, got {}", mult_99);
    }

    #[test]
    fn test_wsd_stable_phase() {
        // Constant 1.0 between warmup and decay
        for step in [100, 200, 500, 799] {
            let mult = wsd_schedule(step, 100, 1000, 0.8, 0.1);
            assert!((mult - 1.0).abs() < 1e-10, "Step {} should be 1.0, got {}", step, mult);
        }
    }

    #[test]
    fn test_wsd_decay_phase() {
        // Cosine decay from 1.0 to min_lr_frac (0.1)
        let mult_start = wsd_schedule(800, 100, 1000, 0.8, 0.1);
        assert!((mult_start - 1.0).abs() < 1e-6, "Decay start should be ~1.0, got {}", mult_start);

        let mult_mid = wsd_schedule(900, 100, 1000, 0.8, 0.1);
        assert!(mult_mid > 0.4 && mult_mid < 0.6, "Decay mid should be ~0.55, got {}", mult_mid);

        let mult_end = wsd_schedule(1000, 100, 1000, 0.8, 0.1);
        assert!((mult_end - 0.1).abs() < 1e-6, "Decay end should be 0.1, got {}", mult_end);
    }
}
