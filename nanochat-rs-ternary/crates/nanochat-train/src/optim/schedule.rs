//! Warmup-Stable-Decay (WSD) learning rate schedule.
//!
//! Provides multiple learning rate scheduling strategies:
//! - WSD (Warmup-Stable-Decay): Linear warmup, constant, cosine decay
//! - Cosine Annealing: Cosine decay with optional restarts
//! - Exponential Decay: Exponential decay with optional staircase
//! - Reduce on Plateau: Adaptive LR reduction based on validation loss

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
    let progress = ((step - decay_start) as f64 / decay_steps as f64).min(1.0); // Clamp at 1.0
    min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Cosine annealing schedule with optional restarts
///
/// # Arguments
/// * `step` - Current training step
/// * `base_lr` - Initial learning rate
/// * `min_lr` - Minimum learning rate
/// * `cycle_steps` - Number of steps per cycle
/// * `cycle_mult` - Cycle length multiplier after each restart (1.0 = no increase)
pub fn cosine_annealing_with_restarts(
    step: usize,
    base_lr: f64,
    min_lr: f64,
    cycle_steps: usize,
    cycle_mult: f64,
) -> f64 {
    if cycle_steps == 0 {
        return base_lr;
    }

    let mut cycle_start = 0usize;
    let mut current_cycle_steps = cycle_steps;
    
    // Find which cycle we're in
    while step >= cycle_start + current_cycle_steps {
        cycle_start += current_cycle_steps;
        current_cycle_steps = (current_cycle_steps as f64 * cycle_mult) as usize;
        if current_cycle_steps == 0 {
            current_cycle_steps = 1;
        }
    }
    
    let step_in_cycle = step - cycle_start;
    let progress = step_in_cycle as f64 / current_cycle_steps as f64;
    
    min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Exponential decay schedule
///
/// # Arguments
/// * `step` - Current training step
/// * `base_lr` - Initial learning rate
/// * `decay_rate` - Decay rate (e.g., 0.96 for 4% decay per period)
/// * `decay_steps` - Number of steps per decay period
/// * `staircase` - If true, decay happens at discrete intervals
pub fn exponential_decay(
    step: usize,
    base_lr: f64,
    decay_rate: f64,
    decay_steps: usize,
    staircase: bool,
) -> f64 {
    if decay_steps == 0 {
        return base_lr;
    }
    
    let exponent = if staircase {
        (step / decay_steps) as f64
    } else {
        step as f64 / decay_steps as f64
    };
    
    base_lr * decay_rate.powf(exponent)
}

/// Linear decay schedule
///
/// Linearly interpolates from base_lr to min_lr over total_steps
pub fn linear_decay(
    step: usize,
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
) -> f64 {
    if step >= total_steps {
        return min_lr;
    }
    
    let progress = step as f64 / total_steps as f64;
    base_lr - (base_lr - min_lr) * progress
}

/// Reduce on plateau learning rate scheduler
///
/// Reduces learning rate when a metric has stopped improving.
#[derive(Debug, Clone)]
pub struct ReduceOnPlateau {
    /// Number of steps with no improvement after which LR will be reduced
    pub patience: usize,
    /// Factor by which the learning rate will be reduced (new_lr = lr * factor)
    pub factor: f64,
    /// Lower bound on the learning rate
    pub min_lr: f64,
    /// Number of steps to wait before resuming normal operation after LR reduction
    pub cooldown: usize,
    /// Threshold for measuring improvement
    pub threshold: f64,
    
    // State
    best_value: f64,
    cooldown_counter: usize,
    num_reductions: usize,
    wait_counter: usize,
}

impl ReduceOnPlateau {
    /// Create a new ReduceOnPlateau scheduler
    ///
    /// # Arguments
    /// * `patience` - Steps to wait before reducing LR (default: 10)
    /// * `factor` - LR reduction factor (default: 0.5)
    /// * `min_lr` - Minimum LR (default: 1e-6)
    /// * `cooldown` - Steps to wait after reduction (default: 0)
    /// * `threshold` - Improvement threshold (default: 1e-4)
    pub fn new(patience: usize, factor: f64, min_lr: f64, cooldown: usize, threshold: f64) -> Self {
        Self {
            patience,
            factor,
            min_lr,
            cooldown,
            threshold,
            best_value: f64::INFINITY,
            cooldown_counter: 0,
            num_reductions: 0,
            wait_counter: 0,
        }
    }
}

impl Default for ReduceOnPlateau {
    fn default() -> Self {
        Self::new(10, 0.5, 1e-6, 0, 1e-4)
    }
}

impl ReduceOnPlateau {
    /// Step the scheduler with a new metric value (lower is better)
    ///
    /// Returns the new learning rate
    pub fn step(&mut self, current_value: f64, current_lr: f64) -> f64 {
        // Handle cooldown
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return current_lr;
        }

        // Check for improvement
        if current_value < self.best_value - self.threshold {
            // Improved
            self.best_value = current_value;
            self.wait_counter = 0;
            return current_lr;
        }

        // No improvement
        self.wait_counter += 1;

        if self.wait_counter >= self.patience {
            // Reduce LR
            let new_lr = (current_lr * self.factor).max(self.min_lr);
            if new_lr < current_lr {
                self.cooldown_counter = self.cooldown;
                self.num_reductions += 1;
                self.wait_counter = 0;
                return new_lr;
            }
        }

        current_lr
    }

    /// Get the number of times LR has been reduced
    pub fn num_reductions(&self) -> usize {
        self.num_reductions
    }

    /// Reset the scheduler state
    pub fn reset(&mut self) {
        self.best_value = f64::INFINITY;
        self.cooldown_counter = 0;
        self.wait_counter = 0;
        self.num_reductions = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wsd_warmup_phase() {
        // Linear ramp from 0 to 1
        let mult_0 = wsd_schedule(0, 100, 1000, 0.8, 0.1);
        assert!(
            (mult_0 - 0.0).abs() < 1e-10,
            "Step 0 should be 0.0, got {}",
            mult_0
        );

        let mult_50 = wsd_schedule(50, 100, 1000, 0.8, 0.1);
        assert!(
            (mult_50 - 0.5).abs() < 1e-10,
            "Step 50 should be 0.5, got {}",
            mult_50
        );

        let mult_99 = wsd_schedule(99, 100, 1000, 0.8, 0.1);
        assert!(
            (mult_99 - 0.99).abs() < 1e-10,
            "Step 99 should be 0.99, got {}",
            mult_99
        );
    }

    #[test]
    fn test_wsd_stable_phase() {
        // Constant 1.0 between warmup and decay
        for step in [100, 200, 500, 799] {
            let mult = wsd_schedule(step, 100, 1000, 0.8, 0.1);
            assert!(
                (mult - 1.0).abs() < 1e-10,
                "Step {} should be 1.0, got {}",
                step,
                mult
            );
        }
    }

    #[test]
    fn test_wsd_decay_phase() {
        // Cosine decay from 1.0 to min_lr_frac (0.1)
        let mult_start = wsd_schedule(800, 100, 1000, 0.8, 0.1);
        assert!(
            (mult_start - 1.0).abs() < 1e-6,
            "Decay start should be ~1.0, got {}",
            mult_start
        );

        let mult_mid = wsd_schedule(900, 100, 1000, 0.8, 0.1);
        assert!(
            mult_mid > 0.4 && mult_mid < 0.6,
            "Decay mid should be ~0.55, got {}",
            mult_mid
        );

        let mult_end = wsd_schedule(1000, 100, 1000, 0.8, 0.1);
        assert!(
            (mult_end - 0.1).abs() < 1e-6,
            "Decay end should be 0.1, got {}",
            mult_end
        );
    }

    #[test]
    fn test_cosine_annealing_no_restart() {
        // With cycle_mult=1.0, cycles don't increase
        let base_lr = 0.1;
        let min_lr = 0.01;
        let cycle_steps = 100;
        
        // Start of cycle
        let lr_start = cosine_annealing_with_restarts(0, base_lr, min_lr, cycle_steps, 1.0);
        assert!((lr_start - base_lr).abs() < 1e-10);
        
        // Middle of cycle
        let lr_mid = cosine_annealing_with_restarts(50, base_lr, min_lr, cycle_steps, 1.0);
        assert!(lr_mid > min_lr && lr_mid < base_lr);
        
        // Near end of cycle - should be very close to min_lr
        // Note: progress=0.99, cos(π*0.99)≈-0.9999, so lr≈min_lr + small amount
        let lr_near_end = cosine_annealing_with_restarts(99, base_lr, min_lr, cycle_steps, 1.0);
        assert!(lr_near_end < base_lr && lr_near_end >= min_lr, 
            "lr_near_end={}, min_lr={}, base_lr={}", lr_near_end, min_lr, base_lr);
    }

    #[test]
    fn test_cosine_annealing_with_restart() {
        let base_lr = 0.1;
        let min_lr = 0.01;
        let cycle_steps = 100;
        let cycle_mult = 2.0; // Each cycle is 2x longer
        
        // First cycle ends near step 99
        let lr_first_end = cosine_annealing_with_restarts(99, base_lr, min_lr, cycle_steps, cycle_mult);
        assert!(lr_first_end >= min_lr && lr_first_end < base_lr);
        
        // Second cycle: 100-299 (200 steps)
        let lr_second_start = cosine_annealing_with_restarts(100, base_lr, min_lr, cycle_steps, cycle_mult);
        assert!((lr_second_start - base_lr).abs() < 1e-10);
        
        let lr_second_mid = cosine_annealing_with_restarts(200, base_lr, min_lr, cycle_steps, cycle_mult);
        assert!(lr_second_mid > min_lr && lr_second_mid < base_lr);
    }

    #[test]
    fn test_exponential_decay_continuous() {
        let base_lr = 0.1;
        let decay_rate = 0.96;
        let decay_steps = 100;
        
        let lr_0 = exponential_decay(0, base_lr, decay_rate, decay_steps, false);
        assert!((lr_0 - base_lr).abs() < 1e-10);
        
        let lr_100 = exponential_decay(100, base_lr, decay_rate, decay_steps, false);
        let expected = base_lr * decay_rate;
        assert!((lr_100 - expected).abs() < 1e-10);
        
        let lr_200 = exponential_decay(200, base_lr, decay_rate, decay_steps, false);
        let expected_2 = base_lr * decay_rate.powi(2);
        assert!((lr_200 - expected_2).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay_staircase() {
        let base_lr = 0.1;
        let decay_rate = 0.5;
        let decay_steps = 100;
        
        // Same LR throughout each staircase step
        for step in 0..100 {
            let lr = exponential_decay(step, base_lr, decay_rate, decay_steps, true);
            assert!((lr - base_lr).abs() < 1e-10, "Step {} should be base_lr", step);
        }
        
        // After first step
        let lr_100 = exponential_decay(100, base_lr, decay_rate, decay_steps, true);
        assert!((lr_100 - base_lr * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_decay() {
        let base_lr = 0.1;
        let min_lr = 0.01;
        let total_steps = 100;
        
        let lr_0 = linear_decay(0, base_lr, min_lr, total_steps);
        assert!((lr_0 - base_lr).abs() < 1e-10);
        
        let lr_50 = linear_decay(50, base_lr, min_lr, total_steps);
        let expected_mid = base_lr - (base_lr - min_lr) * 0.5;
        assert!((lr_50 - expected_mid).abs() < 1e-10);
        
        let lr_100 = linear_decay(100, base_lr, min_lr, total_steps);
        assert!((lr_100 - min_lr).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = ReduceOnPlateau::new(3, 0.5, 1e-6, 0, 1e-4);
        let base_lr = 0.1;
        
        // Initial improvement
        let lr1 = scheduler.step(1.0, base_lr);
        assert!((lr1 - base_lr).abs() < 1e-10);
        
        let lr2 = scheduler.step(0.9, base_lr);
        assert!((lr2 - base_lr).abs() < 1e-10);
        
        // No improvement for 3 steps (patience=3)
        let lr3 = scheduler.step(0.91, base_lr);
        assert!((lr3 - base_lr).abs() < 1e-10);
        
        let lr4 = scheduler.step(0.92, base_lr);
        assert!((lr4 - base_lr).abs() < 1e-10);
        
        // Should reduce LR after patience exceeded
        let lr5 = scheduler.step(0.93, base_lr);
        assert!((lr5 - base_lr * 0.5).abs() < 1e-10);
        assert_eq!(scheduler.num_reductions(), 1);
        
        // Reset and verify
        scheduler.reset();
        assert_eq!(scheduler.num_reductions(), 0);
    }

    #[test]
    fn test_reduce_on_plateau_min_lr() {
        let mut scheduler = ReduceOnPlateau::new(1, 0.5, 0.01, 0, 1e-4);
        let mut lr = 0.1;
        
        // Reduce multiple times
        for _ in 0..10 {
            lr = scheduler.step(1.0, lr);
        }
        
        // Should not go below min_lr
        assert!((lr - 0.01).abs() < 1e-10 || lr > 0.01);
    }
}
