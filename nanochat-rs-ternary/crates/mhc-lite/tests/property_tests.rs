//! Property-based tests for mHC-lite using proptest.
//!
//! Validates invariants that must hold for ALL possible parameter values:
//! - Doubly stochastic matrices (row/col sums = 1)
//! - Composite gain bounded by 1.0
//! - Non-negativity of all matrix elements

use mhc_lite::{MhcLiteN2, MhcLiteN4};
use proptest::prelude::*;

/// Verify doubly stochastic property: rows and columns sum to 1.0.
fn is_doubly_stochastic_2x2(matrix: &[[f32; 2]; 2], tol: f32) -> bool {
    // Check row sums
    for row in matrix {
        let row_sum: f32 = row.iter().sum();
        if (row_sum - 1.0).abs() > tol {
            return false;
        }
    }

    // Check column sums
    for j in 0..2 {
        let col_sum: f32 = (0..2).map(|i| matrix[i][j]).sum();
        if (col_sum - 1.0).abs() > tol {
            return false;
        }
    }

    // Check non-negativity
    for row in matrix {
        for &val in row {
            if val < 0.0 {
                return false;
            }
        }
    }

    true
}

fn is_doubly_stochastic_4x4(matrix: &[[f32; 4]; 4], tol: f32) -> bool {
    // Check row sums
    for row in matrix {
        let row_sum: f32 = row.iter().sum();
        if (row_sum - 1.0).abs() > tol {
            return false;
        }
    }

    // Check column sums
    for j in 0..4 {
        let col_sum: f32 = (0..4).map(|i| matrix[i][j]).sum();
        if (col_sum - 1.0).abs() > tol {
            return false;
        }
    }

    // Check non-negativity
    for row in matrix {
        for &val in row {
            if val < 0.0 {
                return false;
            }
        }
    }

    true
}

proptest! {
    /// Property: N2 mixing matrix is always doubly stochastic for ANY alpha_logit.
    #[test]
    fn n2_mixing_matrix_is_doubly_stochastic(alpha_logit in -10.0f32..10.0f32) {
        let mhc = MhcLiteN2 {
            alpha_logit,
            pre_logits: [0.0; 2],
            pre_bias: [0.0; 2],
            post_logits: [0.0; 2],
            post_bias: [0.0; 2],
        };

        let matrix = mhc.h_res();
        prop_assert!(
            is_doubly_stochastic_2x2(&matrix, 1e-6),
            "N2 matrix not doubly stochastic for alpha_logit={}: {:?}",
            alpha_logit,
            matrix
        );
    }

    /// Property: N2 composite gain is bounded for any sequence of matrices.
    #[test]
    fn n2_composite_gain_bounded(logits in prop::collection::vec(-5.0f32..5.0f32, 1..20)) {
        let matrices: Vec<[[f32; 2]; 2]> = logits
            .iter()
            .map(|&alpha_logit| {
                let mhc = MhcLiteN2 {
                    alpha_logit,
                    pre_logits: [0.0; 2],
                    pre_bias: [0.0; 2],
                    post_logits: [0.0; 2],
                    post_bias: [0.0; 2],
                };
                mhc.h_res()
            })
            .collect();

        // Compute composite: product of all matrices
        let mut composite = [[1.0f32, 0.0], [0.0, 1.0]]; // Identity
        for mat in &matrices {
            let mut result = [[0.0f32; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        result[i][j] += composite[i][k] * mat[k][j];
                    }
                }
            }
            composite = result;
        }

        // Compute Amax gain (max absolute row sum)
        let mut max_row_sum = 0.0f32;
        for row in &composite {
            let row_sum: f32 = row.iter().map(|v| v.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }

        prop_assert!(
            max_row_sum <= 1.0 + 1e-4,
            "Composite gain {} exceeds 1.0 for {} matrices",
            max_row_sum,
            matrices.len()
        );
    }

    /// Property: N4 mixing matrix is always doubly stochastic for ANY logits.
    #[test]
    fn n4_mixing_matrix_is_doubly_stochastic(
        logits in prop::array::uniform24(-5.0f32..5.0f32)
    ) {
        let mhc = MhcLiteN4 {
            res_logits: logits,
            pre_logits: [0.0; 4],
            pre_bias: [0.0; 4],
            post_logits: [0.0; 4],
            post_bias: [0.0; 4],
        };

        let matrix = mhc.h_res();
        prop_assert!(
            is_doubly_stochastic_4x4(&matrix, 1e-5),
            "N4 matrix not doubly stochastic: {:?}",
            matrix
        );
    }

    /// Property: N4 composite gain is bounded.
    #[test]
    fn n4_composite_gain_bounded(
        layer_count in 1usize..10,
        seed in 0u32..100
    ) {
        let mut matrices = Vec::new();
        for i in 0..layer_count {
            let mut logits = [0.0f32; 24];
            for j in 0..24 {
                logits[j] = ((seed * 24 + i as u32 * 24 + j as u32) as f32 * 0.7).sin() * 3.0;
            }
            let mhc = MhcLiteN4 {
                res_logits: logits,
                pre_logits: [0.0; 4],
                pre_bias: [0.0; 4],
                post_logits: [0.0; 4],
                post_bias: [0.0; 4],
            };
            matrices.push(mhc.h_res());
        }

        // Compute composite
        let mut composite = [[0.0f32; 4]; 4];
        for i in 0..4 {
            composite[i][i] = 1.0; // Identity
        }

        for mat in &matrices {
            let mut result = [[0.0f32; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    for k in 0..4 {
                        result[i][j] += composite[i][k] * mat[k][j];
                    }
                }
            }
            composite = result;
        }

        // Compute Amax gain
        let mut max_row_sum = 0.0f32;
        for row in &composite {
            let row_sum: f32 = row.iter().map(|v| v.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }

        prop_assert!(
            max_row_sum <= 1.0 + 1e-3,
            "N4 composite gain {} exceeds 1.0 for {} layers",
            max_row_sum,
            layer_count
        );
    }

    /// Property: Pre-projection is always non-negative.
    #[test]
    fn n2_pre_projection_non_negative(
        pre_logits in prop::array::uniform2(-5.0f32..5.0f32),
        pre_bias in prop::array::uniform2(-2.0f32..2.0f32)
    ) {
        let mhc = MhcLiteN2 {
            alpha_logit: 0.0,
            pre_logits,
            pre_bias,
            post_logits: [0.0; 2],
            post_bias: [0.0; 2],
        };

        let h_pre = mhc.h_pre();
        for &val in &h_pre {
            prop_assert!(val >= 0.0, "Pre-projection negative: {:?}", h_pre);
        }
    }

    /// Property: Post-projection is always non-negative.
    #[test]
    fn n2_post_projection_non_negative(
        post_logits in prop::array::uniform2(-5.0f32..5.0f32),
        post_bias in prop::array::uniform2(-2.0f32..2.0f32)
    ) {
        let mhc = MhcLiteN2 {
            alpha_logit: 0.0,
            pre_logits: [0.0; 2],
            pre_bias: [0.0; 2],
            post_logits,
            post_bias,
        };

        let h_post = mhc.h_post();
        for &val in &h_post {
            prop_assert!(val >= 0.0, "Post-projection negative: {:?}", h_post);
        }
    }
}
