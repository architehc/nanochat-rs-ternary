// verify.rs — Doubly stochastic property checking and composite gain diagnostics
//
// All mHC matrices must be doubly stochastic:
//   - All entries >= 0
//   - Each row sums to 1
//   - Each column sums to 1
//
// The BvN parameterization guarantees this by construction, but we verify
// at runtime as a safety net.

/// Verify that a 4x4 matrix is doubly stochastic within tolerance.
///
/// Checks:
/// 1. Non-negativity of all entries
/// 2. Row sums equal to 1.0
/// 3. Column sums equal to 1.0
pub fn verify_doubly_stochastic(mat: &[[f32; 4]; 4], tol: f32) -> Result<(), String> {
    // Check non-negativity
    for i in 0..4 {
        for j in 0..4 {
            if mat[i][j] < -tol {
                return Err(format!(
                    "Negative entry: mat[{}][{}] = {}",
                    i, j, mat[i][j]
                ));
            }
        }
    }

    // Check row sums = 1
    for i in 0..4 {
        let row_sum: f32 = mat[i].iter().sum();
        if (row_sum - 1.0).abs() > tol {
            return Err(format!("Row {} sum = {} (expected 1.0)", i, row_sum));
        }
    }

    // Check column sums = 1
    for j in 0..4 {
        let col_sum: f32 = (0..4).map(|i| mat[i][j]).sum();
        if (col_sum - 1.0).abs() > tol {
            return Err(format!("Col {} sum = {} (expected 1.0)", j, col_sum));
        }
    }

    Ok(())
}

/// Verify that a 2x2 matrix is doubly stochastic within tolerance.
pub fn verify_doubly_stochastic_2x2(mat: &[[f32; 2]; 2], tol: f32) -> Result<(), String> {
    for i in 0..2 {
        for j in 0..2 {
            if mat[i][j] < -tol {
                return Err(format!(
                    "Negative entry: mat[{}][{}] = {}",
                    i, j, mat[i][j]
                ));
            }
        }
    }
    let r0: f32 = mat[0][0] + mat[0][1];
    let r1: f32 = mat[1][0] + mat[1][1];
    let c0: f32 = mat[0][0] + mat[1][0];
    let c1: f32 = mat[0][1] + mat[1][1];

    if (r0 - 1.0).abs() > tol || (r1 - 1.0).abs() > tol {
        return Err(format!("Row sums: [{}, {}] (expected 1.0)", r0, r1));
    }
    if (c0 - 1.0).abs() > tol || (c1 - 1.0).abs() > tol {
        return Err(format!("Col sums: [{}, {}] (expected 1.0)", c0, c1));
    }
    Ok(())
}

/// Compute composite gain (Amax Gain Magnitude) for a sequence of H_res matrices.
///
/// This is the diagnostic metric from the mHC paper.
/// Amax = max(max_row_sum, max_col_sum) of the composite product.
///
/// For exact doubly stochastic matrices, the composite product is also
/// doubly stochastic, so the gain should be <= 1.0.
///
/// In contrast, unconstrained HC (non-DS) can reach gain > 3000 at depth 64.
pub fn composite_amax_gain(matrices: &[[[f32; 4]; 4]]) -> f32 {
    let mut composite = [[0.0f32; 4]; 4];
    // Start with identity
    for i in 0..4 {
        composite[i][i] = 1.0;
    }

    for mat in matrices {
        let prev = composite;
        composite = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    composite[i][j] += prev[i][k] * mat[k][j];
                }
            }
        }
    }

    // Amax = max of (max row sum, max col sum)
    let max_row_sum = (0..4)
        .map(|i| composite[i].iter().map(|v| v.abs()).sum::<f32>())
        .fold(0.0f32, f32::max);

    let max_col_sum = (0..4)
        .map(|j| (0..4).map(|i| composite[i][j].abs()).sum::<f32>())
        .fold(0.0f32, f32::max);

    f32::max(max_row_sum, max_col_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_identity_is_ds() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        verify_doubly_stochastic(&identity, 1e-6).unwrap();
    }

    #[test]
    fn test_verify_uniform_is_ds() {
        let uniform = [[0.25f32; 4]; 4];
        verify_doubly_stochastic(&uniform, 1e-6).unwrap();
    }

    #[test]
    fn test_verify_rejects_negative() {
        let bad = [
            [1.5, -0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(verify_doubly_stochastic(&bad, 1e-6).is_err());
    }

    #[test]
    fn test_verify_rejects_bad_row_sum() {
        let bad = [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0], // row sum = 0.5
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(verify_doubly_stochastic(&bad, 1e-6).is_err());
    }

    #[test]
    fn test_verify_2x2_identity() {
        let identity = [[1.0, 0.0], [0.0, 1.0]];
        verify_doubly_stochastic_2x2(&identity, 1e-6).unwrap();
    }

    #[test]
    fn test_verify_2x2_swap() {
        let swap = [[0.0, 1.0], [1.0, 0.0]];
        verify_doubly_stochastic_2x2(&swap, 1e-6).unwrap();
    }

    #[test]
    fn test_verify_2x2_mixed() {
        let mat = [[0.7, 0.3], [0.3, 0.7]];
        verify_doubly_stochastic_2x2(&mat, 1e-6).unwrap();
    }

    #[test]
    fn test_verify_2x2_rejects_bad() {
        let bad = [[0.5, 0.3], [0.3, 0.7]];
        assert!(verify_doubly_stochastic_2x2(&bad, 1e-6).is_err());
    }

    #[test]
    fn test_composite_gain_identity_sequence() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let matrices = vec![identity; 100];
        let gain = composite_amax_gain(&matrices);
        assert!((gain - 1.0).abs() < 1e-5, "Identity gain should be 1.0, got {}", gain);
    }

    #[test]
    fn test_composite_gain_uniform_sequence() {
        let uniform = [[0.25f32; 4]; 4];
        let matrices = vec![uniform; 64];
        let gain = composite_amax_gain(&matrices);
        assert!(
            gain <= 1.0 + 1e-4,
            "DS composite gain should be <= 1.0, got {}",
            gain
        );
    }

    #[test]
    fn test_composite_gain_empty() {
        let gain = composite_amax_gain(&[]);
        // Empty sequence -> identity -> gain = 1.0
        assert!((gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_verify_4x4_rejects_bad_col_sum() {
        // Good row sums but bad column sums
        let bad = [
            [0.4, 0.6, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.4, 0.6],
        ];
        // Row sums are all 1.0, but col sums are [0.8, 1.2, 0.8, 1.2]
        let result = verify_doubly_stochastic(&bad, 1e-6);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Col"), "Error: {}", err);
    }

    #[test]
    fn test_verify_2x2_rejects_negative() {
        let bad = [[-0.5, 1.5], [1.5, -0.5]];
        let result = verify_doubly_stochastic_2x2(&bad, 1e-6);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Negative"), "Error: {}", err);
    }

    #[test]
    fn test_verify_2x2_rejects_bad_col_sum() {
        // Row sums are both 1.0 but cols don't match
        let bad = [[0.8, 0.2], [0.8, 0.2]];
        let result = verify_doubly_stochastic_2x2(&bad, 1e-6);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Col") || err.contains("sum"), "Error: {}", err);
    }

    #[test]
    fn test_verify_2x2_rejects_bad_row_sum() {
        let bad = [[0.5, 0.3], [0.5, 0.7]];
        let result = verify_doubly_stochastic_2x2(&bad, 1e-6);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Row") || err.contains("sum"), "Error: {}", err);
    }
}
