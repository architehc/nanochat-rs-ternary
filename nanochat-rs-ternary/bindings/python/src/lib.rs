//! Python bindings for nanochat via PyO3.
//!
//! Exposes PyModel class with:
//! - load(path): Load model from GGUF + mHC
//! - generate(prompt, max_tokens): Generate text
//! - generate_batch(prompts, max_tokens): Batch generation
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for NanochatModel.
#[pyclass]
struct PyModel {
    model: nanochat_model::model::NanochatModel,
}

#[pymethods]
impl PyModel {
    /// Load a model from GGUF and mHC files.
    ///
    /// Args:
    ///     gguf_path: Path to GGUF model file
    ///     mhc_path: Path to mHC weights file
    ///
    /// Returns:
    ///     PyModel instance
    ///
    /// Example:
    ///     >>> model = PyModel.load("model.gguf", "model.mhc")
    #[staticmethod]
    fn load(gguf_path: &str, mhc_path: &str) -> PyResult<Self> {
        let model = nanochat_model::model::NanochatModel::from_gguf(gguf_path, mhc_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Self { model })
    }

    /// Generate text from a prompt.
    ///
    /// Args:
    ///     prompt: List of token IDs (integers)
    ///     max_tokens: Maximum number of tokens to generate
    ///
    /// Returns:
    ///     List of generated token IDs
    ///
    /// Example:
    ///     >>> tokens = model.generate([1, 2, 3], max_tokens=100)
    fn generate(&mut self, prompt: Vec<u32>, max_tokens: usize) -> PyResult<Vec<u32>> {
        let mut tokens = prompt;
        let mut generated = 0;

        while generated < max_tokens && tokens.len() < self.model.config.max_seq_len {
            let logits = self.model.forward_sequence(&tokens);
            if logits.is_empty() {
                return Err(PyRuntimeError::new_err("Model returned empty logits"));
            }
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(PyRuntimeError::new_err(
                    "Model produced NaN/Inf logits during generation",
                ));
            }

            // Greedy sampling (argmax)
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| PyRuntimeError::new_err("Failed to generate token"))?;

            tokens.push(next_token);
            generated += 1;

            // Stop on EOS (token 0)
            if next_token == 0 {
                break;
            }
        }

        Ok(tokens)
    }

    /// Generate text for multiple prompts (batch processing).
    ///
    /// Args:
    ///     prompts: List of prompts (each prompt is a list of token IDs)
    ///     max_tokens: Maximum tokens to generate per prompt
    ///
    /// Returns:
    ///     List of generated token sequences
    ///
    /// Example:
    ///     >>> results = model.generate_batch([[1, 2], [3, 4]], max_tokens=50)
    fn generate_batch(
        &mut self,
        prompts: Vec<Vec<u32>>,
        max_tokens: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        prompts
            .into_iter()
            .map(|prompt| self.generate(prompt, max_tokens))
            .collect()
    }

    /// Get model configuration as a dictionary.
    ///
    /// Returns:
    ///     Dictionary with model architecture details
    fn get_config(&self) -> PyResult<Vec<(String, usize)>> {
        Ok(vec![
            ("dim".to_string(), self.model.config.dim),
            ("n_layers".to_string(), self.model.config.n_layers),
            ("n_heads".to_string(), self.model.config.n_heads),
            ("vocab_size".to_string(), self.model.config.vocab_size),
            ("max_seq_len".to_string(), self.model.config.max_seq_len),
        ])
    }

    /// Verify mHC doubly stochastic constraints.
    ///
    /// Returns:
    ///     True if all mHC matrices are valid, False otherwise
    fn verify_mhc(&self) -> PyResult<bool> {
        Ok(self.model.verify_mhc().is_ok())
    }

    /// Get model info as a string.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PyModel(dim={}, layers={}, heads={}, vocab={})",
            self.model.config.dim,
            self.model.config.n_layers,
            self.model.config.n_heads,
            self.model.config.vocab_size
        ))
    }
}

/// nanochat_py: Python bindings for nanochat ternary models.
#[pymodule]
fn nanochat_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
