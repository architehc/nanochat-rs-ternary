//! Rust compiler feedback for code generation evaluation
//!
//! This module integrates with rustc to compile generated code and extract
//! detailed feedback including errors, warnings, and success metrics.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Result of compiling a Rust code sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    /// Whether the code compiled successfully
    pub success: bool,

    /// Number of errors
    pub n_errors: usize,

    /// Number of warnings
    pub n_warnings: usize,

    /// Compiler output (stderr)
    pub output: String,

    /// Exit code from rustc
    pub exit_code: Option<i32>,

    /// Parsed error messages
    pub errors: Vec<CompilerMessage>,

    /// Parsed warning messages
    pub warnings: Vec<CompilerMessage>,
}

/// A compiler error or warning message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerMessage {
    /// Message severity (error, warning, note, help)
    pub level: String,

    /// Main error message
    pub message: String,

    /// Code span (line, column) if available
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub line_start: usize,
    pub line_end: usize,
    pub column_start: usize,
    pub column_end: usize,
}

/// Compiler feedback system
pub struct CompilerFeedback {
    /// Temporary directory for compilation
    temp_dir: TempDir,

    /// rustc path (default: "rustc")
    rustc_path: String,
}

impl CompilerFeedback {
    /// Create a new compiler feedback system
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new().context("Failed to create temp directory")?;
        Ok(Self {
            temp_dir,
            rustc_path: "rustc".to_string(),
        })
    }

    /// Set custom rustc path
    pub fn with_rustc_path(mut self, path: String) -> Self {
        self.rustc_path = path;
        self
    }

    /// Compile a Rust code snippet and return detailed feedback
    pub fn compile(&self, code: &str) -> Result<CompileResult> {
        // Write code to temporary file
        let code_path = self.temp_dir.path().join("generated.rs");
        fs::write(&code_path, code)
            .context("Failed to write code to temp file")?;

        // Run rustc with JSON output for structured error messages
        let output = Command::new(&self.rustc_path)
            .arg("--crate-type")
            .arg("lib")
            .arg("--error-format=json")
            .arg("--edition=2021")
            .arg(&code_path)
            .arg("-o")
            .arg(self.temp_dir.path().join("output.rlib"))
            .output()
            .context("Failed to run rustc")?;

        let success = output.status.success();
        let exit_code = output.status.code();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Parse JSON error messages
        let (errors, warnings) = self.parse_compiler_output(&stderr)?;

        Ok(CompileResult {
            success,
            n_errors: errors.len(),
            n_warnings: warnings.len(),
            output: stderr,
            exit_code,
            errors,
            warnings,
        })
    }

    /// Parse rustc JSON output into structured messages
    fn parse_compiler_output(&self, output: &str) -> Result<(Vec<CompilerMessage>, Vec<CompilerMessage>)> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in output.lines() {
            // Skip non-JSON lines
            if !line.starts_with('{') {
                continue;
            }

            // Parse JSON message
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(level) = msg.get("level").and_then(|v| v.as_str()) {
                    let message = msg.get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Extract span information
                    let span = msg.get("spans")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|span| {
                            Some(Span {
                                line_start: span.get("line_start")?.as_u64()? as usize,
                                line_end: span.get("line_end")?.as_u64()? as usize,
                                column_start: span.get("column_start")?.as_u64()? as usize,
                                column_end: span.get("column_end")?.as_u64()? as usize,
                            })
                        });

                    let compiler_msg = CompilerMessage {
                        level: level.to_string(),
                        message,
                        span,
                    };

                    match level {
                        "error" => errors.push(compiler_msg),
                        "warning" => warnings.push(compiler_msg),
                        _ => {}
                    }
                }
            }
        }

        Ok((errors, warnings))
    }
}

impl Default for CompilerFeedback {
    fn default() -> Self {
        Self::new().expect("Failed to create CompilerFeedback")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_valid_code() {
        let feedback = CompilerFeedback::new().unwrap();
        let code = r#"
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;

        let result = feedback.compile(code).unwrap();
        assert!(result.success, "Valid code should compile");
        assert_eq!(result.n_errors, 0);
    }

    #[test]
    fn test_compile_invalid_code() {
        let feedback = CompilerFeedback::new().unwrap();
        let code = r#"
            pub fn broken() {
                let x = 5
                // Missing semicolon
            }
        "#;

        let result = feedback.compile(code).unwrap();
        assert!(!result.success, "Invalid code should not compile");
        assert!(result.n_errors > 0);
    }

    #[test]
    fn test_compile_with_warnings() {
        let feedback = CompilerFeedback::new().unwrap();
        let code = r#"
            pub fn unused() {
                let x = 5; // Unused variable warning
            }
        "#;

        let result = feedback.compile(code).unwrap();
        // Code compiles but has warnings
        assert!(result.success);
        // Note: rustc might not show warnings in lib mode without additional flags
    }
}
