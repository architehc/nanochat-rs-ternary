//! Python code execution for testing generated solutions.
//!
//! **WARNING:** This executor runs generated code in a subprocess with minimal
//! isolation (cleared environment, stdin closed). It does NOT provide a true
//! sandbox — there is no seccomp, chroot, namespace, or cgroup isolation.
//! Do not run untrusted code in security-sensitive environments without
//! additional OS-level sandboxing.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// Result of executing generated code with tests.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Whether all tests passed
    pub passed: bool,

    /// Execution time in milliseconds
    pub time_ms: u64,

    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Exit code
    pub exit_code: Option<i32>,

    /// Error type (timeout, runtime error, etc.)
    pub error_type: Option<ErrorType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorType {
    Timeout,
    RuntimeError,
    SyntaxError,
    AssertionError,
    ImportError,
    Other,
}

impl ErrorType {
    fn from_stderr(stderr: &str) -> Self {
        if stderr.contains("TimeoutError") || stderr.contains("timed out") {
            ErrorType::Timeout
        } else if stderr.contains("SyntaxError") {
            ErrorType::SyntaxError
        } else if stderr.contains("AssertionError") {
            ErrorType::AssertionError
        } else if stderr.contains("ImportError") || stderr.contains("ModuleNotFoundError") {
            ErrorType::ImportError
        } else if !stderr.is_empty() {
            ErrorType::RuntimeError
        } else {
            ErrorType::Other
        }
    }
}

/// Python code executor (unsandboxed).
///
/// Runs generated Python code in a subprocess. Environment variables are
/// cleared to reduce information leakage, but no OS-level sandboxing is
/// applied. See module-level documentation for security caveats.
pub struct CodeExecutor {
    /// Python interpreter path (default: "python3")
    python_cmd: String,

    /// Timeout for execution (seconds)
    timeout_secs: u64,

    /// Temporary directory for code files
    _temp_dir: Option<PathBuf>,
}

impl Default for CodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeExecutor {
    pub fn new() -> Self {
        Self {
            python_cmd: "python3".to_string(),
            timeout_secs: 5,
            _temp_dir: None,
        }
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_python_cmd(mut self, cmd: String) -> Self {
        self.python_cmd = cmd;
        self
    }

    /// Execute generated code with test cases.
    ///
    /// Combines the generated solution with test code and runs in a subprocess
    /// with a cleared environment. Uses `kill_on_drop(true)` so the child
    /// process is reliably killed via SIGKILL if the timeout fires.
    pub async fn execute(
        &self,
        solution: &str,
        test_code: &str,
        entry_point: &str,
    ) -> Result<ExecutionResult, anyhow::Error> {
        let start = std::time::Instant::now();

        // Construct full test program
        let program = format!(
            "{}\n\n{}\n\nif __name__ == '__main__':\n    check({})\n    print('TESTS_PASSED')\n",
            solution, test_code, entry_point
        );

        // Preserve PATH so we can find the Python interpreter
        let path_env = std::env::var("PATH")
            .unwrap_or_else(|_| "/usr/bin:/usr/local/bin:/bin".to_string());

        // Execute with timeout; env_clear reduces information leakage,
        // kill_on_drop ensures reliable cleanup on timeout/cancellation.
        let timeout = Duration::from_secs(self.timeout_secs);
        let child = Command::new(&self.python_cmd)
            .arg("-c")
            .arg(&program)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .env_clear()
            .env("PATH", &path_env)
            .kill_on_drop(true)
            .spawn()?;

        // Wait with timeout — on cancellation, child is dropped and SIGKILL'd
        let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        let time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code();

                let passed = output.status.success() && stdout.contains("TESTS_PASSED");
                let error_type = if !passed {
                    Some(ErrorType::from_stderr(&stderr))
                } else {
                    None
                };

                Ok(ExecutionResult {
                    passed,
                    time_ms,
                    stdout,
                    stderr,
                    exit_code,
                    error_type,
                })
            }
            Ok(Err(e)) => {
                // Process error
                Ok(ExecutionResult {
                    passed: false,
                    time_ms,
                    stdout: String::new(),
                    stderr: format!("Process error: {}", e),
                    exit_code: None,
                    error_type: Some(ErrorType::RuntimeError),
                })
            }
            Err(_) => {
                // Timeout — child dropped here, kill_on_drop sends SIGKILL
                Ok(ExecutionResult {
                    passed: false,
                    time_ms,
                    stdout: String::new(),
                    stderr: "Execution timed out".to_string(),
                    exit_code: None,
                    error_type: Some(ErrorType::Timeout),
                })
            }
        }
    }

    /// Execute multiple solutions and return all results.
    pub async fn execute_batch(
        &self,
        solutions: &[String],
        test_code: &str,
        entry_point: &str,
    ) -> Vec<Result<ExecutionResult, anyhow::Error>> {
        let mut results = Vec::new();
        for solution in solutions {
            results.push(self.execute(solution, test_code, entry_point).await);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_successful_execution() {
        let executor = CodeExecutor::new();
        let solution = "def add(a, b):\n    return a + b";
        let test = "def check(candidate):\n    assert candidate(1, 2) == 3";

        let result = executor.execute(solution, test, "add").await.unwrap();
        assert!(result.passed, "stderr: {}", result.stderr);
    }

    #[tokio::test]
    async fn test_failed_assertion() {
        let executor = CodeExecutor::new();
        let solution = "def add(a, b):\n    return a - b"; // Wrong!
        let test = "def check(candidate):\n    assert candidate(1, 2) == 3";

        let result = executor.execute(solution, test, "add").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.error_type, Some(ErrorType::AssertionError));
    }

    #[tokio::test]
    async fn test_syntax_error() {
        let executor = CodeExecutor::new();
        let solution = "def add(a, b)\n    return a + b"; // Missing colon
        let test = "def check(candidate):\n    assert candidate(1, 2) == 3";

        let result = executor.execute(solution, test, "add").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.error_type, Some(ErrorType::SyntaxError));
    }

    #[tokio::test]
    async fn test_timeout() {
        let executor = CodeExecutor::new().with_timeout(1);
        let solution = "def infinite():\n    while True:\n        pass";
        let test = "def check(candidate):\n    candidate()";

        let result = executor.execute(solution, test, "infinite").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.error_type, Some(ErrorType::Timeout));
    }
}
