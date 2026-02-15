//! Dataset loaders for HumanEval and MBPP code generation benchmarks.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A code generation problem with test cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProblem {
    /// Unique problem identifier (e.g., "HumanEval/0")
    pub task_id: String,

    /// Problem description and instructions
    pub prompt: String,

    /// Function signature (entry point)
    pub entry_point: String,

    /// Canonical solution (for reference)
    pub canonical_solution: Option<String>,

    /// Test cases (code to run after solution)
    pub test: String,

    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// HumanEval dataset (164 problems).
pub struct HumanEvalDataset {
    problems: Vec<CodeProblem>,
}

impl HumanEvalDataset {
    /// Load HumanEval from JSONL file.
    ///
    /// Expected format (one JSON object per line):
    /// ```json
    /// {
    ///   "task_id": "HumanEval/0",
    ///   "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    ///   "entry_point": "has_close_elements",
    ///   "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
    ///   "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
    /// }
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut problems = Vec::new();

        for (i, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<CodeProblem>(line) {
                Ok(problem) => problems.push(problem),
                Err(e) => {
                    eprintln!("Warning: Failed to parse line {}: {}", i + 1, e);
                }
            }
        }

        if problems.is_empty() {
            anyhow::bail!("No problems loaded from HumanEval dataset");
        }

        Ok(Self { problems })
    }

    /// Load from standard HumanEval URL (downloads if not cached).
    pub async fn load_from_url() -> Result<Self, anyhow::Error> {
        const _HUMANEVAL_URL: &str =
            "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz";

        // TODO: Download and cache
        anyhow::bail!("URL loading not implemented yet - please download manually");
    }

    pub fn problems(&self) -> &[CodeProblem] {
        &self.problems
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
}

/// MBPP dataset (974 problems).
pub struct MBPPDataset {
    problems: Vec<CodeProblem>,
}

impl MBPPDataset {
    /// Load MBPP from JSON file.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {
    ///     "task_id": 1,
    ///     "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
    ///     "code": "def min_cost(cost, m, n): ...",
    ///     "test_list": ["assert min_cost(...) == ..."],
    ///     "test_setup_code": "",
    ///     "challenge_test_list": []
    ///   }
    /// ]
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let raw_problems: Vec<serde_json::Value> = serde_json::from_str(&content)?;

        let mut problems = Vec::new();
        for (i, raw) in raw_problems.iter().enumerate() {
            // Convert MBPP format to CodeProblem format
            let task_id = format!("MBPP/{}", raw["task_id"].as_u64().unwrap_or(i as u64));

            let prompt = raw["text"].as_str().unwrap_or("").to_string();

            // Extract function name from code
            let code = raw["code"].as_str().unwrap_or("");
            let entry_point = extract_function_name(code).unwrap_or_else(|| "solution".to_string());

            // Convert test_list to test code
            let empty_vec = vec![];
            let test_list = raw["test_list"].as_array().unwrap_or(&empty_vec);
            let test_setup = raw["test_setup_code"].as_str().unwrap_or("");
            let test = format!(
                "{}\ndef check(candidate):\n{}",
                test_setup,
                test_list
                    .iter()
                    .filter_map(|t| t.as_str())
                    .map(|t| format!("    {}", t))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            problems.push(CodeProblem {
                task_id,
                prompt,
                entry_point,
                canonical_solution: raw["code"].as_str().map(|s| s.to_string()),
                test,
                metadata: raw.clone(),
            });
        }

        if problems.is_empty() {
            anyhow::bail!("No problems loaded from MBPP dataset");
        }

        Ok(Self { problems })
    }

    pub fn problems(&self) -> &[CodeProblem] {
        &self.problems
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
}

/// Extract function name from Python code.
fn extract_function_name(code: &str) -> Option<String> {
    use regex::Regex;
    let re = Regex::new(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").ok()?;
    re.captures(code)?.get(1).map(|m| m.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_function_name() {
        assert_eq!(
            extract_function_name("def has_close_elements(numbers, threshold):"),
            Some("has_close_elements".to_string())
        );
        assert_eq!(
            extract_function_name("def _private_func():"),
            Some("_private_func".to_string())
        );
        assert_eq!(extract_function_name("class Foo:"), None);
    }

    #[test]
    fn test_humaneval_parse() {
        let json = r#"{"task_id": "HumanEval/0", "prompt": "def test():\n    pass\n", "entry_point": "test", "canonical_solution": "    return True\n", "test": "def check(candidate):\n    assert candidate() == True\n"}"#;
        let problem: CodeProblem = serde_json::from_str(json).unwrap();
        assert_eq!(problem.task_id, "HumanEval/0");
        assert_eq!(problem.entry_point, "test");
    }
}
