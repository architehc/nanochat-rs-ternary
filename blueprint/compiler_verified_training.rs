//! Compiler-Verified Training with Semantic Analysis
//! Integrates Rust compiler and AST analysis for training data verification

use std::collections::HashMap;
use std::process::Command;
use std::io::Write;
use tempfile::NamedTempFile;
use syn::{File, Item, visit::Visit};
use serde::{Serialize, Deserialize};

/// Result of semantic verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    Valid(VerificationMetadata),
    InvalidSyntax(Vec<SyntaxError>),
    CompilationFailed(Vec<CompilerError>),
    SemanticError(Vec<SemanticIssue>),
}

/// Metadata for verified code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMetadata {
    /// Cyclomatic complexity
    pub complexity: usize,

    /// Number of unsafe blocks
    pub unsafe_blocks: usize,

    /// Lifetime complexity score (0-10)
    pub lifetime_complexity: usize,

    /// Number of trait bounds
    pub trait_bounds: usize,

    /// Number of generic parameters
    pub generic_params: usize,

    /// Lines of code
    pub loc: usize,

    /// Compilation time in ms
    pub compile_time_ms: u64,
}

/// Semantic verifier using Rust compiler and AST analysis
pub struct SemanticVerifier {
    /// Cache for verification results
    cache: HashMap<String, VerificationResult>,

    /// Cache size limit
    max_cache_size: usize,

    /// Compilation timeout in seconds
    compile_timeout: u64,
}

impl SemanticVerifier {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_cache_size: 10000,
            compile_timeout: 30,
        }
    }

    /// Verify code with caching
    pub fn verify(&mut self, code: &str) -> VerificationResult {
        // Check cache
        if let Some(result) = self.cache.get(code) {
            return result.clone();
        }

        // Perform verification
        let result = self.perform_verification(code);

        // Cache result if not at capacity
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(code.to_string(), result.clone());
        }

        result
    }

    /// Main verification pipeline
    fn perform_verification(&self, code: &str) -> VerificationResult {
        // Step 1: Fast syntax check
        match self.check_syntax(code) {
            Ok(_) => {}
            Err(errors) => return VerificationResult::InvalidSyntax(errors),
        }

        // Step 2: Compilation check
        let compile_result = self.check_compilation(code);
        if !compile_result.success {
            return VerificationResult::CompilationFailed(compile_result.errors);
        }

        // Step 3: Semantic analysis
        let semantic_result = self.check_semantics(code);
        if !semantic_result.is_sound {
            return VerificationResult::SemanticError(semantic_result.issues);
        }

        // All checks passed
        VerificationResult::Valid(VerificationMetadata {
            complexity: semantic_result.complexity,
            unsafe_blocks: semantic_result.unsafe_blocks,
            lifetime_complexity: semantic_result.lifetime_complexity,
            trait_bounds: semantic_result.trait_bounds,
            generic_params: semantic_result.generic_params,
            loc: code.lines().count(),
            compile_time_ms: compile_result.compile_time_ms,
        })
    }

    /// Fast syntax check using syn
    fn check_syntax(&self, code: &str) -> Result<(), Vec<SyntaxError>> {
        match syn::parse_file(code) {
            Ok(_) => Ok(()),
            Err(e) => Err(vec![SyntaxError {
                message: e.to_string(),
                line: e.span().start().line,
                column: e.span().start().column,
            }]),
        }
    }

    /// Check compilation using rustc
    fn check_compilation(&self, code: &str) -> CompilationResult {
        let start = std::time::Instant::now();

        // Create temp file
        let mut temp_file = match NamedTempFile::with_suffix(".rs") {
            Ok(f) => f,
            Err(e) => {
                return CompilationResult {
                    success: false,
                    errors: vec![CompilerError {
                        message: format!("Failed to create temp file: {}", e),
                        code: None,
                    }],
                    compile_time_ms: 0,
                }
            }
        };

        // Write code to temp file
        if let Err(e) = temp_file.write_all(code.as_bytes()) {
            return CompilationResult {
                success: false,
                errors: vec![CompilerError {
                    message: format!("Failed to write temp file: {}", e),
                    code: None,
                }],
                compile_time_ms: 0,
            };
        }

        let temp_path = temp_file.path().to_string_lossy().to_string();

        // Run rustc
        let output = Command::new("rustc")
            .args(&[
                "--crate-type", "lib",
                "--emit", "metadata",
                "-o", "/dev/null",
                &temp_path,
            ])
            .timeout(std::time::Duration::from_secs(self.compile_timeout))
            .output();

        let compile_time_ms = start.elapsed().as_millis() as u64;

        match output {
            Ok(output) => {
                if output.status.success() {
                    CompilationResult {
                        success: true,
                        errors: vec![],
                        compile_time_ms,
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let errors = self.parse_compiler_errors(&stderr);
                    CompilationResult {
                        success: false,
                        errors,
                        compile_time_ms,
                    }
                }
            }
            Err(e) => {
                CompilationResult {
                    success: false,
                    errors: vec![CompilerError {
                        message: format!("Compilation failed: {}", e),
                        code: None,
                    }],
                    compile_time_ms,
                }
            }
        }
    }

    /// Semantic analysis using AST
    fn check_semantics(&self, code: &str) -> SemanticResult {
        let ast = match syn::parse_file(code) {
            Ok(ast) => ast,
            Err(_) => {
                return SemanticResult {
                    is_sound: false,
                    complexity: 0,
                    unsafe_blocks: 0,
                    lifetime_complexity: 0,
                    trait_bounds: 0,
                    generic_params: 0,
                    issues: vec![],
                }
            }
        };

        // Analyze AST
        let mut analyzer = AstAnalyzer::new();
        analyzer.visit_file(&ast);

        SemanticResult {
            is_sound: analyzer.issues.is_empty(),
            complexity: analyzer.cyclomatic_complexity,
            unsafe_blocks: analyzer.unsafe_blocks,
            lifetime_complexity: analyzer.lifetime_complexity,
            trait_bounds: analyzer.trait_bounds,
            generic_params: analyzer.generic_params,
            issues: analyzer.issues,
        }
    }

    /// Parse rustc error messages
    fn parse_compiler_errors(&self, stderr: &str) -> Vec<CompilerError> {
        stderr.lines()
            .filter(|line| line.contains("error["))
            .map(|line| {
                // Extract error code if present
                let code = if let Some(start) = line.find("error[") {
                    let end = line.find("]").unwrap_or(line.len());
                    Some(line[start+6..end].to_string())
                } else {
                    None
                };

                CompilerError {
                    message: line.to_string(),
                    code,
                }
            })
            .collect()
    }

    /// Batch verification for training data
    pub fn verify_batch(&mut self, codes: &[String]) -> Vec<VerificationResult> {
        use rayon::prelude::*;

        codes.par_iter()
            .map(|code| self.verify(code))
            .collect()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_cache_size,
            hit_rate: self.calculate_hit_rate(),
        }
    }

    fn calculate_hit_rate(&self) -> f64 {
        // Simplified - would track actual hits in production
        0.0
    }
}

/// AST Analyzer for semantic checks
struct AstAnalyzer {
    cyclomatic_complexity: usize,
    unsafe_blocks: usize,
    lifetime_complexity: usize,
    trait_bounds: usize,
    generic_params: usize,
    issues: Vec<SemanticIssue>,
    current_scope: Scope,
}

impl AstAnalyzer {
    fn new() -> Self {
        Self {
            cyclomatic_complexity: 1,
            unsafe_blocks: 0,
            lifetime_complexity: 0,
            trait_bounds: 0,
            generic_params: 0,
            issues: vec![],
            current_scope: Scope::new(),
        }
    }
}

impl<'ast> Visit<'ast> for AstAnalyzer {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        // Analyze function
        self.cyclomatic_complexity += 1;

        // Check for generics
        self.generic_params += node.sig.generics.params.len();

        // Check for trait bounds
        for param in &node.sig.generics.params {
            if let syn::GenericParam::Type(type_param) = param {
                self.trait_bounds += type_param.bounds.len();
            }
        }

        // Check lifetimes
        for param in &node.sig.generics.params {
            if let syn::GenericParam::Lifetime(_) = param {
                self.lifetime_complexity += 1;
            }
        }

        // Continue visiting
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks += 1;

        // Check for unsafe best practices
        if self.current_scope.is_nested_unsafe {
            self.issues.push(SemanticIssue {
                kind: IssueKind::NestedUnsafe,
                message: "Nested unsafe blocks detected".to_string(),
                span: node.unsafe_token.span,
            });
        }

        syn::visit::visit_expr_unsafe(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        self.cyclomatic_complexity += 1;
        syn::visit::visit_expr_if(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.cyclomatic_complexity += 1;
        syn::visit::visit_expr_while(self, node);
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.cyclomatic_complexity += 1;
        syn::visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.cyclomatic_complexity += node.arms.len();
        syn::visit::visit_expr_match(self, node);
    }
}

/// Scope tracking for semantic analysis
struct Scope {
    is_nested_unsafe: bool,
    depth: usize,
}

impl Scope {
    fn new() -> Self {
        Self {
            is_nested_unsafe: false,
            depth: 0,
        }
    }
}

/// Compiler-verified training pipeline
pub struct CompilerVerifiedTraining {
    verifier: SemanticVerifier,
    config: VerifiedTrainingConfig,
}

#[derive(Clone)]
pub struct VerifiedTrainingConfig {
    /// Minimum compilation success rate
    pub min_compile_rate: f64,

    /// Use negative examples for training
    pub use_negative_examples: bool,

    /// Difficulty balancing
    pub difficulty_distribution: HashMap<DifficultyLevel, f64>,

    /// Reward for compilation success
    pub compile_success_reward: f64,

    /// Reward for semantic complexity
    pub complexity_bonus: f64,

    /// Penalty for unsafe code
    pub unsafe_penalty: f64,
}

impl Default for VerifiedTrainingConfig {
    fn default() -> Self {
        let mut difficulty_distribution = HashMap::new();
        difficulty_distribution.insert(DifficultyLevel::Easy, 0.3);
        difficulty_distribution.insert(DifficultyLevel::Medium, 0.5);
        difficulty_distribution.insert(DifficultyLevel::Hard, 0.2);

        Self {
            min_compile_rate: 0.85,
            use_negative_examples: true,
            difficulty_distribution,
            compile_success_reward: 1.0,
            complexity_bonus: 0.01,
            unsafe_penalty: 0.1,
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
}

/// Training example with verification
#[derive(Debug, Clone)]
pub struct VerifiedExample {
    pub prompt: String,
    pub code: String,
    pub verification: VerificationResult,
    pub difficulty: DifficultyLevel,
    pub reward: f64,
}

impl CompilerVerifiedTraining {
    pub fn new(config: VerifiedTrainingConfig) -> Self {
        Self {
            verifier: SemanticVerifier::new(),
            config,
        }
    }

    /// Process raw training data
    pub fn process_dataset(&mut self, raw_data: &[RawExample]) -> Vec<VerifiedExample> {
        let mut verified = Vec::new();
        let mut stats = ProcessingStats::default();

        for example in raw_data {
            stats.total += 1;

            let result = self.verifier.verify(&example.code);

            match &result {
                VerificationResult::Valid(metadata) => {
                    stats.compiled += 1;

                    let difficulty = self.compute_difficulty(metadata);
                    let reward = self.compute_reward(metadata);

                    verified.push(VerifiedExample {
                        prompt: example.prompt.clone(),
                        code: example.code.clone(),
                        verification: result,
                        difficulty,
                        reward,
                    });
                }
                VerificationResult::CompilationFailed(_) => {
                    stats.failed_compile += 1;

                    if self.config.use_negative_examples {
                        verified.push(VerifiedExample {
                            prompt: example.prompt.clone(),
                            code: example.code.clone(),
                            verification: result,
                            difficulty: DifficultyLevel::Easy,
                            reward: 0.1,  // Small reward for negative example
                        });
                    }
                }
                _ => {
                    stats.invalid += 1;
                }
            }
        }

        // Balance by difficulty
        let balanced = self.balance_by_difficulty(verified);

        println!("Processing stats: {:?}", stats);
        println!("Final dataset size: {}", balanced.len());

        balanced
    }

    /// Compute difficulty based on semantic features
    fn compute_difficulty(&self, metadata: &VerificationMetadata) -> DifficultyLevel {
        let score = metadata.complexity as f64 / 100.0
            + metadata.lifetime_complexity as f64 / 10.0
            + metadata.trait_bounds as f64 / 5.0
            + if metadata.unsafe_blocks > 0 { 0.2 } else { 0.0 };

        if score < 0.3 {
            DifficultyLevel::Easy
        } else if score < 0.7 {
            DifficultyLevel::Medium
        } else {
            DifficultyLevel::Hard
        }
    }

    /// Compute reward for verified code
    fn compute_reward(&self, metadata: &VerificationMetadata) -> f64 {
        let mut reward = self.config.compile_success_reward;

        // Bonus for complexity
        reward += metadata.complexity as f64 * self.config.complexity_bonus;

        // Bonus for proper lifetime usage
        if metadata.lifetime_complexity > 0 {
            reward += 0.1;
        }

        // Penalty for unsafe
        if metadata.unsafe_blocks > 0 {
            reward -= metadata.unsafe_blocks as f64 * self.config.unsafe_penalty;
        }

        reward.max(0.0)
    }

    /// Balance dataset by difficulty
    fn balance_by_difficulty(&self, examples: Vec<VerifiedExample>) -> Vec<VerifiedExample> {
        use rand::seq::SliceRandom;

        let mut by_difficulty: HashMap<DifficultyLevel, Vec<VerifiedExample>> = HashMap::new();

        for ex in examples {
            by_difficulty.entry(ex.difficulty.clone()).or_default().push(ex);
        }

        let mut balanced = Vec::new();
        let total_target = 100000;  // Target dataset size

        for (level, target_pct) in &self.config.difficulty_distribution {
            let target_count = (total_target as f64 * target_pct) as usize;

            if let Some(mut examples) = by_difficulty.remove(level) {
                // Shuffle and sample
                let mut rng = rand::thread_rng();
                examples.shuffle(&mut rng);

                let sampled: Vec<_> = examples.into_iter()
                    .take(target_count)
                    .collect();

                balanced.extend(sampled);
            }
        }

        // Shuffle final dataset
        let mut rng = rand::thread_rng();
        balanced.shuffle(&mut rng);

        balanced
    }
}

/// Raw training example
#[derive(Debug, Clone)]
pub struct RawExample {
    pub prompt: String,
    pub code: String,
}

/// Processing statistics
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub total: usize,
    pub compiled: usize,
    pub failed_compile: usize,
    pub invalid: usize,
}

/// Error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerError {
    pub message: String,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIssue {
    pub kind: IssueKind,
    pub message: String,
    pub span: proc_macro2::Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueKind {
    NestedUnsafe,
    UnusedVariable,
    TypeMismatch,
    LifetimeError,
    OwnershipViolation,
}

/// Compilation result
#[derive(Debug)]
pub struct CompilationResult {
    pub success: bool,
    pub errors: Vec<CompilerError>,
    pub compile_time_ms: u64,
}

/// Semantic analysis result
#[derive(Debug)]
pub struct SemanticResult {
    pub is_sound: bool,
    pub complexity: usize,
    pub unsafe_blocks: usize,
    pub lifetime_complexity: usize,
    pub trait_bounds: usize,
    pub generic_params: usize,
    pub issues: Vec<SemanticIssue>,
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hit_rate: f64,
}
