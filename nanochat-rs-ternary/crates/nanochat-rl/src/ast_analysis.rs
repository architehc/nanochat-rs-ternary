//! Deep AST analysis for Rust code quality evaluation
//!
//! This module uses the syn crate to parse Rust code and extract detailed
//! metrics about code structure, complexity, and idiomatic patterns.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use syn::visit::Visit;
use syn::{Expr, File, Item, Pat, Stmt};

/// Comprehensive AST metrics for a code sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstMetrics {
    /// Whether the code could be parsed
    pub parseable: bool,

    /// Complexity metrics
    pub complexity: ComplexityMetrics,

    /// Structure metrics
    pub structure: StructureMetrics,

    /// Quality metrics
    pub quality: QualityMetrics,

    /// Idiomatic Rust patterns
    pub idioms: IdiomMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity (number of decision points)
    pub cyclomatic: usize,

    /// Maximum nesting depth
    pub max_nesting: usize,

    /// Average nesting depth
    pub avg_nesting: f64,

    /// Number of lines of code (excluding comments/whitespace)
    pub loc: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructureMetrics {
    /// Number of functions
    pub n_functions: usize,

    /// Number of structs
    pub n_structs: usize,

    /// Number of enums
    pub n_enums: usize,

    /// Number of traits
    pub n_traits: usize,

    /// Number of impl blocks
    pub n_impls: usize,

    /// Number of modules
    pub n_modules: usize,

    /// Number of use statements
    pub n_uses: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Uses Result for error handling
    pub uses_result: bool,

    /// Uses Option for nullable values
    pub uses_option: bool,

    /// Has documentation comments
    pub has_docs: bool,

    /// Percentage of functions with docs
    pub doc_coverage: f64,

    /// Uses ? operator for error propagation
    pub uses_try_operator: bool,

    /// Has unsafe blocks
    pub has_unsafe: bool,

    /// Number of panics (unwrap, expect, panic!)
    pub n_panics: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IdiomMetrics {
    /// Uses iterator methods (.map, .filter, .collect, etc.)
    pub uses_iterators: bool,

    /// Uses pattern matching (match expressions)
    pub uses_pattern_matching: bool,

    /// Uses destructuring in let bindings
    pub uses_destructuring: bool,

    /// Uses closures
    pub uses_closures: bool,

    /// Uses method chaining
    pub uses_method_chaining: bool,

    /// Uses turbofish syntax (::<Type>)
    pub uses_turbofish: bool,
}

/// Analyze Rust code and extract AST metrics
pub fn analyze_ast(code: &str) -> Result<AstMetrics> {
    // Try to parse the code
    let syntax = match syn::parse_file(code) {
        Ok(file) => file,
        Err(_) => {
            // Code is not parseable
            return Ok(AstMetrics {
                parseable: false,
                complexity: ComplexityMetrics::default(),
                structure: StructureMetrics::default(),
                quality: QualityMetrics::default(),
                idioms: IdiomMetrics::default(),
            });
        }
    };

    // Visit the AST and collect metrics
    let mut visitor = MetricsVisitor::new(code);
    visitor.visit_file(&syntax);

    Ok(AstMetrics {
        parseable: true,
        complexity: visitor.complexity,
        structure: visitor.structure,
        quality: visitor.quality,
        idioms: visitor.idioms,
    })
}

/// AST visitor that collects metrics
struct MetricsVisitor<'a> {
    _code: &'a str,
    complexity: ComplexityMetrics,
    structure: StructureMetrics,
    quality: QualityMetrics,
    idioms: IdiomMetrics,
    current_nesting: usize,
    nesting_samples: Vec<usize>,
    n_functions_with_docs: usize,
}

impl<'a> MetricsVisitor<'a> {
    fn new(code: &'a str) -> Self {
        let loc = code
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with("//")
            })
            .count();

        Self {
            _code: code,
            complexity: ComplexityMetrics {
                cyclomatic: 1, // Base complexity
                max_nesting: 0,
                avg_nesting: 0.0,
                loc,
            },
            structure: StructureMetrics::default(),
            quality: QualityMetrics::default(),
            idioms: IdiomMetrics::default(),
            current_nesting: 0,
            nesting_samples: Vec::new(),
            n_functions_with_docs: 0,
        }
    }

    fn enter_block(&mut self) {
        self.current_nesting += 1;
        self.nesting_samples.push(self.current_nesting);
        if self.current_nesting > self.complexity.max_nesting {
            self.complexity.max_nesting = self.current_nesting;
        }
    }

    fn exit_block(&mut self) {
        if self.current_nesting > 0 {
            self.current_nesting -= 1;
        }
    }

    fn finalize(&mut self) {
        // Calculate average nesting
        if !self.nesting_samples.is_empty() {
            let sum: usize = self.nesting_samples.iter().sum();
            self.complexity.avg_nesting = sum as f64 / self.nesting_samples.len() as f64;
        }

        // Calculate doc coverage
        if self.structure.n_functions > 0 {
            self.quality.doc_coverage =
                self.n_functions_with_docs as f64 / self.structure.n_functions as f64;
        }
    }
}

impl<'a> Visit<'a> for MetricsVisitor<'a> {
    fn visit_item(&mut self, item: &'a Item) {
        match item {
            Item::Fn(func) => {
                self.structure.n_functions += 1;

                // Check for documentation
                if func.attrs.iter().any(|attr| attr.path().is_ident("doc")) {
                    self.n_functions_with_docs += 1;
                    self.quality.has_docs = true;
                }

                // Check return type for Result/Option
                if let syn::ReturnType::Type(_, ty) = &func.sig.output {
                    let type_str = quote::quote!(#ty).to_string();
                    if type_str.contains("Result") {
                        self.quality.uses_result = true;
                    }
                    if type_str.contains("Option") {
                        self.quality.uses_option = true;
                    }
                }
            }
            Item::Struct(_) => self.structure.n_structs += 1,
            Item::Enum(_) => self.structure.n_enums += 1,
            Item::Trait(_) => self.structure.n_traits += 1,
            Item::Impl(_) => self.structure.n_impls += 1,
            Item::Mod(_) => self.structure.n_modules += 1,
            Item::Use(_) => self.structure.n_uses += 1,
            _ => {}
        }

        syn::visit::visit_item(self, item);
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::If(_) | Expr::While(_) | Expr::Loop(_) | Expr::ForLoop(_) => {
                self.complexity.cyclomatic += 1;
                self.enter_block();
            }
            Expr::Match(m) => {
                self.complexity.cyclomatic += m.arms.len();
                self.idioms.uses_pattern_matching = true;
                self.enter_block();
            }
            Expr::Closure(_) => {
                self.idioms.uses_closures = true;
            }
            Expr::MethodCall(mc) => {
                let method = mc.method.to_string();
                // Check for iterator methods
                if ["map", "filter", "fold", "collect", "iter", "into_iter"]
                    .contains(&method.as_str())
                {
                    self.idioms.uses_iterators = true;
                }

                // Check for method chaining (method call on another method call)
                if matches!(*mc.receiver, Expr::MethodCall(_)) {
                    self.idioms.uses_method_chaining = true;
                }
            }
            Expr::Try(_) => {
                self.quality.uses_try_operator = true;
            }
            Expr::Unsafe(_) => {
                self.quality.has_unsafe = true;
            }
            Expr::Call(call) => {
                let func_str = quote::quote!(#call).to_string();
                if func_str.contains("unwrap")
                    || func_str.contains("expect")
                    || func_str.contains("panic")
                {
                    self.quality.n_panics += 1;
                }

                // Check for turbofish
                if func_str.contains("::<") {
                    self.idioms.uses_turbofish = true;
                }
            }
            _ => {}
        }

        syn::visit::visit_expr(self, expr);

        // Exit block for control flow
        match expr {
            Expr::If(_) | Expr::While(_) | Expr::Loop(_) | Expr::ForLoop(_) | Expr::Match(_) => {
                self.exit_block();
            }
            _ => {}
        }
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::Local(local) = stmt {
            // Check for destructuring
            if matches!(local.pat, Pat::Tuple(_) | Pat::Struct(_)) {
                self.idioms.uses_destructuring = true;
            }
        }

        syn::visit::visit_stmt(self, stmt);
    }

    fn visit_file(&mut self, file: &'a File) {
        syn::visit::visit_file(self, file);
        self.finalize();
    }
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic: 0,
            max_nesting: 0,
            avg_nesting: 0.0,
            loc: 0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            uses_result: false,
            uses_option: false,
            has_docs: false,
            doc_coverage: 0.0,
            uses_try_operator: false,
            has_unsafe: false,
            n_panics: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_simple_function() {
        let code = r#"
            /// Adds two numbers
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;

        let metrics = analyze_ast(code).unwrap();
        assert!(metrics.parseable);
        assert_eq!(metrics.structure.n_functions, 1);
        assert!(metrics.quality.has_docs);
    }

    #[test]
    fn test_analyze_with_result() {
        let code = r#"
            pub fn divide(a: i32, b: i32) -> Result<i32, String> {
                if b == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(a / b)
                }
            }
        "#;

        let metrics = analyze_ast(code).unwrap();
        assert!(metrics.parseable);
        assert!(metrics.quality.uses_result);
        assert!(metrics.complexity.cyclomatic >= 2); // base + if
    }

    #[test]
    fn test_analyze_with_iterators() {
        let code = r#"
            pub fn sum_squares(nums: &[i32]) -> i32 {
                nums.iter()
                    .map(|x| x * x)
                    .sum()
            }
        "#;

        let metrics = analyze_ast(code).unwrap();
        assert!(metrics.parseable);
        assert!(metrics.idioms.uses_iterators);
        assert!(metrics.idioms.uses_closures);
        assert!(metrics.idioms.uses_method_chaining);
    }

    #[test]
    fn test_analyze_invalid_code() {
        let code = "fn broken ( {";

        let metrics = analyze_ast(code).unwrap();
        assert!(!metrics.parseable);
    }
}
