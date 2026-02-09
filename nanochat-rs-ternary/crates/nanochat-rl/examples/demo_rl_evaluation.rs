//! Demo: Show detailed RL evaluation for different code samples

use nanochat_rl::{CompilerFeedback, analyze_ast, compute_reward, RewardConfig};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  RL Code Evaluation Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    let compiler = CompilerFeedback::new()?;
    let reward_config = RewardConfig::default();

    // Example 1: Good idiomatic Rust
    println!("━━━ Example 1: Idiomatic Rust with iterators ━━━");
    let good_code = r#"
pub fn sum_squares(nums: &[i32]) -> i32 {
    nums.iter()
        .map(|x| x * x)
        .sum()
}

pub fn filter_positive(nums: Vec<i32>) -> Vec<i32> {
    nums.into_iter()
        .filter(|&x| x > 0)
        .collect()
}
"#;
    evaluate_sample(&compiler, good_code, &reward_config)?;

    // Example 2: Code with proper error handling
    println!("\n━━━ Example 2: Proper error handling with Result ━━━");
    let error_handling_code = r#"
use std::fs;
use std::io;

/// Reads a file and returns its contents
pub fn read_config(path: &str) -> io::Result<String> {
    let contents = fs::read_to_string(path)?;
    Ok(contents.trim().to_string())
}

pub fn parse_number(s: &str) -> Result<i32, std::num::ParseIntError> {
    s.parse()
}
"#;
    evaluate_sample(&compiler, error_handling_code, &reward_config)?;

    // Example 3: Bad code with panics
    println!("\n━━━ Example 3: Code with panics (penalized) ━━━");
    let bad_code = r#"
pub fn divide(a: i32, b: i32) -> i32 {
    a / b  // Can panic if b == 0!
}

pub fn get_first(nums: Vec<i32>) -> i32 {
    nums[0]  // Can panic if empty!
}

pub fn parse_unwrap(s: &str) -> i32 {
    s.parse().unwrap()  // Panic if parse fails!
}
"#;
    evaluate_sample(&compiler, bad_code, &reward_config)?;

    // Example 4: Broken code that doesn't compile
    println!("\n━━━ Example 4: Code with compilation errors ━━━");
    let broken_code = r#"
pub fn broken() {
    let x = 5  // Missing semicolon
    let y = x + 1;
}

pub fn wrong_type() -> String {
    42  // Type mismatch
}
"#;
    evaluate_sample(&compiler, broken_code, &reward_config)?;

    // Example 5: Complex nested code
    println!("\n━━━ Example 5: High complexity (penalized) ━━━");
    let complex_code = r#"
pub fn complex_logic(x: i32) -> i32 {
    if x > 0 {
        if x < 10 {
            if x % 2 == 0 {
                if x > 5 {
                    return x * 2;
                } else {
                    return x + 1;
                }
            } else {
                return x - 1;
            }
        } else {
            return x / 2;
        }
    } else {
        return 0;
    }
}
"#;
    evaluate_sample(&compiler, complex_code, &reward_config)?;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Summary: The RL system evaluates multiple dimensions");
    println!("═══════════════════════════════════════════════════════════");
    println!("✓ Compilation success (most important)");
    println!("✓ AST parseability");
    println!("✓ Idiomatic patterns (iterators, pattern matching, closures)");
    println!("✓ Quality metrics (Result/Option, documentation, error handling)");
    println!("✓ Complexity (low cyclomatic, reasonable nesting)");
    println!("✗ Penalizes: panics (unwrap/expect), unsafe, high complexity");

    Ok(())
}

fn evaluate_sample(
    compiler: &CompilerFeedback,
    code: &str,
    config: &RewardConfig,
) -> Result<()> {
    // Compile
    let compile_result = compiler.compile(code)?;
    println!("Compilation: {}", if compile_result.success { "✓ SUCCESS" } else { "✗ FAILED" });
    if compile_result.n_errors > 0 {
        println!("  Errors: {}", compile_result.n_errors);
    }
    if compile_result.n_warnings > 0 {
        println!("  Warnings: {}", compile_result.n_warnings);
    }

    // AST analysis
    let ast_metrics = analyze_ast(code)?;
    if ast_metrics.parseable {
        println!("\nAST Analysis:");
        println!("  Complexity:");
        println!("    Cyclomatic: {}", ast_metrics.complexity.cyclomatic);
        println!("    Max nesting: {}", ast_metrics.complexity.max_nesting);
        println!("    LOC: {}", ast_metrics.complexity.loc);

        println!("  Structure:");
        println!("    Functions: {}", ast_metrics.structure.n_functions);
        println!("    Structs: {}", ast_metrics.structure.n_structs);
        println!("    Uses statements: {}", ast_metrics.structure.n_uses);

        println!("  Quality:");
        println!("    Uses Result: {}", ast_metrics.quality.uses_result);
        println!("    Uses Option: {}", ast_metrics.quality.uses_option);
        println!("    Has docs: {}", ast_metrics.quality.has_docs);
        println!("    Uses ? operator: {}", ast_metrics.quality.uses_try_operator);
        println!("    Panics (unwrap/expect): {}", ast_metrics.quality.n_panics);
        println!("    Has unsafe: {}", ast_metrics.quality.has_unsafe);

        println!("  Idiomatic Rust:");
        println!("    Uses iterators: {}", ast_metrics.idioms.uses_iterators);
        println!("    Pattern matching: {}", ast_metrics.idioms.uses_pattern_matching);
        println!("    Closures: {}", ast_metrics.idioms.uses_closures);
        println!("    Method chaining: {}", ast_metrics.idioms.uses_method_chaining);
    } else {
        println!("\nAST Analysis: ✗ NOT PARSEABLE");
    }

    // Compute reward
    let reward = compute_reward(&compile_result, &ast_metrics, config);
    println!("\n🎯 TOTAL REWARD: {:.2}", reward);

    if reward > 20.0 {
        println!("   Rating: ⭐⭐⭐ EXCELLENT");
    } else if reward > 10.0 {
        println!("   Rating: ⭐⭐ GOOD");
    } else if reward > 0.0 {
        println!("   Rating: ⭐ ACCEPTABLE");
    } else {
        println!("   Rating: ❌ POOR");
    }

    Ok(())
}
