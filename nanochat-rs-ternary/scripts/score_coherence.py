#!/usr/bin/env python3
"""Score model generations for coherence metrics."""

import re
import sys
import os
from collections import Counter
from pathlib import Path

def count_brackets(text):
    """Count bracket balance and matching."""
    pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    opens = Counter()
    closes = Counter()
    for ch in text:
        if ch in pairs:
            opens[ch] += 1
        elif ch in pairs.values():
            closes[ch] += 1
    total_open = sum(opens.values())
    total_close = sum(closes.values())
    balance = abs(total_open - total_close)
    # Score: 1.0 = perfect balance, 0.0 = terrible
    if total_open + total_close == 0:
        return 1.0, 0
    score = 1.0 - min(balance / max(total_open + total_close, 1), 1.0)
    return score, balance

def repetition_score(text, n=3):
    """Measure n-gram repetition. Lower = more repetitive."""
    words = text.split()
    if len(words) < n + 1:
        return 1.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    unique = len(set(ngrams))
    total = len(ngrams)
    return unique / total if total > 0 else 1.0

def rust_keyword_ratio(text):
    """Ratio of real Rust keywords/patterns found."""
    keywords = {
        'fn', 'let', 'mut', 'pub', 'struct', 'impl', 'enum', 'trait',
        'match', 'if', 'else', 'for', 'while', 'loop', 'return',
        'use', 'mod', 'crate', 'self', 'Self', 'where', 'type',
        'async', 'await', 'unsafe', 'const', 'static', 'ref',
        'Ok', 'Err', 'Some', 'None', 'Result', 'Option', 'Vec',
        'String', 'str', 'bool', 'usize', 'i32', 'u8', 'f32',
    }
    words = set(re.findall(r'\b\w+\b', text))
    if not words:
        return 0.0
    found = words & keywords
    return len(found) / min(len(words), 50)  # cap denominator

def invented_token_ratio(text):
    """Detect likely invented/garbled tokens."""
    words = re.findall(r'\b[a-zA-Z_]\w*\b', text)
    if not words:
        return 0.0
    # Heuristics for "invented" tokens
    invented = 0
    for w in words:
        # Very long tokens (>25 chars) are likely garbled
        if len(w) > 25:
            invented += 1
        # Tokens with unusual character patterns
        elif re.match(r'^[a-z]+[A-Z]{3,}', w):  # camelCase with 3+ caps
            invented += 1
        elif w.count('_') > 4:  # too many underscores
            invented += 1
    return 1.0 - (invented / len(words))

def syntactic_coherence(text):
    """Check for basic syntactic patterns being correct."""
    score = 0
    checks = 0

    # Check: fn keyword followed by name and (
    checks += 1
    if re.search(r'\bfn\s+\w+\s*[(<]', text):
        score += 1

    # Check: let binding
    checks += 1
    if re.search(r'\blet\s+(mut\s+)?\w+\s*[=:]', text):
        score += 1

    # Check: struct with fields
    checks += 1
    if re.search(r'\bstruct\s+\w+\s*[{<(]', text):
        score += 1

    # Check: impl block
    checks += 1
    if re.search(r'\bimpl\s+(<[^>]+>\s*)?\w+', text):
        score += 1

    # Check: type annotations
    checks += 1
    if re.search(r':\s*((&\s*(mut\s+)?)?\w+|Vec<|Option<|Result<|Box<)', text):
        score += 1

    # Check: method calls
    checks += 1
    if re.search(r'\.\w+\(', text):
        score += 1

    # Check: semicolons at statement ends
    checks += 1
    if text.count(';') >= 2:
        score += 1

    # Check: proper string literals
    checks += 1
    if re.search(r'"[^"]*"', text):
        score += 1

    return score / checks if checks > 0 else 0

def parse_eval_file(filepath):
    """Parse an evaluation output file into prompts and outputs."""
    with open(filepath) as f:
        content = f.read()

    samples = []
    parts = content.split('--- PROMPT ')
    for part in parts[1:]:  # skip header
        lines = part.strip().split('\n')
        prompt_line = ''
        output_lines = []
        in_output = False
        for line in lines:
            if line.startswith('INPUT: '):
                prompt_line = line[7:]
            elif line.startswith('OUTPUT:'):
                in_output = True
            elif in_output:
                output_lines.append(line)
        output = '\n'.join(output_lines).strip()
        samples.append((prompt_line, output))
    return samples

def score_model(filepath):
    """Score all samples from a model."""
    samples = parse_eval_file(filepath)

    scores = {
        'bracket_balance': [],
        'repetition_3gram': [],
        'repetition_5gram': [],
        'rust_keywords': [],
        'token_quality': [],
        'syntax_coherence': [],
    }

    for prompt, output in samples:
        if not output or output == '[TIMEOUT/ERROR]':
            continue

        bracket_score, _ = count_brackets(output)
        scores['bracket_balance'].append(bracket_score)
        scores['repetition_3gram'].append(repetition_score(output, 3))
        scores['repetition_5gram'].append(repetition_score(output, 5))
        scores['rust_keywords'].append(rust_keyword_ratio(output))
        scores['token_quality'].append(invented_token_ratio(output))
        scores['syntax_coherence'].append(syntactic_coherence(output))

    # Average scores
    avg = {}
    for k, v in scores.items():
        avg[k] = sum(v) / len(v) if v else 0

    # Composite score (weighted)
    avg['composite'] = (
        avg['bracket_balance'] * 0.15 +
        avg['repetition_3gram'] * 0.15 +
        avg['repetition_5gram'] * 0.15 +
        avg['rust_keywords'] * 0.15 +
        avg['token_quality'] * 0.15 +
        avg['syntax_coherence'] * 0.25
    )

    return avg, len(samples)

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/coherence_20260228_0944'

    print("=" * 80)
    print("COHERENCE BENCHMARK RESULTS")
    print("=" * 80)
    print()

    all_scores = {}
    for f in sorted(Path(results_dir).glob('*.txt')):
        model_name = f.stem
        scores, n_samples = score_model(f)
        all_scores[model_name] = scores

        print(f"### {model_name} ({n_samples} samples)")
        print(f"  Bracket Balance:    {scores['bracket_balance']:.3f}")
        print(f"  Repetition (3-gram): {scores['repetition_3gram']:.3f}")
        print(f"  Repetition (5-gram): {scores['repetition_5gram']:.3f}")
        print(f"  Rust Keywords:      {scores['rust_keywords']:.3f}")
        print(f"  Token Quality:      {scores['token_quality']:.3f}")
        print(f"  Syntax Coherence:   {scores['syntax_coherence']:.3f}")
        print(f"  *** COMPOSITE:      {scores['composite']:.3f} ***")
        print()

    # Ranking
    if all_scores:
        print("=" * 80)
        print("RANKING (by composite score)")
        print("=" * 80)
        ranked = sorted(all_scores.items(), key=lambda x: x[1]['composite'], reverse=True)
        for i, (name, scores) in enumerate(ranked, 1):
            print(f"  {i}. {name:20s}  composite={scores['composite']:.3f}  "
                  f"syntax={scores['syntax_coherence']:.3f}  "
                  f"rep3={scores['repetition_3gram']:.3f}  "
                  f"brackets={scores['bracket_balance']:.3f}")

if __name__ == '__main__':
    main()
