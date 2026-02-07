"""
evaluate.py — Benchmark and evaluate a trained nanochat-ternary model.

Runs:
  1. Text generation from various prompts (greedy + sampled)
  2. Validation perplexity on TinyStories validation split
  3. Inference speed measurement (tokens/sec)
"""

import argparse
import time
import math
import torch
import torch.nn.functional as F
import tiktoken

from model import NanochatConfig, NanochatTernary
from mhc_lite import measure_composite_gain


# =============================================================================
# Text Generation
# =============================================================================

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_tokens: int = 200,
                  temperature: float = 0.0, top_k: int = 0,
                  device: str = 'cuda') -> str:
    """Autoregressive text generation."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Truncate to max_seq_len if needed
        if input_ids.shape[1] >= model.config.max_seq_len:
            break

        logits = model(input_ids)  # [1, T, vocab]
        next_logits = logits[0, -1, :]  # [vocab]

        if temperature <= 0:
            # Greedy
            next_token = next_logits.argmax().unsqueeze(0)
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop on EOT
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(input_ids[0].tolist())


# =============================================================================
# Validation Perplexity
# =============================================================================

@torch.no_grad()
def measure_perplexity(model, config, device: str = 'cuda',
                       n_batches: int = 500, batch_size: int = 8,
                       seq_len: int = 256) -> dict:
    """Measure perplexity on TinyStories validation set."""
    from datasets import load_dataset
    enc = tiktoken.get_encoding('gpt2')

    print("Loading TinyStories validation split...")
    ds = load_dataset('roneneldan/TinyStories', split='validation')
    print(f"  Validation stories: {len(ds):,}")

    # Tokenize
    print("Tokenizing validation data...")
    all_tokens = []
    for i, story in enumerate(ds):
        tokens = enc.encode(story['text'], allowed_special=set())
        all_tokens.extend(tokens)
        all_tokens.append(enc.eot_token)
        if len(all_tokens) > n_batches * batch_size * (seq_len + 1) * 2:
            break  # enough tokens

    print(f"  Validation tokens: {len(all_tokens):,}")

    # Create chunks
    chunk_size = seq_len + 1
    n_chunks = len(all_tokens) // chunk_size
    tokens_tensor = torch.tensor(all_tokens[:n_chunks * chunk_size], dtype=torch.long)
    chunks = tokens_tensor.view(n_chunks, chunk_size)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_actual = min(n_batches, n_chunks // batch_size)

    print(f"Evaluating {n_actual} batches...")
    for i in range(n_actual):
        batch = chunks[i * batch_size:(i + 1) * batch_size].to(device)
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            target_ids.reshape(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += target_ids.numel()

        if (i + 1) % 100 == 0:
            avg = total_loss / total_tokens
            print(f"  batch {i+1}/{n_actual} | avg_loss={avg:.4f} | ppl={math.exp(avg):.2f}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'n_batches': n_actual,
        'n_tokens': total_tokens,
    }


# =============================================================================
# Inference Speed
# =============================================================================

@torch.no_grad()
def measure_inference_speed(model, tokenizer, device: str = 'cuda',
                            n_tokens: int = 200) -> dict:
    """Measure autoregressive generation speed."""
    model.eval()
    prompt = "Once upon a time there was a"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Warmup
    _ = model(input_ids)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed generation
    start = time.perf_counter()
    for _ in range(n_tokens):
        if input_ids.shape[1] >= model.config.max_seq_len:
            break
        logits = model(input_ids)
        next_token = logits[0, -1, :].argmax().unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = input_ids.shape[1] - len(tokens)
    tok_per_sec = generated / elapsed if elapsed > 0 else 0

    return {
        'generated_tokens': generated,
        'elapsed_sec': elapsed,
        'tok_per_sec': tok_per_sec,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained nanochat model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_batches', type=int, default=500)
    parser.add_argument('--gen_tokens', type=int, default=200)
    args = parser.parse_args()

    device = args.device
    enc = tiktoken.get_encoding('gpt2')

    # Load model
    print("=" * 70)
    print("Nanochat-Ternary Model Evaluation")
    print("=" * 70)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    config = checkpoint.get('config', NanochatConfig.nano_125m())
    print(f"Config: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}, "
          f"vocab={config.vocab_size}, streams={config.mhc_n_streams}")

    model = NanochatTernary(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    counts = model.count_params()
    print(f"Parameters: {counts['total']:,} total")

    # =========================================================================
    # 1. Text Generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEXT GENERATION SAMPLES")
    print("=" * 70)

    prompts = [
        "Once upon a time",
        "The little cat",
        "Tom was a good boy who",
        "One day, a girl named Lucy",
        "The dog and the bird were",
        "There was a big red",
        "Mom said to her son",
        "The sun was shining and",
    ]

    # Greedy samples
    print("\n--- Greedy (temperature=0) ---\n")
    for prompt in prompts[:4]:
        text = generate_text(model, enc, prompt, max_tokens=args.gen_tokens,
                             temperature=0.0, device=device)
        print(f"Prompt: \"{prompt}\"")
        print(f"Output: {text}")
        print()

    # Sampled
    print("\n--- Sampled (temperature=0.8, top_k=50) ---\n")
    for prompt in prompts:
        text = generate_text(model, enc, prompt, max_tokens=args.gen_tokens,
                             temperature=0.8, top_k=50, device=device)
        print(f"Prompt: \"{prompt}\"")
        print(f"Output: {text}")
        print()

    # =========================================================================
    # 2. Validation Perplexity
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION PERPLEXITY")
    print("=" * 70)

    ppl_results = measure_perplexity(model, config, device=device,
                                     n_batches=args.val_batches)
    print(f"\n  Average loss:  {ppl_results['avg_loss']:.4f}")
    print(f"  Perplexity:    {ppl_results['perplexity']:.2f}")
    print(f"  Eval tokens:   {ppl_results['n_tokens']:,}")

    # =========================================================================
    # 3. Inference Speed
    # =========================================================================
    print("\n" + "=" * 70)
    print("INFERENCE SPEED")
    print("=" * 70)

    speed = measure_inference_speed(model, enc, device=device,
                                    n_tokens=args.gen_tokens)
    print(f"\n  Generated:     {speed['generated_tokens']} tokens")
    print(f"  Time:          {speed['elapsed_sec']:.2f}s")
    print(f"  Speed:         {speed['tok_per_sec']:.1f} tok/s")

    # =========================================================================
    # 4. mHC Diagnostics
    # =========================================================================
    print("\n" + "=" * 70)
    print("mHC DIAGNOSTICS")
    print("=" * 70)

    diag = measure_composite_gain(model)
    print(f"\n  Layers:         {diag['n_layers']}")
    print(f"  Composite gain: {diag['composite_gain']:.6f}")
    print(f"  DS violations:  {diag['ds_violations']}")

    # =========================================================================
    # 5. Model Size Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL SIZE ANALYSIS")
    print("=" * 70)

    fp32_size = counts['total'] * 4  # 4 bytes per float32
    # Ternary weights: 2 bits per weight + scales overhead
    ternary_bits = counts['linear'] * 2
    scales_overhead = (counts['linear'] / config.group_size) * 32  # f32 scales
    embed_size = counts['embed'] * 2  # FP16
    norm_size = counts['norm'] * 4  # FP32
    mhc_size = counts['mhc'] * 4  # FP32

    gguf_est = (ternary_bits / 8) + (scales_overhead / 8) + embed_size + norm_size + mhc_size

    print(f"\n  FP32 model size:    {fp32_size / 1e6:.1f} MB")
    print(f"  GGUF ternary size:  {gguf_est / 1e6:.1f} MB (estimated)")
    print(f"  Compression ratio:  {fp32_size / gguf_est:.1f}x")
    print(f"  Bits per weight:    {(gguf_est * 8) / counts['total']:.2f} (avg)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Validation Perplexity: {ppl_results['perplexity']:.2f}")
    print(f"  Generation Speed:      {speed['tok_per_sec']:.1f} tok/s")
    print(f"  mHC Composite Gain:    {diag['composite_gain']:.6f}")
    print(f"  GGUF Size:             ~{gguf_est / 1e6:.0f} MB ({fp32_size / gguf_est:.1f}x compression)")


if __name__ == '__main__':
    main()
