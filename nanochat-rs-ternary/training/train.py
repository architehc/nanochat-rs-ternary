"""
train.py — Training loop for nanochat-rs Ternary + mHC-lite

Optimizer strategy:
  - Linear weight matrices (2D+) -> AdamW (Muon requires custom impl, fallback to AdamW)
  - mHC params + norms + biases -> AdamW with lower LR
  - Embeddings -> AdamW with no weight decay

Schedule: Warmup-Stable-Decay (WSD)
  - Warmup: linear from 0 -> lr over warmup_steps
  - Stable: constant lr for most of training
  - Decay: cosine anneal to 0.1 * lr

Data: synthetic random tokens for initial validation, then real data.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tiktoken

from model import NanochatConfig, NanochatTernary
from mhc_lite import measure_composite_gain


# =============================================================================
# Synthetic Dataset (for initial training validation)
# =============================================================================

class SyntheticTextDataset(Dataset):
    """Generates random token sequences for training loop validation.

    Provides simple patterns that a model can learn to verify gradient flow:
      - Repeated tokens: [A, A, A, ...] -> predict A
      - Sequential: [1, 2, 3, ...] -> predict next
      - Copy: [A, B, C, _, A, B, C] -> predict copy of prefix
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000,
                 seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        torch.manual_seed(seed)
        # Mix of patterns
        self.data = []
        for i in range(num_samples):
            pattern = i % 3
            if pattern == 0:
                # Repeated token
                tok = torch.randint(1, vocab_size, (1,)).item()
                seq = torch.full((seq_len,), tok, dtype=torch.long)
            elif pattern == 1:
                # Sequential (modular)
                start = torch.randint(0, vocab_size, (1,)).item()
                seq = torch.arange(start, start + seq_len, dtype=torch.long) % vocab_size
            else:
                # Random (harder)
                seq = torch.randint(0, vocab_size, (seq_len,))
            self.data.append(seq)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Input: seq[:-1], Target: seq[1:]
        return seq[:-1], seq[1:]


# =============================================================================
# TinyStories Dataset (real data for actual training)
# =============================================================================

class TinyStoriesDataset(Dataset):
    """Loads TinyStories from HuggingFace, tokenizes with tiktoken GPT-2.

    Each sample is a contiguous chunk of seq_len+1 tokens from the corpus,
    split into input[:-1] and target[1:] for next-token prediction.

    The full corpus is tokenized once and concatenated into a single flat
    token buffer, then sliced into non-overlapping chunks.
    """

    def __init__(self, seq_len: int, split: str = 'train',
                 max_stories: int = 0, cache_dir: str = None):
        from datasets import load_dataset

        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding('gpt2')
        self.vocab_size = self.enc.n_vocab  # 50257

        # Load dataset
        print(f"Loading TinyStories ({split} split)...")
        ds = load_dataset('roneneldan/TinyStories', split=split,
                          cache_dir=cache_dir, trust_remote_code=False)

        # Optionally limit number of stories for faster iteration
        if max_stories > 0:
            ds = ds.select(range(min(max_stories, len(ds))))
        print(f"  Stories: {len(ds):,}")

        # Tokenize all stories and concatenate into flat buffer
        print("Tokenizing...")
        all_tokens = []
        eot = self.enc.eot_token  # end-of-text separator
        for i, example in enumerate(ds):
            text = example['text']
            if text.strip():
                tokens = self.enc.encode_ordinary(text)
                all_tokens.extend(tokens)
                all_tokens.append(eot)
            if (i + 1) % 200000 == 0:
                print(f"  Tokenized {i+1:,} stories ({len(all_tokens):,} tokens)...")

        print(f"  Total tokens: {len(all_tokens):,}")

        # Slice into non-overlapping chunks of seq_len+1
        chunk_size = seq_len + 1  # +1 for target shift
        n_chunks = len(all_tokens) // chunk_size
        # Trim to exact multiple
        all_tokens = all_tokens[:n_chunks * chunk_size]
        self.data = torch.tensor(all_tokens, dtype=torch.long).reshape(n_chunks, chunk_size)
        print(f"  Chunks: {n_chunks:,} (seq_len={seq_len})")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# =============================================================================
# Learning Rate Schedule: WSD (Warmup-Stable-Decay)
# =============================================================================

def wsd_schedule(step: int, warmup_steps: int, total_steps: int,
                 decay_start_frac: float = 0.8, min_lr_frac: float = 0.1) -> float:
    """Warmup-Stable-Decay learning rate multiplier.

    Returns a multiplier in [min_lr_frac, 1.0].
    """
    if step < warmup_steps:
        # Linear warmup
        return step / max(1, warmup_steps)

    decay_start = int(total_steps * decay_start_frac)

    if step < decay_start:
        # Stable phase
        return 1.0

    # Cosine decay
    decay_steps = total_steps - decay_start
    progress = (step - decay_start) / max(1, decay_steps)
    return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Parameter Groups
# =============================================================================

def build_param_groups(model: NanochatTernary, lr: float, mhc_lr: float,
                       weight_decay: float) -> list:
    """Split parameters into optimizer groups.

    Groups:
      1. Linear weights (BitLinearSTE.weight, 2D) -> lr, weight_decay
      2. mHC params (alpha_logit, pre/post logits/bias) -> mhc_lr, no weight_decay
      3. Norms (RMSNorm.weight, 1D) -> mhc_lr, no weight_decay
      4. Embeddings -> mhc_lr, no weight_decay
    """
    linear_params = []
    mhc_params = []
    norm_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'tok_embed' in name or 'embed' in name:
            embed_params.append(param)
        elif 'mhc' in name:
            mhc_params.append(param)
        elif 'norm' in name:
            norm_params.append(param)
        elif param.dim() >= 2:
            linear_params.append(param)
        else:
            # Fallback
            mhc_params.append(param)

    groups = []
    if linear_params:
        groups.append({
            'params': linear_params,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'linear',
        })
    if mhc_params:
        groups.append({
            'params': mhc_params,
            'lr': mhc_lr,
            'weight_decay': 0.0,
            'name': 'mhc',
        })
    if norm_params:
        groups.append({
            'params': norm_params,
            'lr': mhc_lr,
            'weight_decay': 0.0,
            'name': 'norm',
        })
    if embed_params:
        groups.append({
            'params': embed_params,
            'lr': mhc_lr,
            'weight_decay': 0.0,
            'name': 'embed',
        })

    return groups


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Config
    if args.config == 'd20':
        config = NanochatConfig.d20()
    elif args.config == '125m':
        config = NanochatConfig.nano_125m()
    elif args.config == '560m':
        config = NanochatConfig.nano_560m()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Dataset (load first so we can override vocab_size if needed)
    seq_len = min(config.max_seq_len, args.seq_len)

    if args.dataset == 'tinystories':
        dataset = TinyStoriesDataset(
            seq_len=seq_len,
            split='train',
            max_stories=args.max_stories,
        )
        # Override vocab_size to match tiktoken GPT-2
        config.vocab_size = dataset.vocab_size
        print(f"Using TinyStories dataset (vocab_size overridden to {config.vocab_size})")
    else:
        dataset = SyntheticTextDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len + 1,  # +1 for target shift
            num_samples=args.num_samples,
        )
        print(f"Using synthetic dataset")

    print(f"Config: {args.config} (dim={config.dim}, layers={config.n_layers}, "
          f"heads={config.n_heads}, vocab={config.vocab_size}, streams={config.mhc_n_streams})")

    # Model
    model = NanochatTernary(config).to(device)
    counts = model.count_params()
    print(f"Parameters: {counts['total']:,} total "
          f"(linear={counts['linear']:,}, mhc={counts['mhc']:,}, "
          f"embed={counts['embed']:,}, norm={counts['norm']:,})")

    # GPU memory report
    if device.type == 'cuda':
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU memory after model load: {mem_mb:.0f} MB")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    # Optimizer
    param_groups = build_param_groups(
        model, lr=args.lr, mhc_lr=args.mhc_lr, weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = args.epochs * (len(dataloader) // args.grad_accum_steps)

    print(f"\nTraining: {args.epochs} epochs, {len(dataloader)} micro-steps/epoch, "
          f"{total_steps} optimizer steps")
    print(f"LR: linear={args.lr}, mhc={args.mhc_lr}, wd={args.weight_decay}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum_steps} accum = "
          f"{args.batch_size * args.grad_accum_steps} effective, Seq len: {seq_len}")
    print(f"Grad clip: {args.grad_clip}")
    print()

    # Training
    global_step = 0
    micro_step = 0
    best_loss = float('inf')
    loss_history = []
    accum_steps = args.grad_accum_steps

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # LR schedule (update on optimizer steps, not micro steps)
            if micro_step % accum_steps == 0:
                lr_mult = wsd_schedule(global_step, args.warmup_steps, total_steps)
                for group in optimizer.param_groups:
                    if '_base_lr' not in group:
                        group['_base_lr'] = group['lr']
                    group['lr'] = group['_base_lr'] * lr_mult

            # Forward
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                target_ids.reshape(-1),
            )
            # Scale loss for gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            # Logging
            batch_loss = loss.item()
            batch_tokens = target_ids.numel()
            epoch_loss += batch_loss * batch_tokens
            epoch_tokens += batch_tokens
            micro_step += 1

            # Optimizer step every accum_steps
            if micro_step % accum_steps == 0:
                # Gradient clipping
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = sum(
                        p.grad.norm().item() ** 2 for p in model.parameters()
                        if p.grad is not None
                    ) ** 0.5

                optimizer.step()
                optimizer.zero_grad()

                # GPU memory report after first optimizer step
                if global_step == 0 and device.type == 'cuda':
                    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    curr_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"  GPU memory after first batch: {curr_mb:.0f} MB current, "
                          f"{peak_mb:.0f} MB peak")

                loss_history.append(batch_loss)

                if global_step % args.log_interval == 0:
                    elapsed = time.time() - t0
                    tokens_per_sec = epoch_tokens / max(elapsed, 1e-6)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  step {global_step:5d} | loss {batch_loss:.4f} | "
                          f"grad_norm {grad_norm:.4f} | lr {current_lr:.2e} | "
                          f"{tokens_per_sec:.0f} tok/s")

                # mHC diagnostics
                if global_step > 0 and global_step % args.diag_interval == 0:
                    diag = measure_composite_gain(model)
                    gain = diag.get('composite_gain', 0)
                    ds_viol = diag.get('ds_violations', 'none')
                    print(f"  [mHC] composite_gain={gain:.4f} | ds_violations={ds_viol}")
                    if gain > 2.0:
                        print(f"  WARNING: composite gain {gain:.2f} > 2.0!")

                global_step += 1

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} | "
              f"time={elapsed:.1f}s | {epoch_tokens/elapsed:.0f} tok/s")

        # Check loss is decreasing
        if avg_loss < best_loss:
            best_loss = avg_loss
            if args.save_path:
                save_checkpoint(model, optimizer, epoch, global_step, args.save_path)
                print(f"  Saved checkpoint to {args.save_path}")

        # Loss sanity check
        if len(loss_history) > 100:
            recent = sum(loss_history[-50:]) / 50
            early = sum(loss_history[:50]) / 50
            if recent < early:
                print(f"  Loss decreasing: {early:.4f} -> {recent:.4f}")
            else:
                print(f"  WARNING: Loss not decreasing: {early:.4f} -> {recent:.4f}")

    # Save final checkpoint
    if args.save_path:
        save_checkpoint(model, optimizer, args.epochs - 1, global_step, args.save_path)
        print(f"Saved final checkpoint to {args.save_path}")

    # Final diagnostics
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Best loss:  {best_loss:.4f}")
    if len(loss_history) > 10:
        print(f"First 10 avg: {sum(loss_history[:10])/10:.4f}")
        print(f"Last 10 avg:  {sum(loss_history[-10:])/10:.4f}")

    diag = measure_composite_gain(model)
    print(f"\nmHC final diagnostics:")
    print(f"  Layers:         {diag['n_layers']}")
    print(f"  Composite gain: {diag['composite_gain']:.6f}")
    print(f"  DS violations:  {diag['ds_violations']}")

    return model


def save_checkpoint(model, optimizer, epoch, step, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': model.config,
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train nanochat ternary model')
    parser.add_argument('--config', type=str, default='d20',
                        choices=['d20', '125m', '560m'],
                        help='Model config preset')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'tinystories'],
                        help='Dataset: synthetic (random patterns) or tinystories (real data)')
    parser.add_argument('--max_stories', type=int, default=0,
                        help='Limit TinyStories to N stories (0 = all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for linear weights')
    parser.add_argument('--mhc_lr', type=float, default=1e-3,
                        help='Learning rate for mHC/norm/embed params')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--diag_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save checkpoint (empty = no save)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
