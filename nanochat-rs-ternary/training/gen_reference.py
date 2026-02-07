"""
gen_reference.py — Generate reference data for Python-Rust cross-validation.

Creates a tiny model (or loads a checkpoint), runs a fixed input through it,
and exports:
  - ref_logits.npy: output logits for the last token
  - ref.gguf: model weights in GGUF format
  - ref.mhc: mHC parameters in binary format

Usage:
  python gen_reference.py                         # tiny random model
  python gen_reference.py --checkpoint path.pt    # from training checkpoint
"""

import os
import sys
import argparse

import torch
import numpy as np

# Ensure training/ is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import NanochatConfig, NanochatTernary
from export import export_gguf
from mhc_lite import export_mhc_weights


def main():
    parser = argparse.ArgumentParser(
        description='Generate reference data for Python-Rust cross-validation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to training checkpoint (.pt). '
                             'If not provided, creates a tiny random model.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    ref_dir = os.path.dirname(os.path.abspath(__file__))

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu',
                                weights_only=False)
        if 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
        else:
            config = NanochatConfig.d20()
        model = NanochatTernary(config)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Create a tiny model for cross-validation testing.
        # dim=128, group_size=128 ensures clean quantization groups.
        print("Creating tiny random model for cross-validation...")
        torch.manual_seed(args.seed)
        config = NanochatConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            ffn_mult=2.667,
            vocab_size=256,
            max_seq_len=64,
            group_size=128,
            mhc_n_streams=2,
        )
        model = NanochatTernary(config)

    model.eval()

    # Fixed reference input sequence
    input_ids = torch.tensor([[1, 5, 10, 20, 42]], dtype=torch.long)

    print(f"\nModel config: dim={config.dim}, layers={config.n_layers}, "
          f"heads={config.n_heads}, vocab={config.vocab_size}")
    print(f"Input tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        # Take the last token's logits
        ref_logits = logits[0, -1, :].numpy()

    # Save reference logits
    npy_path = os.path.join(ref_dir, 'ref_logits.npy')
    np.save(npy_path, ref_logits)
    print(f"\nSaved ref_logits.npy: shape={ref_logits.shape}, "
          f"min={ref_logits.min():.6f}, max={ref_logits.max():.6f}, "
          f"mean={ref_logits.mean():.6f}")

    # Export model to GGUF + mHC
    gguf_path = os.path.join(ref_dir, 'ref.gguf')
    mhc_path = os.path.join(ref_dir, 'ref.mhc')

    print("\nExporting GGUF...")
    export_gguf(model, gguf_path)

    print("\nExporting mHC...")
    export_mhc_weights(model, mhc_path)

    print(f"\nCross-validation reference data generated:")
    print(f"  Logits:  {npy_path}")
    print(f"  GGUF:    {gguf_path}")
    print(f"  mHC:     {mhc_path}")
    print(f"\nTo run Rust cross-validation:")
    print(f"  cargo test cross_validate -- --ignored")


if __name__ == '__main__':
    main()
