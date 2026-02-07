"""
export.py — Export trained PyTorch checkpoint to GGUF + mHC binary for Rust inference.

Exports:
  1. Ternary weights + scales -> GGUF file (custom Q1_58 type)
  2. mHC parameters -> binary file (matching Rust io.rs format)
  3. Embeddings (FP16) + norms (FP32) -> included in GGUF
"""

import struct
import argparse
import numpy as np
import torch
import torch.nn as nn

from model import NanochatConfig, NanochatTernary
from ternary_qat import absmean_quantize, pack_ternary_2bit
from mhc_lite import MhcLiteN2, MhcLiteN4, export_mhc_weights


# =============================================================================
# GGUF Constants and Helpers
# =============================================================================

GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3
GGUF_TYPE_Q1_58 = 100  # Custom ternary type

# GGUF metadata value types
GGUF_META_UINT32 = 4
GGUF_META_STRING = 8

# GGUF tensor types
GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1


def write_gguf_string(f, s: str):
    """Write a GGUF string (length-prefixed, no null terminator)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_metadata_kv(f, key: str, val_type: int, value):
    """Write a GGUF metadata key-value pair."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', val_type))
    if val_type == GGUF_META_UINT32:
        f.write(struct.pack('<I', value))
    elif val_type == GGUF_META_STRING:
        write_gguf_string(f, value)


# =============================================================================
# Export Functions
# =============================================================================

def export_gguf(model: NanochatTernary, path: str):
    """Export model weights to GGUF format.

    Layout:
      - Token embeddings: FP16
      - BitLinearSTE weights: Q1_58 (2-bit packed + per-group f32 scales)
      - RMSNorm weights: F32
    """
    config = model.config
    tensors = []  # list of (name, data_bytes, shape, dtype_id)

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            param = module.weight.data

            if 'tok_embed' in name:
                # Embeddings: FP16
                data = param.half().cpu().numpy().tobytes()
                tensors.append((name + '.weight', data, list(param.shape), GGUF_TENSOR_F16))

            elif hasattr(module, 'get_ternary_weights'):
                # BitLinearSTE: quantize to ternary, pack
                w_ternary, scales = module.get_ternary_weights()
                packed = pack_ternary_2bit(w_ternary)
                # Combine packed data + scales into single blob
                packed_bytes = packed.cpu().numpy().tobytes()
                scales_bytes = scales.cpu().numpy().astype(np.float32).tobytes()
                data = packed_bytes + scales_bytes
                tensors.append((name + '.weight', data, list(param.shape), GGUF_TYPE_Q1_58))

            elif 'norm' in name:
                # Norms: F32
                data = param.float().cpu().numpy().tobytes()
                tensors.append((name + '.weight', data, list(param.shape), GGUF_TENSOR_F32))

    # Build metadata
    metadata = {
        'general.architecture': ('nanochat-ternary', GGUF_META_STRING),
        'general.name': (f'nanochat-{config.dim}', GGUF_META_STRING),
        'nanochat.dim': (config.dim, GGUF_META_UINT32),
        'nanochat.n_layers': (config.n_layers, GGUF_META_UINT32),
        'nanochat.n_heads': (config.n_heads, GGUF_META_UINT32),
        'nanochat.vocab_size': (config.vocab_size, GGUF_META_UINT32),
        'nanochat.group_size': (config.group_size, GGUF_META_UINT32),
        'nanochat.mhc_n_streams': (config.mhc_n_streams, GGUF_META_UINT32),
    }

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors)))    # n_tensors
        f.write(struct.pack('<Q', len(metadata)))   # n_kv

        # Metadata
        for key, (value, vtype) in metadata.items():
            write_gguf_metadata_kv(f, key, vtype, value)

        # Tensor info (headers)
        data_offset = 0
        tensor_offsets = []
        for name, data, shape, dtype_id in tensors:
            write_gguf_string(f, name)
            f.write(struct.pack('<I', len(shape)))       # n_dims
            for dim in shape:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', dtype_id))
            f.write(struct.pack('<Q', data_offset))      # offset into data section
            tensor_offsets.append(data_offset)
            data_offset += len(data)

        # Alignment padding to 32 bytes
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b'\x00' * pad)

        # Tensor data
        for name, data, shape, dtype_id in tensors:
            f.write(data)

    total_size = 0
    for _, data, _, _ in tensors:
        total_size += len(data)

    print(f"Exported GGUF: {path}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Total data: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")


def export_checkpoint(checkpoint_path: str, gguf_path: str, mhc_path: str,
                      config_name: str = 'd20'):
    """Full export pipeline: checkpoint -> GGUF + mHC binary."""
    # Load checkpoint first to get config (may have overridden vocab_size)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in checkpoint and checkpoint['config'] is not None:
        config = checkpoint['config']
        print(f"Using config from checkpoint (vocab_size={config.vocab_size})")
    else:
        # Fallback to preset
        if config_name == 'd20':
            config = NanochatConfig.d20()
        elif config_name == '125m':
            config = NanochatConfig.nano_125m()
        elif config_name == '560m':
            config = NanochatConfig.nano_560m()
        else:
            raise ValueError(f"Unknown config: {config_name}")

    # Load model
    model = NanochatTernary(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Export GGUF
    export_gguf(model, gguf_path)

    # Export mHC
    export_mhc_weights(model, mhc_path)

    print(f"\nExport complete:")
    print(f"  GGUF: {gguf_path}")
    print(f"  mHC:  {mhc_path}")


# =============================================================================
# Self-test
# =============================================================================

def _self_test():
    print("=" * 60)
    print("Export Self-Test")
    print("=" * 60)

    import tempfile
    import os

    config = NanochatConfig.d20()
    model = NanochatTernary(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, 'test.gguf')
        mhc_path = os.path.join(tmpdir, 'test.mhc')

        # Export GGUF
        print("\n[1] Testing GGUF export...")
        export_gguf(model, gguf_path)
        gguf_size = os.path.getsize(gguf_path)
        print(f"  File size: {gguf_size:,} bytes")
        assert gguf_size > 0

        # Verify GGUF header
        with open(gguf_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            assert magic == GGUF_MAGIC, f"Bad magic: {hex(magic)}"
            assert version == GGUF_VERSION, f"Bad version: {version}"
        print(f"  GGUF header valid (magic={hex(GGUF_MAGIC)}, version={GGUF_VERSION})")

        # Export mHC
        print("\n[2] Testing mHC export...")
        export_mhc_weights(model, mhc_path)
        mhc_size = os.path.getsize(mhc_path)
        expected_mhc_size = 16 + config.n_layers * 2 * 36  # header + 2 mHC per layer * 36 bytes
        print(f"  File size: {mhc_size} bytes (expected {expected_mhc_size})")
        assert mhc_size == expected_mhc_size

    print("\n" + "=" * 60)
    print("ALL EXPORT TESTS PASSED")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export trained model to GGUF + mHC')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to training checkpoint (.pt)')
    parser.add_argument('--gguf', type=str, required=True,
                        help='Output GGUF path')
    parser.add_argument('--mhc', type=str, required=True,
                        help='Output mHC binary path')
    parser.add_argument('--config', type=str, default='d20',
                        choices=['d20', '125m', '560m'])

    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.gguf, args.mhc, args.config)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        _self_test()
