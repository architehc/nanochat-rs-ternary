#!/usr/bin/env python3
"""
7-Day RTX 5090 Blackwell Training: Rust Code LLM

Strategy:
- QLoRA 4-bit NF4 on Qwen2.5-Coder-7B (saves 75% base model memory)
- LoRA rank 256 on all linear layers (high capacity)
- Sequence length 8192 (2x previous, deep context)
- Batch size 4 with grad_accum=8 (effective batch 32)
- BF16 compute with gradient checkpointing
- Cosine LR with warmup, 7-day schedule
- Hourly checkpoints + auto git push
- GPU temperature monitoring with thermal throttle protection
- Auto-resume from latest checkpoint

VRAM budget (QLoRA 4-bit, RTX 5090 32GB):
  Base model (NF4): ~4.0 GB
  LoRA adapters:     ~1.5 GB
  Optimizer states:  ~3.0 GB
  Activations (bs=4, seq=8192, grad_ckpt): ~18 GB
  CUDA overhead:     ~3.0 GB
  Total:            ~29.5 GB (fits in 32GB)
"""

import os
import sys
import json
import math
import time
import signal
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "rust_code_v2"  # Expanded dataset
DATA_DIR_FALLBACK = SCRIPT_DIR / "data" / "rust_code_fim"  # Original
OUTPUT_DIR = SCRIPT_DIR / "output_5090"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Training duration
TOTAL_DAYS = 7
TOTAL_HOURS = TOTAL_DAYS * 24  # 168 hours
SAVE_INTERVAL_HOURS = 1
GIT_PUSH_INTERVAL_HOURS = 2

# Hyperparameters optimized for 5090 + QLoRA
BATCH_SIZE = 2              # Per-device (QLoRA allows higher)
GRAD_ACCUM_STEPS = 8        # Effective batch = 2 * 8 = 16
MAX_SEQ_LEN = 8192          # Long context
LEARNING_RATE = 1e-4        # Lower LR for long training (7 days)
WEIGHT_DECAY = 0.05         # Slightly higher for regularization
WARMUP_RATIO = 0.01         # 1% warmup (short relative to 7 days)
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0

# QLoRA 4-bit config
USE_QLORA = True
QLORA_BITS = 4              # NF4 quantization
QLORA_COMPUTE_DTYPE = "bfloat16"
QLORA_DOUBLE_QUANT = True   # Double quantization saves ~0.4 GB

# LoRA config - high rank for quality
LORA_R = 256                # Higher rank = more capacity
LORA_ALPHA = 512            # 2x rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Blackwell optimizations
USE_BF16 = True
USE_TF32 = True
GRADIENT_CHECKPOINTING = True

# Thermal protection
MAX_GPU_TEMP = 85           # Celsius - pause if exceeded
TEMP_CHECK_INTERVAL = 60    # Check every 60 seconds

# Resume
RESUME_FROM_CHECKPOINT = True  # Auto-find latest checkpoint

DEVICE = "cuda:0"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"train_7day_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def get_gpu_info():
    """Get GPU memory and temperature."""
    info = {}
    if torch.cuda.is_available():
        info["mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        info["mem_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,fan.speed,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["temp_c"] = int(parts[0])
            info["power_w"] = float(parts[1])
            info["fan_pct"] = int(parts[2]) if parts[2].strip() != "[N/A]" else -1
            info["gpu_util_pct"] = int(parts[3])
    except:
        pass

    return info


def gpu_info_str():
    info = get_gpu_info()
    parts = []
    if "mem_allocated_gb" in info:
        parts.append(f"VRAM: {info['mem_allocated_gb']:.1f}/{info['mem_total_gb']:.0f}GB")
    if "temp_c" in info:
        parts.append(f"Temp: {info['temp_c']}°C")
    if "power_w" in info:
        parts.append(f"Power: {info['power_w']:.0f}W")
    if "gpu_util_pct" in info:
        parts.append(f"Util: {info['gpu_util_pct']}%")
    return " | ".join(parts)


def find_latest_checkpoint():
    """Find latest checkpoint to resume from."""
    if not CHECKPOINT_DIR.exists():
        return None

    checkpoints = []
    for d in CHECKPOINT_DIR.iterdir():
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            # Get modification time
            mtime = (d / "adapter_model.safetensors").stat().st_mtime
            checkpoints.append((mtime, d))

    # Also check HF trainer checkpoints in output dir
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            trainer_state = d / "trainer_state.json"
            if trainer_state.exists():
                mtime = trainer_state.stat().st_mtime
                checkpoints.append((mtime, d))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return str(checkpoints[0][1])


def git_push_progress(logger, step, loss, elapsed_hours):
    """Push training progress to 5090 branch."""
    try:
        repo_root = SCRIPT_DIR.parent.parent

        # Write progress file
        progress = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": loss,
            "elapsed_hours": round(elapsed_hours, 1),
            "remaining_hours": round(TOTAL_HOURS - elapsed_hours, 1),
            "gpu": get_gpu_info(),
        }
        progress_file = OUTPUT_DIR / "training_progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        # Git operations
        cmds = [
            ["git", "-C", str(repo_root), "add", str(progress_file)],
            ["git", "-C", str(repo_root), "add", str(LOG_DIR)],
            ["git", "-C", str(repo_root), "commit", "-m",
             f"5090 training: step {step}, loss {loss:.4f}, {elapsed_hours:.1f}h"],
            ["git", "-C", str(repo_root), "push", "origin", "5090"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        logger.info(f"Pushed progress to 5090 branch (step {step})")
    except Exception as e:
        logger.warning(f"Git push failed: {e}")


def create_model(logger):
    """Load model with QLoRA 4-bit and apply LoRA."""
    logger.info(f"Loading model: {MODEL_NAME}")

    if USE_QLORA:
        compute_dtype = getattr(torch, QLORA_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
        )
        logger.info("Using QLoRA 4-bit NF4 quantization")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        if GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    logger.info(f"Model loaded. {gpu_info_str()}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"LoRA r={LORA_R}: {trainable/1e6:.0f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")
    logger.info(f"After LoRA: {gpu_info_str()}")

    return model


class SevenDayTrainer(Trainer):
    """Custom trainer with 7-day monitoring, thermal protection, and auto git push."""

    def __init__(self, *args, start_time=None, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = start_time or time.time()
        self._logger = logger
        self._last_save_time = time.time()
        self._last_push_time = time.time()
        self._last_temp_check = time.time()
        self._save_interval = SAVE_INTERVAL_HOURS * 3600
        self._push_interval = GIT_PUSH_INTERVAL_HOURS * 3600
        self._best_loss = float('inf')
        self._step_times = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        step_start = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch)
        step_time = time.time() - step_start
        self._step_times.append(step_time)

        now = time.time()
        elapsed = now - self.start_time

        # Thermal protection
        if now - self._last_temp_check >= TEMP_CHECK_INTERVAL:
            self._last_temp_check = now
            info = get_gpu_info()
            temp = info.get("temp_c", 0)
            if temp > MAX_GPU_TEMP:
                if self._logger:
                    self._logger.warning(f"GPU at {temp}°C > {MAX_GPU_TEMP}°C! Pausing 60s...")
                time.sleep(60)

        # Hourly checkpoint
        if now - self._last_save_time >= self._save_interval:
            elapsed_hours = elapsed / 3600
            if self._logger:
                self._logger.info(f"Hourly checkpoint at {elapsed_hours:.1f}h")
            save_path = CHECKPOINT_DIR / f"checkpoint-hour-{int(elapsed_hours)}"
            self.save_model(str(save_path))
            self._last_save_time = now

            # Track best
            loss_val = loss.item()
            if loss_val < self._best_loss:
                self._best_loss = loss_val
                best_path = CHECKPOINT_DIR / "checkpoint-best"
                self.save_model(str(best_path))
                if self._logger:
                    self._logger.info(f"New best loss: {loss_val:.4f}")

        # Git push
        if now - self._last_push_time >= self._push_interval:
            elapsed_hours = elapsed / 3600
            git_push_progress(self._logger, self.state.global_step, loss.item(), elapsed_hours)
            self._last_push_time = now

        return loss

    def log(self, logs, start_time=None):
        super().log(logs, start_time)
        step = self.state.global_step
        if "loss" in logs and self._logger and step % 25 == 0:
            elapsed = (time.time() - self.start_time) / 3600
            remaining = TOTAL_HOURS - elapsed

            # Calculate throughput
            avg_step_time = sum(self._step_times[-100:]) / max(1, len(self._step_times[-100:]))
            tokens_per_sec = (BATCH_SIZE * MAX_SEQ_LEN) / avg_step_time if avg_step_time > 0 else 0

            self._logger.info(
                f"[{elapsed:.1f}h/{TOTAL_HOURS}h] "
                f"step={step} loss={logs.get('loss', 0):.4f} "
                f"lr={logs.get('learning_rate', 0):.2e} "
                f"tok/s={tokens_per_sec:.0f} "
                f"step_t={avg_step_time:.2f}s | "
                f"{gpu_info_str()}"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("7-DAY RTX 5090 BLACKWELL TRAINING: RUST CODE LLM")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"QLoRA: {USE_QLORA} ({QLORA_BITS}-bit NF4)")
    logger.info(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    logger.info(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"Seq length: {MAX_SEQ_LEN}")
    logger.info(f"Duration: {TOTAL_DAYS} days ({TOTAL_HOURS} hours)")
    logger.info(f"GPU: {gpu_info_str()}")

    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._inductor.config.triton.cudagraphs = False
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    logger.info("TF32 + BF16 enabled for Blackwell")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = create_model(logger)

    # Load dataset
    data_dir = DATA_DIR if DATA_DIR.exists() else DATA_DIR_FALLBACK
    logger.info(f"Loading dataset from: {data_dir}")
    dataset = load_from_disk(str(data_dir))
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    logger.info(f"Train: {train_size:,} sequences, Val: {val_size:,} sequences")

    # Calculate steps for 7 days
    # Estimate: ~2 sec/step for QLoRA 7B with bs=2, seq=8192
    estimated_sps = 0.5  # conservative
    max_steps = int(TOTAL_HOURS * 3600 * estimated_sps)
    steps_per_epoch = train_size // (BATCH_SIZE * GRAD_ACCUM_STEPS)

    # Cap at reasonable number
    max_steps = min(max_steps, 500_000)

    logger.info(f"Max steps: {max_steps:,}")
    logger.info(f"Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"Expected epochs: {max_steps / max(1, steps_per_epoch):.1f}")

    # Check for resume
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = find_latest_checkpoint()
        if resume_checkpoint:
            logger.info(f"Resuming from: {resume_checkpoint}")

    # Training arguments
    save_steps = max(500, steps_per_epoch // 4)  # ~4 saves per epoch

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=max_steps,

        # Batch
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        # Optimizer
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,

        # Precision
        bf16=USE_BF16,
        tf32=USE_TF32,

        # Memory
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=False,

        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=10,

        # Evaluation
        eval_strategy="steps",
        eval_steps=min(2000, save_steps),

        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",

        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,

        # Misc
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create trainer
    start_time = time.time()
    trainer = SevenDayTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        start_time=start_time,
        logger=logger,
    )

    # Signal handlers
    def shutdown(sig, frame):
        logger.info(f"Received signal {sig}. Saving emergency checkpoint...")
        elapsed = (time.time() - start_time) / 3600
        save_path = CHECKPOINT_DIR / f"checkpoint-shutdown-{int(elapsed)}h"
        trainer.save_model(str(save_path))
        git_push_progress(logger, trainer.state.global_step, 0, elapsed)
        logger.info(f"Saved to {save_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Initial eval
    logger.info("Running initial evaluation...")
    try:
        eval_results = trainer.evaluate()
        logger.info(f"Initial eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    except Exception as e:
        logger.warning(f"Initial eval failed: {e}")

    # Train!
    logger.info("=" * 70)
    logger.info("STARTING 7-DAY TRAINING")
    logger.info(f"Target: {TOTAL_HOURS}h, {max_steps:,} steps")
    logger.info("=" * 70)

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

        # Save final
        final_path = OUTPUT_DIR / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        metrics = train_result.metrics
        elapsed = (time.time() - start_time) / 3600
        metrics["total_hours"] = elapsed
        metrics["model"] = MODEL_NAME
        metrics["lora_r"] = LORA_R
        metrics["qlora"] = USE_QLORA

        with open(OUTPUT_DIR / "final_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Duration: {elapsed:.1f}h")
        logger.info(f"Final loss: {metrics.get('train_loss', 'N/A')}")
        logger.info(f"Model: {final_path}")
        logger.info("=" * 70)

        # Final push
        git_push_progress(logger, trainer.state.global_step,
                         metrics.get("train_loss", 0), elapsed)

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        try:
            save_path = CHECKPOINT_DIR / "checkpoint-emergency"
            trainer.save_model(str(save_path))
            logger.info(f"Emergency save: {save_path}")
            git_push_progress(logger, trainer.state.global_step, 0,
                            (time.time() - start_time) / 3600)
        except:
            pass
        raise


if __name__ == "__main__":
    main()
