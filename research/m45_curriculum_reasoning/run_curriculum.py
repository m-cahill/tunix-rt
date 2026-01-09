#!/usr/bin/env python3
"""M45 Curriculum Training Orchestrator.

This script runs three sequential training stages:
- Stage A: Base Gemma → checkpoint_a (2 epochs on stage_a.jsonl)
- Stage B: checkpoint_a → checkpoint_b (2 epochs on stage_b.jsonl)
- Stage C: checkpoint_b → checkpoint_c (3 epochs on stage_c.jsonl)

All stages use identical hyperparameters (locked per M45_answers.md).
No interleaving. No retries unless infra failure.

Usage:
    cd research/m45_curriculum_reasoning
    python run_curriculum.py

    # Or run individual stages:
    python run_curriculum.py --stage A
    python run_curriculum.py --stage B --resume-from checkpoints/stage_a/final_model
    python run_curriculum.py --stage C --resume-from checkpoints/stage_b/final_model

Author: M45 Curriculum Reasoning Milestone
Date: 2026-01-08
"""

import argparse
import codecs
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
CONFIGS_DIR = SCRIPT_DIR / "configs"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

STAGES = {
    "A": {
        "config": CONFIGS_DIR / "stage_a.yaml",
        "data": DATA_DIR / "stage_a.jsonl",
        "output": CHECKPOINTS_DIR / "stage_a",
        "init_from": None,  # Base model
        "epochs": 2,
    },
    "B": {
        "config": CONFIGS_DIR / "stage_b.yaml",
        "data": DATA_DIR / "stage_b.jsonl",
        "output": CHECKPOINTS_DIR / "stage_b",
        "init_from": CHECKPOINTS_DIR / "stage_a" / "final_model",
        "epochs": 2,
    },
    "C": {
        "config": CONFIGS_DIR / "stage_c.yaml",
        "data": DATA_DIR / "stage_c.jsonl",
        "output": CHECKPOINTS_DIR / "stage_c",
        "init_from": CHECKPOINTS_DIR / "stage_b" / "final_model",
        "epochs": 3,
    },
}


# ============================================================
# SFT Dataset (copied from training_pt/train.py for isolation)
# ============================================================


class SFTDataset(Dataset):
    """Simple SFT dataset for curriculum training."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 128):
        """Initialize dataset with tokenized samples.

        Args:
            samples: List of trace dictionaries
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.encodings = []
        for s in samples:
            # Use 'prompts' (Tunix format) or build from 'prompt' + 'final_answer'
            text = s.get("prompts")
            if not text:
                # Build from raw trace
                prompt = s.get("prompt", "")
                answer = s.get("final_answer", "")
                steps = s.get("steps", [])
                # Format: prompt -> steps -> answer
                step_text = " ".join(step.get("content", "") for step in steps)
                text = f"{prompt}\n{step_text}\n{answer}"

            if not text.strip():
                continue

            try:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                self.encodings.append(enc)
            except Exception:
                continue

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings)


# ============================================================
# Training Logic
# ============================================================


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(data_path: Path) -> list[dict]:
    """Load JSONL dataset file."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def run_stage(
    stage_name: str,
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    init_from: Path | None = None,
) -> dict:
    """Run a single curriculum training stage.

    Args:
        stage_name: Stage identifier (A, B, or C)
        config_path: Path to stage config YAML
        data_path: Path to stage dataset JSONL
        output_dir: Output directory for checkpoints
        init_from: Path to previous checkpoint (None for base model)

    Returns:
        Dict with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Stage {stage_name}: Training")
    print(f"{'='*60}")

    start_time = datetime.now(timezone.utc)

    # Load config
    config = load_config(config_path)
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    # Determine model source
    if init_from and init_from.exists():
        model_id = str(init_from)
        print(f"[INIT] Loading from checkpoint: {init_from}")
    else:
        model_id = model_config.get("model_id", "google/gemma-2b")
        revision = model_config.get("revision", "main")
        print(f"[INIT] Loading base model: {model_id} (revision: {revision})")

    # Load dataset
    print(f"[DATA] Loading: {data_path}")
    samples = load_dataset(data_path)
    print(f"       {len(samples)} samples loaded")

    # Load tokenizer and model
    print(f"[MODEL] Loading tokenizer and model...")
    try:
        if init_from and init_from.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(init_from))
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.get("model_id", "google/gemma-2b"),
                revision=model_config.get("revision", "main"),
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine dtype
        torch_dtype = torch.float32
        if training_config.get("dtype") == "bfloat16":
            torch_dtype = torch.bfloat16
        elif training_config.get("dtype") == "float16":
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if training_config.get("device") != "cpu" else None,
        )
        print(f"       Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

    # Prepare dataset
    print(f"[TOKENIZE] Tokenizing dataset...")
    train_dataset = SFTDataset(
        samples,
        tokenizer,
        max_length=int(training_config.get("max_seq_length", 128)),
    )
    print(f"           {len(train_dataset)} samples after tokenization")

    if len(train_dataset) == 0:
        raise ValueError("Dataset empty after tokenization!")

    # Setup training arguments
    use_cpu = training_config.get("device") == "cpu" or not torch.cuda.is_available()
    optim = "adafactor" if training_config.get("optimizer") == "adafactor" else "adamw_torch"

    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        num_train_epochs=float(training_config.get("num_epochs", 2)),
        per_device_train_batch_size=int(training_config.get("batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
        logging_dir=str(output_dir / "logs"),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(training_config.get("save_every_steps", 50)),
        use_cpu=use_cpu,
        report_to="none",
        bf16=(training_config.get("dtype") == "bfloat16") and not use_cpu,
        fp16=(training_config.get("dtype") == "float16") and not use_cpu,
        optim=optim,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Run training
    print(f"[TRAIN] Starting training...")
    print(f"        Epochs: {training_config.get('num_epochs')}")
    print(f"        Batch size: {training_config.get('batch_size')}")
    print(f"        Grad accum: {training_config.get('gradient_accumulation_steps')}")

    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

    # Save final model
    final_model_dir = output_dir / "final_model"
    print(f"[SAVE] Saving final model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save metrics
    end_time = datetime.now(timezone.utc)
    metrics = {
        "stage": stage_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "samples": len(samples),
        "tokenized_samples": len(train_dataset),
        "epochs": training_config.get("num_epochs"),
        "init_from": str(init_from) if init_from else "base",
        "config_path": str(config_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
    }

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics saved to: {metrics_path}")

    print(f"\n[DONE] Stage {stage_name} complete!")
    print(f"       Loss: {train_result.training_loss:.4f}")
    print(f"       Steps: {train_result.global_step}")
    print(f"       Runtime: {metrics['runtime_seconds']:.1f}s")

    return metrics


def run_all_stages() -> list[dict]:
    """Run all three curriculum stages sequentially."""
    print("=" * 60)
    print("M45 Curriculum Training — Full Pipeline")
    print("=" * 60)
    print(f"Stages: A (2 epochs) → B (2 epochs) → C (3 epochs)")

    all_metrics = []

    for stage_name in ["A", "B", "C"]:
        stage = STAGES[stage_name]
        metrics = run_stage(
            stage_name=stage_name,
            config_path=stage["config"],
            data_path=stage["data"],
            output_dir=stage["output"],
            init_from=stage["init_from"],
        )
        all_metrics.append(metrics)

    # Save combined metrics
    combined_path = CHECKPOINTS_DIR / "curriculum_training_summary.json"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": sum(m["runtime_seconds"] for m in all_metrics),
        "stages": all_metrics,
    }
    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {combined_path}")

    print("\n" + "=" * 60)
    print("M45 Curriculum Training Complete!")
    print("=" * 60)
    for m in all_metrics:
        print(f"  Stage {m['stage']}: {m['training_loss']:.4f} loss, {m['global_step']} steps, {m['runtime_seconds']:.1f}s")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="M45 Curriculum Training Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["A", "B", "C", "all"],
        default="all",
        help="Stage to run (A, B, C, or all)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Override init checkpoint (for manual stage runs)",
    )

    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        run_all_stages()
    else:
        stage = STAGES[args.stage]
        init_from = args.resume_from if args.resume_from else stage["init_from"]
        run_stage(
            stage_name=args.stage,
            config_path=stage["config"],
            data_path=stage["data"],
            output_dir=stage["output"],
            init_from=init_from,
        )


if __name__ == "__main__":
    main()

