#!/usr/bin/env python3
"""M46 Training Orchestrator — Structured Self-Correction.

This script runs two parallel training experiments:
1. Control: Training on unchanged Stage-C data
2. Self-Correction: Training on Stage-C with VERIFY/CORRECT blocks

Both runs:
- Start from the same M45 Stage-C checkpoint
- Use identical hyperparameters (1 epoch each)
- Save to separate checkpoint directories

Usage:
    cd research/m46_structured_self_correction
    python scripts/run_training.py

    # Run individual experiments:
    python scripts/run_training.py --run control
    python scripts/run_training.py --run self_correct

Author: M46 Structured Self-Correction Milestone
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
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIGS_DIR = PROJECT_DIR / "configs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

# M45 Stage-C checkpoint (shared init for both runs)
M45_STAGE_C_CHECKPOINT = PROJECT_DIR.parent / "m45_curriculum_reasoning" / "checkpoints" / "stage_c" / "final_model"

# Run configurations
RUNS = {
    "control": {
        "config": CONFIGS_DIR / "control.yaml",
        "data": DATA_DIR / "stage_c_control.jsonl",
        "output": CHECKPOINTS_DIR / "control",
        "description": "Unchanged Stage-C (baseline)",
    },
    "self_correct": {
        "config": CONFIGS_DIR / "self_correct.yaml",
        "data": DATA_DIR / "stage_c_self_correct.jsonl",
        "output": CHECKPOINTS_DIR / "self_correct",
        "description": "Stage-C with VERIFY/CORRECT",
    },
}


# ============================================================
# SFT Dataset (adapted from M45 run_curriculum.py)
# ============================================================


class SFTDataset(Dataset):
    """Simple SFT dataset for M46 training."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 128):
        """Initialize dataset with tokenized samples.

        Args:
            samples: List of trace dictionaries
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.encodings = []
        for s in samples:
            # Use 'prompts' (Tunix format) or build from 'prompt' + 'steps' + 'final_answer'
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


def run_experiment(
    run_name: str,
    config_path: Path,
    data_path: Path,
    output_dir: Path,
    init_checkpoint: Path,
) -> dict:
    """Run a single M46 training experiment.

    Args:
        run_name: Experiment name (control or self_correct)
        config_path: Path to config YAML
        data_path: Path to dataset JSONL
        output_dir: Output directory for checkpoints
        init_checkpoint: Path to M45 Stage-C checkpoint

    Returns:
        Dict with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"M46 Experiment: {run_name}")
    print(f"{'='*60}")

    start_time = datetime.now(timezone.utc)

    # Load config
    config = load_config(config_path)
    training_config = config.get("training", {})

    # Verify init checkpoint exists
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

    print(f"[INIT] Loading from M45 Stage-C: {init_checkpoint}")

    # Load dataset
    print(f"[DATA] Loading: {data_path}")
    samples = load_dataset(data_path)
    print(f"       {len(samples)} samples loaded")

    # Load tokenizer and model
    print(f"[MODEL] Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(init_checkpoint))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine dtype
        torch_dtype = torch.float32
        if training_config.get("dtype") == "bfloat16":
            torch_dtype = torch.bfloat16
        elif training_config.get("dtype") == "float16":
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            str(init_checkpoint),
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
        num_train_epochs=float(training_config.get("num_epochs", 1)),
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
    print(f"        Samples: {len(train_dataset)}")

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
        "run_name": run_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "samples": len(samples),
        "tokenized_samples": len(train_dataset),
        "epochs": training_config.get("num_epochs"),
        "init_checkpoint": str(init_checkpoint),
        "config_path": str(config_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
    }

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics saved to: {metrics_path}")

    print(f"\n[DONE] {run_name} complete!")
    print(f"       Loss: {train_result.training_loss:.4f}")
    print(f"       Steps: {train_result.global_step}")
    print(f"       Runtime: {metrics['runtime_seconds']:.1f}s")

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return metrics


def run_all_experiments() -> list[dict]:
    """Run both control and self-correction experiments sequentially."""
    print("=" * 60)
    print("M46 Structured Self-Correction — Full Pipeline")
    print("=" * 60)
    print(f"Experiments: Control (1 epoch) + Self-Correct (1 epoch)")
    print(f"Init checkpoint: {M45_STAGE_C_CHECKPOINT}")

    all_metrics = []

    for run_name in ["control", "self_correct"]:
        run_config = RUNS[run_name]
        metrics = run_experiment(
            run_name=run_name,
            config_path=run_config["config"],
            data_path=run_config["data"],
            output_dir=run_config["output"],
            init_checkpoint=M45_STAGE_C_CHECKPOINT,
        )
        all_metrics.append(metrics)

    # Save combined summary
    summary_path = CHECKPOINTS_DIR / "training_summary.json"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": sum(m["runtime_seconds"] for m in all_metrics),
        "init_checkpoint": str(M45_STAGE_C_CHECKPOINT),
        "experiments": all_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("M46 Training Complete!")
    print("=" * 60)
    for m in all_metrics:
        print(f"  {m['run_name']}: {m['training_loss']:.4f} loss, {m['global_step']} steps, {m['runtime_seconds']:.1f}s")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="M46 Training Orchestrator")
    parser.add_argument(
        "--run",
        type=str,
        choices=["control", "self_correct", "all"],
        default="all",
        help="Experiment to run (control, self_correct, or all)",
    )

    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run == "all":
        run_all_experiments()
    else:
        run_config = RUNS[args.run]
        run_experiment(
            run_name=args.run,
            config_path=run_config["config"],
            data_path=run_config["data"],
            output_dir=run_config["output"],
            init_checkpoint=M45_STAGE_C_CHECKPOINT,
        )


if __name__ == "__main__":
    main()

