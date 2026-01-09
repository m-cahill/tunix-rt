#!/usr/bin/env python3
"""M47 Training Orchestrator — Error Correction Fidelity.

This script runs two parallel training experiments:
1. Clean: Training on clean data (M46-style, no errors)
2. Error-Aware: Training on error-injected data with explicit corrections

Both runs:
- Start from the M46 Self-Correct checkpoint
- Use identical hyperparameters (1 epoch each)
- Save to separate checkpoint directories

Usage:
    cd research/m47_error_correction_fidelity
    python scripts/run_training.py

    # Run individual experiments:
    python scripts/run_training.py --run clean
    python scripts/run_training.py --run error_aware

Author: M47 Error Correction Fidelity Milestone
Date: 2026-01-09
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

# M46 Self-Correct checkpoint (init for both runs)
M46_SELF_CORRECT_CHECKPOINT = PROJECT_DIR.parent / "m46_structured_self_correction" / "checkpoints" / "self_correct" / "final_model"

# Run configurations
RUNS = {
    "clean": {
        "config": CONFIGS_DIR / "clean.yaml",
        "data": DATA_DIR / "stage_c_clean.jsonl",
        "output": CHECKPOINTS_DIR / "clean",
        "description": "Clean (no errors)",
    },
    "error_aware": {
        "config": CONFIGS_DIR / "error_aware.yaml",
        "data": DATA_DIR / "stage_c_error_self_correct.jsonl",
        "output": CHECKPOINTS_DIR / "error_aware",
        "description": "Error-aware (with corrections)",
    },
}


# ============================================================
# SFT Dataset
# ============================================================


class SFTDataset(Dataset):
    """Simple SFT dataset for M47 training."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 128):
        self.encodings = []
        for s in samples:
            text = s.get("prompts")
            if not text:
                prompt = s.get("prompt", "")
                answer = s.get("final_answer", "")
                steps = s.get("steps", [])
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
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(data_path: Path) -> list[dict]:
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
    print(f"\n{'='*60}")
    print(f"M47 Experiment: {run_name}")
    print(f"{'='*60}")

    start_time = datetime.now(timezone.utc)

    config = load_config(config_path)
    training_config = config.get("training", {})

    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

    print(f"[INIT] Loading from M46 Self-Correct: {init_checkpoint}")

    print(f"[DATA] Loading: {data_path}")
    samples = load_dataset(data_path)
    print(f"       {len(samples)} samples loaded")

    print(f"[MODEL] Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(init_checkpoint))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

    print(f"[TOKENIZE] Tokenizing dataset...")
    train_dataset = SFTDataset(
        samples,
        tokenizer,
        max_length=int(training_config.get("max_seq_length", 128)),
    )
    print(f"           {len(train_dataset)} samples after tokenization")

    if len(train_dataset) == 0:
        raise ValueError("Dataset empty after tokenization!")

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

    print(f"[TRAIN] Starting training...")
    print(f"        Epochs: {training_config.get('num_epochs')}")
    print(f"        Samples: {len(train_dataset)}")

    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

    final_model_dir = output_dir / "final_model"
    print(f"[SAVE] Saving final model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

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

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return metrics


def run_all_experiments() -> list[dict]:
    print("=" * 60)
    print("M47 Error Correction Fidelity — Full Pipeline")
    print("=" * 60)
    print(f"Experiments: Clean (1 epoch) + Error-Aware (1 epoch)")
    print(f"Init checkpoint: {M46_SELF_CORRECT_CHECKPOINT}")

    all_metrics = []

    for run_name in ["clean", "error_aware"]:
        run_config = RUNS[run_name]
        metrics = run_experiment(
            run_name=run_name,
            config_path=run_config["config"],
            data_path=run_config["data"],
            output_dir=run_config["output"],
            init_checkpoint=M46_SELF_CORRECT_CHECKPOINT,
        )
        all_metrics.append(metrics)

    summary_path = CHECKPOINTS_DIR / "training_summary.json"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": sum(m["runtime_seconds"] for m in all_metrics),
        "init_checkpoint": str(M46_SELF_CORRECT_CHECKPOINT),
        "experiments": all_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("M47 Training Complete!")
    print("=" * 60)
    for m in all_metrics:
        print(f"  {m['run_name']}: {m['training_loss']:.4f} loss, {m['global_step']} steps, {m['runtime_seconds']:.1f}s")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="M47 Training Orchestrator")
    parser.add_argument(
        "--run",
        type=str,
        choices=["clean", "error_aware", "all"],
        default="all",
        help="Experiment to run",
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
            init_checkpoint=M46_SELF_CORRECT_CHECKPOINT,
        )


if __name__ == "__main__":
    main()

