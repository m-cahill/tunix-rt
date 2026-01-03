"""PyTorch/Transformers SFT Training Implementation (Local GPU Path)."""
import argparse
import json
import logging
import sys
import codecs
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.encodings = []
        for s in samples:
            # Use 'prompts' (Tunix format) or 'prompt' (raw trace)
            text = s.get("prompts") or s.get("prompt", "")
            if not text:
                continue
            try:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
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

def run_torch_sft_training(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run SFT training using PyTorch/Transformers."""
    print("\n🚀 Starting SFT Training (PyTorch/Transformers)...")

    model_config = config.get("model", {})
    model_id = model_config.get("model_id", "distilgpt2")
    revision = model_config.get("revision", "main")

    training_args_config = config.get("training", {})

    # M39: Explicitly log config
    print(f"   Model: {model_id} (revision: {revision})")
    print(f"   Device: {training_args_config.get('device', 'auto')}")
    print(f"   Batch Size: {training_args_config.get('batch_size', 4)}")
    print(f"   Grad Accum: {training_args_config.get('gradient_accumulation_steps', 1)}")
    print(f"   Dtype: {training_args_config.get('dtype', 'float32')}")

    # Load Tokenizer & Model
    print("   Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # M39: bfloat16 support
        torch_dtype = torch.float32
        if training_args_config.get("dtype") == "bfloat16":
            torch_dtype = torch.bfloat16
        elif training_args_config.get("dtype") == "float16":
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map="auto" if training_args_config.get("device") != "cpu" else None
        )
    except Exception as e:
        print(f"❌ Failed to load model/tokenizer: {e}")
        sys.exit(1)

    # Dataset Preparation
    print("   Tokenizing dataset...")
    train_dataset = SFTDataset(
        dataset,
        tokenizer,
        max_length=int(training_args_config.get("max_seq_length", 128))
    )

    if len(train_dataset) == 0:
        print("❌ Dataset empty after tokenization.")
        sys.exit(1)

    # Training Arguments
    use_cpu = training_args_config.get("device") == "cpu" or not torch.cuda.is_available()

    # Optimizer map
    optim = "adamw_torch"
    if training_args_config.get("optimizer") == "adafactor":
        optim = "adafactor"

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(training_args_config.get("learning_rate", 2e-5)),
        weight_decay=float(training_args_config.get("weight_decay", 1e-4)),
        num_train_epochs=float(training_args_config.get("num_epochs", 1.0)),
        per_device_train_batch_size=int(training_args_config.get("batch_size", 4)),
        gradient_accumulation_steps=int(training_args_config.get("gradient_accumulation_steps", 1)),
        logging_dir=str(output_dir / "logs"),
        logging_steps=int(training_args_config.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(training_args_config.get("save_every_steps", 100)),
        use_cpu=use_cpu,
        report_to="none",
        bf16=(training_args_config.get("dtype") == "bfloat16") and not use_cpu,
        fp16=(training_args_config.get("dtype") == "float16") and not use_cpu,
        optim=optim,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("   Training...")
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

    # Save Model
    print("   Saving final model...")
    checkpoint_dir = output_dir / "final_model"
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"✅ Saved model to: {checkpoint_dir}")

    # Save Metrics
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        metric = {
            "step": train_result.global_step,
            "loss": train_result.training_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": train_result.metrics
        }
        f.write(json.dumps(metric) + "\n")
    print(f"✅ Saved metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Tunix PyTorch SFT Training")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key (e.g., dev-reasoning-v2)")

    # Optional overrides
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Override device if provided
    if args.device:
        config.setdefault("training", {})["device"] = args.device

    # Resolve Dataset
    # Try typical paths
    candidates = [
        Path(f"backend/datasets/{args.dataset}/{args.dataset}.jsonl"),
        Path(f"datasets/{args.dataset}/{args.dataset}.jsonl"),
        Path(f"backend/datasets/{args.dataset}/dataset.jsonl"),
    ]

    dataset_path = None
    for c in candidates:
        if c.exists():
            dataset_path = c
            break

    if not dataset_path:
        print(f"❌ Dataset not found for key: {args.dataset}")
        print(f"   Checked: {[str(c) for c in candidates]}")
        sys.exit(1)

    print(f"   Loading dataset from: {dataset_path}")
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_torch_sft_training(config, dataset, output_dir)

if __name__ == "__main__":
    main()

