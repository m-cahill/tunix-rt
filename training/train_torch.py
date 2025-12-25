"""PyTorch/Transformers SFT Training Implementation."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def run_torch_sft_training(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run SFT training using PyTorch/Transformers."""
    print("\n🚀 Starting SFT Training (PyTorch/Transformers)...")

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling
        )
        from torch.utils.data import Dataset
    except ImportError:
        print("❌ PyTorch/Transformers not installed.")
        sys.exit(1)

    model_id = config.get("model", {}).get("model_id", "distilgpt2")
    training_args_config = config.get("training", {})

    print(f"   Model: {model_id}")
    print(f"   Steps: {training_args_config.get('num_epochs', 1)} epochs (or max_steps)")

    # Load Tokenizer & Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ Failed to load model/tokenizer: {e}")
        sys.exit(1)

    # Simple Dataset Wrapper
    class SFTDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length=128):
            self.encodings = []
            for s in samples:
                # Use 'prompts' as text. In Tunix format, this contains the full turn usually.
                text = s.get("prompts", "")
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

    train_dataset = SFTDataset(
        dataset,
        tokenizer,
        max_length=training_args_config.get("max_seq_length", 128)
    )

    if len(train_dataset) == 0:
        print("❌ Dataset empty after tokenization.")
        sys.exit(1)

    # Training Arguments
    device_config = training_args_config.get("device", "auto")
    use_cpu = not torch.cuda.is_available()
    if device_config == "cpu":
        use_cpu = True
    elif device_config == "cuda":
        if not torch.cuda.is_available():
             print("❌ CUDA requested but not available.")
             sys.exit(1)
        use_cpu = False

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(training_args_config.get("num_epochs", 1.0)),
        per_device_train_batch_size=int(training_args_config.get("batch_size", 4)),
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=use_cpu,
        report_to="none",
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
        }
        f.write(json.dumps(metric) + "\n")
    print(f"✅ Saved metrics: {metrics_path}")
