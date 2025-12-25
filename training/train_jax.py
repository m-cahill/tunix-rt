"""JAX/Flax SFT Training Implementation (Real Tunix Path)."""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

def run_jax_sft_training(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run SFT training using JAX/Flax/Optax."""
    print("\n🚀 Starting SFT Training (JAX/Flax)...")

    try:
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state
        from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"❌ JAX/Flax dependencies not installed: {e}")
        print("   Install with: pip install -e '.[training]'")
        sys.exit(1)

    # Config
    model_id = config.get("model", {}).get("model_id", "distilgpt2")
    training_args = config.get("training", {})
    learning_rate = float(training_args.get("learning_rate", 2e-5))
    num_epochs = int(training_args.get("num_epochs", 3))
    batch_size = int(training_args.get("batch_size", 4))
    max_length = int(training_args.get("max_seq_length", 128))
    seed = int(training_args.get("seed", 42))
    device_config = training_args.get("device", "auto")

    if device_config == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print(f"   Model: {model_id}")
    try:
        print(f"   Device: {jax.devices()[0]}")
    except Exception:
        print("   Device: Unknown (JAX init issue?)")

    # 1. Load Tokenizer & Model
    print("   Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try loading from PT weights if Flax weights don't exist
        try:
            model = FlaxAutoModelForCausalLM.from_pretrained(model_id, from_pt=True)
        except OSError:
             # Maybe it has Flax weights?
            model = FlaxAutoModelForCausalLM.from_pretrained(model_id, from_pt=False)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 2. Prepare Dataset
    print("   Tokenizing dataset...")
    encodings = []
    for s in dataset:
        text = s.get("prompts", "")
        if not text:
            continue
        try:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="np"
            )
            encodings.append({
                "input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0]
            })
        except Exception:
            continue

    if not encodings:
        print("❌ Dataset empty after tokenization")
        sys.exit(1)

    # Convert to batches
    data_size = len(encodings)
    # Ensure at least one batch
    if batch_size > data_size:
        print(f"⚠️  Warning: Dataset size ({data_size}) smaller than batch size ({batch_size}). Adjusting.")
        batch_size = data_size

    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1

    def get_batch(data_source, step_idx):
        start = step_idx * batch_size
        end = start + batch_size
        # Handle last batch (drop remainder or include? standard is usually drop if not full or pad)
        # For simplicity, slice
        batch_items = data_source[start:end]
        if not batch_items:
            return None

        batch = {
            "input_ids": np.stack([x["input_ids"] for x in batch_items]),
            "attention_mask": np.stack([x["attention_mask"] for x in batch_items])
        }
        return batch

    # 3. Training Setup
    tx = optax.adamw(learning_rate=learning_rate)

    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
    )

    # Loss function
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                params=params
            ).logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = batch["input_ids"][..., 1:]
            shift_mask = batch["attention_mask"][..., 1:]

            loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
            loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-9)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    train_step_jit = jax.jit(train_step)

    # 4. Training Loop
    print("   Training...")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    rng = jax.random.PRNGKey(seed)

    global_step = 0
    import random

    for epoch in range(num_epochs):
        # Shuffle
        random.seed(seed + epoch)
        random.shuffle(encodings)

        for step in range(steps_per_epoch):
            batch = get_batch(encodings, step)
            if batch is None:
                break

            state, loss = train_step_jit(state, batch)
            loss_val = jax.device_get(loss)

            global_step += 1

            if global_step % 10 == 0 or step == steps_per_epoch - 1:
                print(f"   Epoch {epoch+1}/{num_epochs} Step {step+1}/{steps_per_epoch} Loss: {loss_val:.4f}")
                metric = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss_val),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                metrics_file.write(json.dumps(metric) + "\n")
                metrics_file.flush()

    metrics_file.close()

    # 5. Save Model
    print("   Saving model...")
    checkpoint_dir = output_dir / "final_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(checkpoint_dir), params=state.params)
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"✅ Saved model to: {checkpoint_dir}")
    print(f"✅ Saved metrics: {metrics_path}")
