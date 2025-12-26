"""JAX/Flax SFT Training Implementation (Real Tunix Path)."""
import argparse
import json
import logging
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass

logger = logging.getLogger(__name__)

def run_jax_sft_training(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    output_dir: Path,
    device: str | None = None,
    device_index: int = 0,
    smoke_steps: int | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: str | None = None,
    save_every_steps: int | None = None,
    eval_after_train: bool = False,
    eval_set: Path | None = None,
) -> None:
    """Run SFT training using JAX/Flax/Optax."""
    print("\n🚀 Starting SFT Training (JAX/Flax)...")

    try:
        import jax
        import jax.numpy as jnp
        import optax
        import orbax.checkpoint as ocp
        from flax.training import train_state, orbax_utils
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

    # Device Selection
    requested_device = device or training_args.get("device", "auto")

    if requested_device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
        print("   Device request: CPU")
    elif requested_device == "gpu":
        # Fail if GPU requested but not available
        try:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("No GPU found")
            # Select specific GPU if index provided
            if device_index >= len(gpus):
                 print(f"❌ Requested GPU index {device_index} but only found {len(gpus)} GPUs")
                 sys.exit(1)

            # Set default device (JAX 0.4.x+)
            try:
                jax.config.update("jax_default_device", gpus[device_index])
            except AttributeError:
                # Fallback for older JAX
                pass

            print(f"   Device request: GPU (Index {device_index})")
        except RuntimeError:
            print("❌ Device 'gpu' requested but no GPU devices found.")
            sys.exit(1)
    else:
        # auto
        print("   Device request: Auto")

    print(f"   Model: {model_id}")
    try:
        print(f"   Active Device: {jax.devices()[0]}")
        print(f"   Platform: {jax.default_backend()}")
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
        # Support both 'prompts' (Tunix SFT format) and 'prompt' (raw trace format)
        text = s.get("prompts") or s.get("prompt", "")
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

    # Checkpoint Manager Setup
    ckpt_path = checkpoint_dir or output_dir / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=2,
        save_interval_steps=save_every_steps or 100
    )
    # Using StandardCheckpointer which handles PyTrees (TrainState) automatically
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_path,
        checkpointer,
        ckpt_options
    )

    # Resume if requested
    start_step = 0
    start_epoch = 0

    if resume_from:
        print(f"   Resuming from: {resume_from}")
        if resume_from == "latest":
             step = checkpoint_manager.latest_step()
             if step is not None:
                 state = checkpoint_manager.restore(step, args=ocp.args.StandardRestore(state))
                 start_step = step
                 # Estimate epoch (approx)
                 start_epoch = start_step // steps_per_epoch
                 print(f"   Resumed at step {start_step} (Epoch ~{start_epoch})")
             else:
                 print("   ⚠️ No checkpoint found to resume from. Starting fresh.")
        else:
             # Resume from specific path/step not fully implemented in this basic script,
             # assuming resume_from maps to managing directory logic or specific step int if needed.
             # For M26, 'latest' via manager is the key requirement.
             pass

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
    # Append if resuming? For now, we overwrite or append. 'a' is safer for resume.
    metrics_file = open(metrics_path, "a" if resume_from else "w")

    rng = jax.random.PRNGKey(seed)

    global_step = start_step
    import random

    # Adjust epochs for resume
    for epoch in range(start_epoch, num_epochs):
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

            # Logging
            if global_step % 10 == 0 or step == steps_per_epoch - 1:
                print(f"   Epoch {epoch+1}/{num_epochs} Step {step+1}/{steps_per_epoch} (Global {global_step}) Loss: {loss_val:.4f}")
                metric = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss_val),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "device": str(jax.devices()[0]),
                }
                metrics_file.write(json.dumps(metric) + "\n")
                metrics_file.flush()

            # Checkpointing
            checkpoint_manager.save(
                global_step,
                args=ocp.args.StandardSave(state)
            )

            # Smoke Test / Early Exit
            if smoke_steps and global_step >= smoke_steps:
                print(f"   🛑 Smoke steps limit reached ({smoke_steps}). Stopping.")
                metrics_file.close()
                return

    metrics_file.close()

    # 5. Save Final Model
    print("   Saving final model...")
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_dir), params=state.params)
    tokenizer.save_pretrained(str(final_dir))

    # Wait for explicit checkpoint manager completion
    checkpoint_manager.wait_until_finished()

    print(f"✅ Saved model to: {final_dir}")
    print(f"✅ Saved metrics: {metrics_path}")

    # 6. Evaluation (Optional)
    if eval_after_train:
        print("\n🔍 Running post-training evaluation...")
        if not eval_set or not eval_set.exists():
             print(f"⚠️  Eval set not found: {eval_set}. Skipping evaluation.")
        else:
            eval_script = Path(__file__).parent / "eval_generate.py"
            cmd = [
                sys.executable,
                str(eval_script),
                "--model", str(final_dir),
                "--eval-set", str(eval_set),
                "--output", str(output_dir / "eval_results.jsonl"),
            ]
            print(f"   Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Evaluation complete. Results: {output_dir / 'eval_results.jsonl'}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tunix JAX SFT Training")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")

    # Dataset arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=Path, help="Path to JSONL dataset file (legacy)")
    group.add_argument("--dataset", type=str, help="Dataset key (e.g., golden-v2)")

    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--smoke_steps", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--eval_after_train", action="store_true", help="Run evaluation after training (offline mode)")
    parser.add_argument("--eval_set", type=Path, default=Path("training/evalsets/eval_v1.jsonl"), help="Path to evaluation dataset")

    args = parser.parse_args()

    # Load config
    try:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except ImportError:
        print("❌ PyYAML not found. Install backend[training] or pyyaml.")
        sys.exit(1)

    # Resolve dataset path
    dataset_path = None
    if args.data:
        dataset_path = args.data
    elif args.dataset:
        # Resolve key to backend/datasets/{key}/dataset.jsonl or datasets/{key}/dataset.jsonl
        # Try local paths first
        candidates = [
            Path("backend/datasets") / args.dataset / f"{args.dataset}.jsonl",
            Path("datasets") / args.dataset / f"{args.dataset}.jsonl",
            Path("backend/datasets") / args.dataset / "dataset.jsonl", # Fallback naming?
            Path("datasets") / args.dataset / "dataset.jsonl",
        ]

        # M26 standardizes on manifest.json but raw file might be dataset.jsonl or {key}.jsonl
        # Let's try to find manifest and read 'path'? Or just check file existence.
        # tools/seed_dataset.py writes to datasets/{key}/manifest.json.
        # But where is the JSONL? Usually kept alongside or in database?
        # seed_golden_v2.py wrote: "Created dataset golden-v2 at .../manifest.json"
        # The actual traces are in DB.
        # But train_jax.py needs a JSONL file.
        # The build_dataset_manifest logic in backend DOES NOT write a JSONL by default, it writes manifest.json.
        # However, `tunix_execution.py` handles EXPORTING the dataset to JSONL before training.
        # Offline training (manual) expects a JSONL file.
        # If I use --dataset, I should probably fail if JSONL doesn't exist, OR implicitly export it?
        # Exporting requires DB access which `train_jax.py` might not have (it's a training script).
        # So --dataset implies expecting a pre-exported file at a standard location.
        # seed_golden_v2.py log said: "Created dataset golden-v2 at D:\Coding\tunix-rt\backend\datasets\golden-v2\manifest.json".
        # It doesn't seem to export JSONL.
        # So for offline training, the user might need to export it first?
        # OR `train_jax.py` could assume there's a file.
        # For M27 manual run, I will assume the user exports it or I point --data to it.
        # But the prompt says "Add --dataset support... resolves to backend/datasets/golden-v2/dataset.jsonl".
        # So I will check for that file.

        for cand in candidates:
            if cand.exists():
                dataset_path = cand
                break

        if not dataset_path:
             print(f"❌ Could not find dataset file for key '{args.dataset}'")
             print(f"   Checked: {[str(c) for c in candidates]}")
             print("   Make sure to export the dataset to JSONL first.")
             sys.exit(1)

    # Load data
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except:
                pass

    args.output.mkdir(parents=True, exist_ok=True)
    output_dir_abs = args.output.resolve()

    run_jax_sft_training(
        config=config,
        dataset=dataset,
        output_dir=output_dir_abs,
        device=args.device,
        device_index=args.device_index,
        smoke_steps=args.smoke_steps,
        checkpoint_dir=args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
        resume_from=args.resume_from,
        save_every_steps=args.save_every_steps,
        eval_after_train=args.eval_after_train,
        eval_set=args.eval_set,
    )

if __name__ == "__main__":
    main()
