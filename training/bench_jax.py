"""JAX/Flax Benchmarking Script for Tunix RT."""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

def run_benchmark(
    config: dict[str, Any],
    output_dir: Path,
    device: str | None = None,
    device_index: int = 0,
    steps: int = 50,
    warmup_steps: int = 10,
    profile: bool = False,
) -> None:
    """Run throughput benchmark using JAX/Flax."""
    print("\n🚀 Starting JAX Benchmark...")

    try:
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state
        from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"❌ JAX/Flax dependencies not installed: {e}")
        sys.exit(1)

    # Config
    model_id = config.get("model", {}).get("model_id", "distilgpt2")
    training_args = config.get("training", {})
    batch_size = int(training_args.get("batch_size", 4))
    seq_length = int(training_args.get("max_seq_length", 128))

    # Device Selection
    requested_device = device or training_args.get("device", "auto")

    if requested_device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif requested_device == "gpu":
        try:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("No GPU found")
            if device_index >= len(gpus):
                 print(f"❌ Requested GPU index {device_index} but only found {len(gpus)} GPUs")
                 sys.exit(1)
            try:
                jax.config.update("jax_default_device", gpus[device_index])
            except AttributeError:
                pass
        except RuntimeError:
            print("❌ Device 'gpu' requested but no GPU devices found.")
            sys.exit(1)

    print(f"   Model: {model_id}")
    try:
        print(f"   Device: {jax.devices()[0]}")
    except Exception:
        print("   Device: Unknown")

    # 1. Load Tokenizer & Model
    print("   Loading model...")
    try:
        # Load random weights or pretrained? Pretrained is better for realism of memory usage
        config_args = {"n_layer": 2, "n_head": 4, "n_embd": 128} if "tiny" in model_id else {}
        if config_args:
            # If using a tiny dummy model for speed
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, **config_args)
            model = FlaxAutoModelForCausalLM.from_config(cfg)
        else:
             try:
                 model = FlaxAutoModelForCausalLM.from_pretrained(model_id, from_pt=True)
             except OSError:
                 model = FlaxAutoModelForCausalLM.from_pretrained(model_id, from_pt=False)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 2. Prepare Dummy Batch
    print(f"   Creating dummy batch (BS={batch_size}, SeqLen={seq_length})...")
    dummy_input = np.random.randint(0, 1000, (batch_size, seq_length), dtype=np.int32)
    dummy_mask = np.ones((batch_size, seq_length), dtype=np.int32)

    batch = {
        "input_ids": dummy_input,
        "attention_mask": dummy_mask
    }

    # 3. Training Setup (Minimal)
    tx = optax.adamw(learning_rate=1e-4)

    class TrainState(train_state.TrainState):
        pass

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
    )

    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                params=params
            ).logits
            # Simple loss calculation for computation load
            loss = logits.sum()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    train_step_jit = jax.jit(train_step)

    # 4. Benchmark Loop
    print(f"   Running {steps} steps (Warmup: {warmup_steps})...")

    # Warmup
    for _ in range(warmup_steps):
        state = train_step_jit(state, batch)
        # Block until ready
        jax.block_until_ready(state)

    # Profile if requested
    if profile:
        print("   Profiling active...")
        jax.profiler.start_trace(str(output_dir / "trace"))

    start_time = time.perf_counter()

    for i in range(steps):
        state = train_step_jit(state, batch)
        jax.block_until_ready(state)

    end_time = time.perf_counter()

    if profile:
        jax.profiler.stop_trace()
        print(f"   Profile saved to {output_dir / 'trace'}")

    total_time = end_time - start_time
    steps_per_sec = steps / total_time
    tokens_per_sec = (steps * batch_size * seq_length) / total_time

    print(f"\n✅ Benchmark Complete:")
    print(f"   Time: {total_time:.4f}s")
    print(f"   Throughput: {steps_per_sec:.2f} steps/sec")
    print(f"   Throughput: {tokens_per_sec:.2f} tokens/sec")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "steps": steps,
        "total_time": total_time,
        "steps_per_sec": steps_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "device": str(jax.devices()[0]),
    }

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {output_dir / 'benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Tunix JAX Benchmark")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    # Load config
    try:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except ImportError:
        print("❌ PyYAML not found. Install backend[training] or pyyaml.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    run_benchmark(
        config=config,
        output_dir=args.output,
        device=args.device,
        device_index=args.device_index,
        steps=args.steps,
        warmup_steps=args.warmup,
        profile=args.profile,
    )

if __name__ == "__main__":
    main()
