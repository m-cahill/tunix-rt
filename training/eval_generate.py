import logging
import sys
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except Exception:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles model loading and generation (Auto-detect backend)."""

    def __init__(self, model_path: str, device: str = "auto", seed: int = 42):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.backend = "pytorch" # Default
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")

        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            sys.exit(1)

        # Check for Flax weights
        is_flax = (Path(self.model_path) / "flax_model.msgpack").exists()

        if is_flax:
            self._load_flax_model()
        else:
            self._load_pytorch_model()

    def _load_flax_model(self):
        """Load Flax model."""
        logger.info("Detected Flax weights. Using JAX/Flax backend.")
        self.backend = "flax"
        try:
            import jax
            from transformers import FlaxAutoModelForCausalLM

            self.model = FlaxAutoModelForCausalLM.from_pretrained(self.model_path)
            self.params = self.model.params

            # JIT compile generate function?
            # Transformers Flax generate is not fully JIT-ed by default but works.

        except ImportError:
            logger.error("JAX/Flax not installed but Flax weights found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load Flax model: {e}")
            sys.exit(1)

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        logger.info("Using PyTorch backend.")
        try:
            import torch
            from transformers import AutoModelForCausalLM

            # Device setup
            if self.device == "auto":
                self.device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device_obj = torch.device(self.device)

            logger.info(f"Using device: {self.device_obj}")

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device_obj)
            self.model.eval()

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        except ImportError:
            logger.error("PyTorch not installed.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            sys.exit(1)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate response for a prompt."""
        try:
            if self.backend == "flax":
                return self._generate_flax(prompt, max_new_tokens)
            else:
                return self._generate_pytorch(prompt, max_new_tokens)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR: Generation failed: {e}]"

    def _generate_flax(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="np")

        # Simple greedy generation
        # Flax `generate` expects `params` if using `FlaxAutoModel`?
        # No, `model.generate` handles it if `model.params` is set or passed.
        # But `FlaxAutoModelForCausalLM` wraps it.

        sequences = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        ).sequences

        # sequences is a JAX array
        import jax
        sequences_list = jax.device_get(sequences).tolist()

        # Decode
        text = self.tokenizer.decode(sequences_list[0], skip_special_tokens=True)
        # Remove prompt from output if present (model.generate includes it)
        if text.startswith(prompt):
            text = text[len(prompt):]

        return text.strip()

    def _generate_pytorch(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_obj)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()


def load_eval_set(eval_path: Path) -> list[dict]:
    """Load evaluation set from JSONL."""
    examples = []
    if not eval_path.exists():
        logger.error(f"Eval set not found: {eval_path}")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {i+1}: {e}")
                continue
    return examples


def create_trace_from_eval(
    eval_example: dict,
    generated_response: str,
    model: str,
) -> dict:
    """Create a trace-like record from eval output."""
    trace = {
        "trace_version": "1.0",
        "prompt": eval_example.get("prompt", ""),
        "final_answer": generated_response,
        "steps": [
            {
                "i": 0,
                "type": "reasoning",
                "content": "Model-generated reasoning step",
            }
        ],
        "meta": {
            "source": "evaluation",
            "eval_id": eval_example.get("id"),
            "model": model,
            "category": eval_example.get("category", "unknown"),
        },
    }
    return trace


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate evaluation outputs from a model")
    parser.add_argument("--model", type=str, required=True, help="Model to use ('base' or path to checkpoint)")
    parser.add_argument("--eval-set", type=Path, required=True, help="Path to eval set JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output path for generated traces JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum examples to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")

    args = parser.parse_args()

    print(" Tunix RT - Evaluation Generation (Real Inference)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Eval set: {args.eval_set}")
    print(f"Seed: {args.seed}")

    # Load engine
    engine = InferenceEngine(args.model, device=args.device, seed=args.seed)

    # Load eval set
    eval_examples = load_eval_set(args.eval_set)
    print(f" Loaded {len(eval_examples)} eval examples")

    if args.max_examples:
        eval_examples = eval_examples[:args.max_examples]
        print(f"   Limited to {len(eval_examples)} examples")

    # Generate responses
    print("\n Generating responses...")
    generated_traces = []

    for i, example in enumerate(eval_examples):
        prompt = example.get("prompt", "")
        print(f"   [{i+1}/{len(eval_examples)}] {example.get('id', 'unknown')}: {prompt[:50]}...")

        response = engine.generate(prompt)

        trace = create_trace_from_eval(
            eval_example=example,
            generated_response=response,
            model=args.model,
        )
        generated_traces.append(trace)

    # Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for trace in generated_traces:
            f.write(json.dumps(trace) + "\n")

    print(f"\n Saved {len(generated_traces)} traces to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
