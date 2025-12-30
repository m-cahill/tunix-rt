#!/usr/bin/env python3
"""
Launcher for JAX/Flax SFT Training with GPU Memory Management.

This script sets critical XLA/JAX environment variables BEFORE importing JAX.
This is required because JAX reads these variables at import time.

See: https://docs.jax.dev/en/latest/gpu_memory_allocation.html

Usage:
    python training/run_train_jax.py --config <config.yaml> --output <dir> --dataset <name>

For smoke tests (memory-constrained):
    python training/run_train_jax.py --config <config.yaml> --output <dir> --dataset <name> --smoke_steps 2
"""
import os
import sys

# ============================================================================
# CRITICAL: GPU Memory Configuration
# Must be set BEFORE importing JAX (or any module that imports JAX)
# ============================================================================

# Disable GPU memory preallocation - JAX normally grabs 90% of VRAM at startup
# Setting to "false" makes JAX allocate memory on-demand
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Cap the fraction of GPU memory JAX can use (when preallocation is enabled)
# Lower values leave more headroom for model loading and other allocations
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

# Use the "platform" allocator which is more flexible for tight VRAM scenarios
# This can reduce fragmentation issues at the cost of some performance
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# ============================================================================
# Now safe to import JAX-dependent modules
# ============================================================================

# Import main function from train_jax (which imports JAX internally)
from train_jax import main

if __name__ == "__main__":
    sys.exit(main())
