"""
utilities/seed.py

Purpose
-------
Set global random seeds across Python, NumPy, and PyTorch to improve
reproducibility of experiments. Also configures a few PyTorch/CuDNN flags
to favour deterministic behaviour on GPU.

What is implemented
-------------------
- `set_all_seeds(seed: int)`: sets seeds for `random`, `numpy`, and `torch`.
  If CUDA is available, sets CUDA seeds for all devices. Also sets
  `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

Design notes and caveats
-----------------------
- Deterministic behaviour in deep learning is difficult to guarantee fully:
  some CUDA kernels and third‑party libraries may still introduce nondeterminism.
  The function sets the common flags that maximise determinism for PyTorch.
- Setting `cudnn.deterministic = True` can slow down training; use it for
  reproducibility runs and debugging, and consider disabling it for final
  large-scale training runs where throughput matters.
- This helper does not set environment variables such as `PYTHONHASHSEED`
  or control other libraries (e.g., TensorFlow). If you need full process-level
  determinism, set `PYTHONHASHSEED` and other environment flags before Python starts.

Suggested usage
---------------
Call `set_all_seeds(seed)` at the start of each independent experiment (seed run),
before constructing models, environments, or data loaders.

Minimal tests
-------------
- Call `set_all_seeds(42)` twice and verify that a small sequence of random draws
  from `random`, `numpy`, and `torch` are identical across runs.
- Run a short training step twice with the same seed and confirm identical
  model parameter updates (within floating point tolerance) when `cudnn.deterministic`
  is enabled.

Example
-------
>>> set_all_seeds(1234)
>>> import random, numpy as np, torch
>>> random.random(), np.random.rand(), torch.rand(1)
(0.9664535356921388, array([0.19151945]), tensor([0.2961]))
"""

import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    """
    Set random seeds for Python, NumPy, and PyTorch to improve reproducibility.

    Args:
        seed (int): integer seed to use for all RNGs.

    Behaviour:
        - Seeds Python's `random` module.
        - Seeds NumPy's RNG.
        - Seeds PyTorch CPU RNG and, if available, all CUDA device RNGs.
        - Sets CuDNN to deterministic mode and disables benchmarking to reduce
          nondeterministic algorithm selection.

    Notes and limitations:
        - Full determinism is not guaranteed across all hardware and PyTorch
          versions. Some CUDA operations remain nondeterministic even with these
          flags. Use this function to make runs as reproducible as reasonably
          possible for research experiments.
        - Enabling `torch.backends.cudnn.deterministic = True` may reduce
          performance. For large production runs where exact reproducibility
          is not required, consider disabling it.
    """
    # Python stdlib RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # If CUDA is available, seed all CUDA devices for reproducibility across GPUs.
    if torch.cuda.is_available():
        # Seeds all GPUs (if multiple) with the same seed.
        torch.cuda.manual_seed_all(seed)

    # Configure CuDNN for deterministic behaviour.
    # - deterministic=True forces CuDNN to use deterministic algorithms when available.
    # - benchmark=False prevents CuDNN from selecting algorithms based on runtime
    #   heuristics which can introduce nondeterminism.
    # These settings are standard for reproducible research but may slow down training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Quick verification snippet (for tests or README)
# -------------------------
# Example smoke test to include in a test file:
#
# def test_set_all_seeds_reproducible():
#     set_all_seeds(123)
#     a1 = (random.random(), np.random.rand(), torch.rand(1).item())
#     set_all_seeds(123)
#     a2 = (random.random(), np.random.rand(), torch.rand(1).item())
#     assert a1 == a2
#
# Note: Floating point draws from torch may differ in shape/dtype; compare
# values within a tolerance if necessary.
