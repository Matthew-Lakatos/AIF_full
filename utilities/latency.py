"""
utilities/latency.py

Purpose
-------
Deterministic latency buffer used to simulate fixed observation delays
(compatible with Dreamer-style temporal alignment). The buffer provides a
simple FIFO delay mechanism with explicit reset semantics for episode
boundaries. It is intentionally minimal and deterministic to make behaviour
predictable for experiments involving delayed or repeated observations.

What is implemented
-------------------
- Class `LatencyBuffer`:
    - Fixed, deterministic latency (no randomness).
    - FIFO queue semantics.
    - `reset()` to clear buffer at episode start.
    - `push(obs)` to append a new observation.
    - `pop()` to retrieve the observation from exactly `latency_steps` ago,
      or `None` if the buffer is not yet full.
- The buffer stores observations as-is (numpy arrays, torch tensors, or
  other objects) to avoid unnecessary conversions; callers are responsible
  for device/dtype handling.

Design notes and experimental uses
---------------------------------
- This buffer is useful when evaluating agents under delayed sensing or
  when reproducing Dreamer-style latency experiments where the agent acts
  on slightly stale observations.
- Because behaviour is deterministic, it is suitable for reproducibility
  and for use with overshooting / multi-step prediction losses that expect
  predictable temporal structure.
- Keep `latency_steps` small relative to episode length; large latencies
  may require changes to agent memory or planner horizons.

Suggested ablations / experiments
--------------------------------
- Sweep `latency_steps` ∈ {0, 1, 2, 5, 10} to measure sensitivity of blackout
  robustness to observation delay.
- Compare deterministic FIFO delay vs stochastic delay (random jitter) to
  evaluate robustness to unpredictable latency.
- Use the buffer with and without replay to test whether repeated observations
  (when buffer not full) affect learning dynamics.

Minimal tests to include in repo
--------------------------------
- Forward/backward smoke test: push a sequence of dummy observations and
  assert `pop()` returns `None` until buffer is full, then returns the
  expected delayed observation.
- Reset test: after `reset()`, buffer should be empty and `pop()` should
  return `None`.
"""

import torch


class LatencyBuffer:
    """
    Deterministic latency buffer for Dreamer-style temporal alignment.

    Behaviour summary
    -----------------
    - `latency_steps` controls how many steps observations are delayed.
      If `latency_steps == 0`, the buffer returns the most recent observation.
    - `push(obs)` appends an observation to the internal FIFO list.
    - `pop()` returns the observation from exactly `latency_steps` steps ago,
      or `None` if the buffer does not yet contain enough history.
    - `reset()` clears the buffer at episode boundaries.

    Notes on stored objects
    -----------------------
    - Observations are stored as-is (no conversion). They can be numpy arrays,
      torch tensors, or other serialisable objects. The caller should ensure
      consistent types and device placement when passing observations to agents.
    """

    def __init__(self, latency_steps=0):
        """
        Initialize the latency buffer.

        Args:
            latency_steps (int): number of steps to delay observations.
                                 0 means no latency (pop returns latest).
        """
        # Ensure non-negative latency
        self.latency_steps = max(0, int(latency_steps))
        # Internal FIFO storage; append with push(), read with pop()
        self.buffer = []

    def reset(self):
        """Clear the buffer at episode start so the next episode begins fresh."""
        self.buffer = []

    def push(self, obs):
        """
        Add a new observation to the buffer.

        Args:
            obs: observation object (numpy array, torch.Tensor, list, etc.)
                 Stored as-is to avoid implicit device/dtype conversions.

        Note (minor fix): the buffer is trimmed to only retain the
        `latency_steps + 1` most recent entries, since `pop()` never needs
        anything older than that. Previously the buffer grew unboundedly for
        the entire episode even though only a small tail was ever read.
        """
        self.buffer.append(obs)
        max_needed = self.latency_steps + 1
        if len(self.buffer) > max_needed:
            del self.buffer[:-max_needed]

    def pop(self):
        """
        Retrieve the delayed observation.

        Returns:
            - If latency_steps == 0: the most recent observation (buffer[-1]),
              or None if buffer is empty.
            - If latency_steps > 0: the observation from exactly
              `latency_steps` steps ago, i.e., buffer[-(latency_steps + 1)],
              or None if the buffer does not yet contain enough history.
        """
        # No latency: return the latest observation immediately (if any).
        if self.latency_steps == 0:
            return self.buffer[-1] if self.buffer else None

        # If buffer length is less than or equal to latency_steps, we don't yet
        # have an observation that is old enough to return.
        if len(self.buffer) <= self.latency_steps:
            return None

        # Deterministic FIFO delay: return the element that was pushed
        # `latency_steps` pushes ago.
        return self.buffer[-(self.latency_steps + 1)]


# -------------------------
# Minimal test snippets (to include in tests/test_latency.py)
# -------------------------
# These are small examples you can paste into a pytest file.
#
# import numpy as np
# from utilities.latency import LatencyBuffer
#
# def test_latency_basic():
#     buf = LatencyBuffer(latency_steps=2)
#     assert buf.pop() is None
#     buf.push(1)
#     assert buf.pop() is None
#     buf.push(2)
#     assert buf.pop() is None
#     buf.push(3)
#     # Now buffer has [1,2,3]; latency_steps=2 -> return buffer[-3] == 1
#     assert buf.pop() == 1
#
# def test_reset_behavior():
#     buf = LatencyBuffer(latency_steps=1)
#     buf.push('a')
#     buf.push('b')
#     assert buf.pop() == 'a'
#     buf.reset()
#     assert buf.pop() is None
#
# def test_zero_latency():
#     buf = LatencyBuffer(latency_steps=0)
#     assert buf.pop() is None
#     buf.push(42)
#     assert buf.pop() == 42
#
# Note: these tests intentionally use simple Python objects to avoid
# device/dtype dependencies; adapt to tensors if you prefer.
