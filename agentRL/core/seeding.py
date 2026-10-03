"""Deterministic seed derivation. One base seed -> independent named streams."""
from __future__ import annotations

import random

import numpy as np

_STREAMS = ("env", "track", "torch", "mutator", "eval", "worker")


def seed_tree(base_seed: int) -> dict[str, int]:
    """Derive independent child seeds from one base seed.

    Deterministic: same base_seed -> same tree. Streams are drawn from
    SeedSequence.spawn so they are statistically independent.
    """
    seq = np.random.SeedSequence(int(base_seed))
    children = seq.spawn(len(_STREAMS))
    return {name: int(child.generate_state(1)[0])
            for name, child in zip(_STREAMS, children)}


def set_global_seeds(seed: int) -> None:
    """Seed python/numpy/torch global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch always present here
        pass
