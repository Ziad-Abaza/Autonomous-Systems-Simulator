"""Core scaffolding tests: versioning, seeding, config."""
from __future__ import annotations

import numpy as np


def test_version_constants_exist():
    from agentRL.core import versioning

    assert isinstance(versioning.AGENT_VERSION, str)
    assert versioning.AGENT_VERSION.count(".") == 2
    assert isinstance(versioning.CKPT_SCHEMA_V, int) and versioning.CKPT_SCHEMA_V >= 1
    assert isinstance(versioning.OBS_SPEC_V, int)
    assert isinstance(versioning.ACT_SPEC_V, int)


def test_seed_tree_deterministic():
    from agentRL.core.seeding import seed_tree

    a = seed_tree(42)
    b = seed_tree(42)
    c = seed_tree(43)
    assert a == b, "same base seed must produce identical tree"
    assert a != c, "different base seed must produce different tree"
    for key in ("env", "track", "torch", "mutator", "eval"):
        assert key in a, f"missing seed stream {key}"
        assert isinstance(a[key], int)


def test_set_global_seeds_reproducible():
    from agentRL.core.seeding import set_global_seeds

    set_global_seeds(123)
    first = np.random.random(4).copy()
    set_global_seeds(123)
    second = np.random.random(4)
    assert np.array_equal(first, second)


def test_config_dataclasses():
    from agentRL.core.config import AgentConfig, TrainConfig

    ac = AgentConfig(algo_id="sac", hidden_sizes=(256, 256), lr=3e-4, gamma=0.99)
    assert ac.algo_id == "sac"
    tc = TrainConfig(total_steps=1000, eval_interval=500, ckpt_interval=500,
                     num_envs=1, seed=42, run_dir="agentRL/runs/test")
    assert tc.total_steps == 1000 and tc.num_envs == 1
