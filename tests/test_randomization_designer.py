"""
Unit tests for Domain Randomization Designer, distributions (Fixed, Uniform, Normal),
reproducibility with seeds, and bound enforcement.
"""

import pytest
import numpy as np

from sim_env.randomization_designer import (
    DomainRandomizationDefinition,
    RandomParamConfig,
    DistributionType
)


def test_distribution_types_and_bounds():
    rng = np.random.default_rng(12345)

    # 1. FIXED distribution
    p_fixed = RandomParamConfig("mass", DistributionType.FIXED, param1=1200.0)
    for _ in range(10):
        assert p_fixed.sample(rng) == 1200.0

    # 2. UNIFORM distribution with clipping
    p_uniform = RandomParamConfig("friction", DistributionType.UNIFORM, param1=0.5, param2=1.5, clip_min=0.6, clip_max=1.4)
    samples = [p_uniform.sample(rng) for _ in range(100)]
    assert all(0.6 <= s <= 1.4 for s in samples)
    assert min(samples) < 0.75
    assert max(samples) > 1.25

    # 3. NORMAL distribution
    p_normal = RandomParamConfig("jitter", DistributionType.NORMAL, param1=0.0, param2=1.0)
    normal_samples = [p_normal.sample(rng) for _ in range(500)]
    mean_val = np.mean(normal_samples)
    std_val = np.std(normal_samples)
    assert pytest.approx(mean_val, abs=0.15) == 0.0
    assert pytest.approx(std_val, abs=0.15) == 1.0


def test_seed_reproducibility():
    rand_def = DomainRandomizationDefinition.create_default()
    rand_def.enabled = True
    rand_def.global_seed = 42

    # Run 1
    rng1 = np.random.default_rng(rand_def.global_seed)
    sample1 = rand_def.sample_all(rng1)

    # Run 2 with identical seed
    rng2 = np.random.default_rng(rand_def.global_seed)
    sample2 = rand_def.sample_all(rng2)

    for k in sample1:
        assert sample1[k] == sample2[k]

    # Run 3 with different seed
    rng3 = np.random.default_rng(9999)
    sample3 = rand_def.sample_all(rng3)
    assert sample1 != sample3


def test_disabled_randomization_returns_identity():
    rand_def = DomainRandomizationDefinition.create_default()
    rand_def.enabled = False

    sample = rand_def.sample_all()
    assert sample["vehicle_mass_mult"] == 1.0
    assert sample["tire_friction_mult"] == 1.0
    assert sample["surface_friction_mult"] == 1.0
    assert sample["spawn_lateral_jitter_m"] == 0.0
