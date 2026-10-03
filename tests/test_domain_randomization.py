"""
Unit tests for domain randomization and seed reproducibility.
"""

import numpy as np
import pytest
from sim_env.domain_randomizer import DomainRandomizer, DomainRandomizationConfig


def test_seed_reproducibility():
    cfg = DomainRandomizationConfig(enabled=True)
    dr = DomainRandomizer(cfg)

    rng1 = np.random.default_rng(999)
    sample1 = dr.sample_parameters(rng1)

    rng2 = np.random.default_rng(999)
    sample2 = dr.sample_parameters(rng2)

    for k in sample1:
        assert sample1[k] == sample2[k], f"Key {k} did not match with same seed!"


def test_parameter_bounds():
    cfg = DomainRandomizationConfig(
        enabled=True,
        mass_range=(0.9, 1.1),
        tire_friction_range=(0.8, 1.2)
    )
    dr = DomainRandomizer(cfg)
    rng = np.random.default_rng(42)

    for _ in range(50):
        s = dr.sample_parameters(rng)
        assert 0.9 <= s['mass_factor'] <= 1.1
        assert 0.8 <= s['tire_friction_factor'] <= 1.2
