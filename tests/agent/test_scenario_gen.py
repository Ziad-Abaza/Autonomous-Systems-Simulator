"""ScenarioMutator tests — seeded randomized obstacle/spawn scenarios."""
from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.fixture()
def oval_env():
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry

    return EnvFactory().build(TrackRegistry.default().load("oval"), seed=42)


def test_deterministic(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    m = ScenarioMutator(seed=7, n_obstacles=(3, 3))
    a = m.draw(oval_env)
    m2 = ScenarioMutator(seed=7, n_obstacles=(3, 3))
    b = m2.draw(oval_env)
    assert a.to_dict() == b.to_dict()


def test_different_seeds_differ(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    a = ScenarioMutator(seed=1, n_obstacles=(3, 3)).draw(oval_env)
    b = ScenarioMutator(seed=2, n_obstacles=(3, 3)).draw(oval_env)
    assert a.to_dict() != b.to_dict()


def test_obstacles_on_road(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    m = ScenarioMutator(seed=11, n_obstacles=(5, 5))
    sc = m.draw(oval_env)
    assert len(sc.obstacle_overrides) == 5
    spline = oval_env.track.spline
    for ob in sc.obstacle_overrides:
        from sim_core.math_utils import Vec2
        s, lat, _, sample = spline.get_closest_point(
            Vec2(ob["pos"][0], ob["pos"][1]))
        half = sample.width * 0.5
        assert abs(lat) < half * 0.8, (
            f"obstacle at lateral {lat:.2f} outside road half-width {half:.2f}"
        )


def test_apply_spawns_entities(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    m = ScenarioMutator(seed=5, n_obstacles=(2, 2))
    m.apply(oval_env, m.draw(oval_env))
    oval_env.reset(seed=1)
    spawned = [e for e in oval_env.entities
               if getattr(e, "_scenario_spawned", False)]
    assert len(spawned) == 2


def test_min_gap_respected(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    m = ScenarioMutator(seed=3, n_obstacles=(4, 4), min_gap_m=15.0)
    sc = m.draw(oval_env)
    spline = oval_env.track.spline
    ss = []
    from sim_core.math_utils import Vec2
    for ob in sc.obstacle_overrides:
        s, _, _, _ = spline.get_closest_point(Vec2(ob["pos"][0], ob["pos"][1]))
        ss.append(s)
    ss.sort()
    L = spline.total_length
    for i in range(len(ss)):
        gap = (ss[(i + 1) % len(ss)] - ss[i]) % L
        assert gap >= 14.0, f"obstacle gap {gap:.1f} < min_gap"


def test_spawn_and_friction(oval_env):
    from agentRL.envs.scenario_gen import ScenarioMutator

    m = ScenarioMutator(seed=9, n_obstacles=(0, 0),
                        friction=(0.9, 0.9), spawn_jitter=(-1.0, 1.0))
    sc = m.draw(oval_env)
    assert sc.surface_friction_mult == pytest.approx(0.9)
    if sc.spawn_override is not None:
        assert "pos" in sc.spawn_override
        assert "yaw_deg" in sc.spawn_override
