"""ScenarioMutator — seeded randomized scenario generation.

Produces per-episode `ScenarioDefinition`s with obstacles placed along the
track's own centerline (via env.track.spline), spawn jitter, and surface
friction / sensor-noise multipliers. Placement is scenario-level
configuration, not policy input — the agent only ever sees obstacles
through its sensors (lidar), never memorized positions.

Deterministic: a mutator with seed S draws the same sequence of scenarios
on every fresh instance; successive draws differ (per-episode variety).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim_env.scenario_designer import ScenarioDefinition


class ScenarioMutator:
    """Seeded per-episode scenario generator bound to no specific track."""

    def __init__(
        self,
        seed: int,
        n_obstacles: tuple[int, int] = (0, 3),
        entity_types: tuple[str, ...] = ("cone", "barrier", "obstacle"),
        lateral_frac: tuple[float, float] = (-0.4, 0.4),
        s_range: tuple[float, float] = (0.15, 0.95),
        min_gap_m: float = 15.0,
        spawn_jitter: tuple[float, float] | None = (-1.0, 1.0),
        spawn_yaw_jitter_deg: float = 15.0,
        friction: tuple[float, float] = (0.85, 1.1),
        noise: tuple[float, float] = (0.8, 1.5),
        scenario_id: str = "agentrl_mutated",
    ) -> None:
        self._rng = np.random.default_rng(int(seed))
        self.n_obstacles = n_obstacles
        self.entity_types = entity_types
        self.lateral_frac = lateral_frac
        self.s_range = s_range
        self.min_gap_m = float(min_gap_m)
        self.spawn_jitter = spawn_jitter
        self.spawn_yaw_jitter_deg = float(spawn_yaw_jitter_deg)
        self.friction = friction
        self.noise = noise
        self.scenario_id = scenario_id
        self._draw_count = 0
        self._params = dict(
            n_obstacles=n_obstacles, entity_types=entity_types,
            lateral_frac=lateral_frac, s_range=s_range,
            min_gap_m=min_gap_m, spawn_jitter=spawn_jitter,
            spawn_yaw_jitter_deg=spawn_yaw_jitter_deg,
            friction=friction, noise=noise, scenario_id=scenario_id)

    def fork(self, seed: int) -> "ScenarioMutator":
        """Independent mutator with the same params and a derived seed —
        use per eval cell / per env so draws aren't order-coupled."""
        return ScenarioMutator(seed=int(seed), **self._params)

    # ------------------------------------------------------------------ draw

    def draw(self, env) -> ScenarioDefinition:
        """Draw a fresh scenario against env's track geometry."""
        rng = self._rng
        spline = env.track.spline
        L = float(spline.total_length)
        closed = bool(spline.is_closed)

        n = int(rng.integers(self.n_obstacles[0], self.n_obstacles[1] + 1)) \
            if self.n_obstacles[1] > 0 else 0
        obstacles: list[dict[str, Any]] = []
        chosen_s: list[float] = []
        for i in range(n):
            s = None
            for _ in range(10):
                cand = float(rng.uniform(self.s_range[0], self.s_range[1])) * L
                if all(self._gap(cand, prev, L, closed) >= self.min_gap_m
                       for prev in chosen_s):
                    s = cand
                    break
            if s is None:
                continue  # could not place with min_gap — skip obstacle
            chosen_s.append(s)
            sp = spline.sample_at_distance(s)
            lat = float(rng.uniform(*self.lateral_frac)) * (sp.width * 0.5)
            obstacles.append({
                "name": f"mut_{self.scenario_id}_{i}",
                "entity_type": str(rng.choice(self.entity_types)),
                "pos": [sp.pos.x + sp.normal.x * lat,
                        sp.pos.y + sp.normal.y * lat, 0.2],
                "yaw": float(rng.uniform(0.0, 2.0 * math.pi)),
            })

        spawn_override = self._draw_spawn(env, spline, rng)

        self._draw_count += 1
        return ScenarioDefinition(
            scenario_id=f"{self.scenario_id}_{self._draw_count}",
            name="agentRL Mutated Scenario",
            surface_friction_mult=float(rng.uniform(*self.friction)),
            sensor_noise_mult=float(rng.uniform(*self.noise)),
            spawn_override=spawn_override,
            obstacle_overrides=obstacles,
        )

    def _draw_spawn(self, env, spline, rng) -> dict[str, Any] | None:
        if self.spawn_jitter is None:
            return None
        sp_base = env.road_def.spawn_point
        from sim_core.math_utils import Vec2
        _, _, _, sample = spline.get_closest_point(
            Vec2(sp_base.x, sp_base.y))
        half = sample.width * 0.5
        lat = float(np.clip(rng.uniform(*self.spawn_jitter),
                            -half * 0.3, half * 0.3))
        yaw_deg = math.degrees(sp_base.yaw) + float(
            rng.uniform(-self.spawn_yaw_jitter_deg,
                        self.spawn_yaw_jitter_deg))
        return {
            "pos": [sp_base.x + sample.normal.x * lat,
                    sp_base.y + sample.normal.y * lat, 0.2],
            "yaw_deg": float(yaw_deg),
        }

    @staticmethod
    def _gap(a: float, b: float, L: float, closed: bool) -> float:
        d = abs(a - b)
        return min(d, L - d) if closed else d

    # ----------------------------------------------------------------- apply

    def draw_and_apply(self, env) -> ScenarioDefinition:
        """Draw a scenario and install it for the env's next reset."""
        scenario = self.draw(env)
        env.set_scenario(scenario)
        return scenario

    @staticmethod
    def apply(env, scenario: ScenarioDefinition) -> None:
        env.set_scenario(scenario)
