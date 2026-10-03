"""Parametric track generation for multi-track / continual learning.

Only two closed tracks ship in tracks/ (oval, serpentine); continual and
generalization experiments need more geometry than that. These generators
produce deterministic road_definition dicts (and full project dicts via a
base project template) from seeds — including guaranteed-unseen holdout
tracks no training run has touched.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_TRACK = REPO_ROOT / "tracks" / "basic_driving_proving_ground.sim.json"


def load_base_project() -> dict[str, Any]:
    """The oval project — canonical template for generated tracks."""
    with open(BASE_TRACK, encoding="utf-8") as f:
        return json.load(f)


def _cp(x: float, y: float, width: float = 12.0) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": 0.0, "width": float(width),
            "banking": 0.0, "friction": 1.0}


def _spawn_at(cps: list[dict], i: int = 0) -> dict[str, float]:
    """Spawn on cp[i] facing cp[i+1] (track direction)."""
    p0, p1 = cps[i], cps[(i + 1) % len(cps)]
    yaw = math.atan2(p1["y"] - p0["y"], p1["x"] - p0["x"])
    return {"x": p0["x"], "y": p0["y"], "z": 0.1, "yaw": float(yaw),
            "initial_speed": 0.0}


def gen_oval(rx: float = 65.0, ry: float = 40.0, width: float = 12.0,
             n_cps: int = 8, name: str = "gen_oval") -> dict[str, Any]:
    """Elliptical closed loop."""
    cps = [_cp(rx * math.cos(2 * math.pi * i / n_cps),
              ry * math.sin(2 * math.pi * i / n_cps), width)
           for i in range(n_cps)]
    base = load_base_project()["road_definition"]
    return {
        "name": name, "is_closed": True, "control_points": cps,
        "boundary_config": copy.deepcopy(base["boundary_config"]),
        "spawn_point": _spawn_at(cps), "num_checkpoints": 16,
        "default_friction": 1.0,
    }


def gen_loop(seed: int, n_cps: int = 10, base_r: float = 55.0,
             radius_jitter: float = 18.0, width: float = 12.0,
             name: str = "gen_loop") -> dict[str, Any]:
    """Seeded perturbed-circle closed loop — unique geometry per seed.

    Deterministic: same seed -> same track. Radius per control point is
    smoothed (moving average) so loops stay drivable rather than spiky.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    radii = base_r + rng.uniform(-radius_jitter, radius_jitter, n_cps)
    # smooth radius profile to avoid self-intersecting geometry
    smoothed = np.convolve(np.r_[radii[-1], radii, radii[0]],
                           [0.25, 0.5, 0.25], mode="same")[1:-1]
    cps = [_cp(smoothed[i] * math.cos(2 * math.pi * i / n_cps),
              smoothed[i] * math.sin(2 * math.pi * i / n_cps), width)
           for i in range(n_cps)]
    base = load_base_project()["road_definition"]
    return {
        "name": name, "is_closed": True, "control_points": cps,
        "boundary_config": copy.deepcopy(base["boundary_config"]),
        "spawn_point": _spawn_at(cps), "num_checkpoints": 16,
        "default_friction": 1.0,
    }


def gen_straight(length: float = 100.0, width: float = 12.0,
                 name: str = "gen_straight") -> dict[str, Any]:
    """Short open straight segment — trivial drivable track for smoke
    tests (replaces the removed tracks/smoke_test.sim.json)."""
    cps = [_cp(0.0, 0.0, width), _cp(length, 0.0, width)]
    base = load_base_project()["road_definition"]
    return {
        "name": name, "is_closed": False, "control_points": cps,
        "boundary_config": copy.deepcopy(base["boundary_config"]),
        "spawn_point": {"x": 0.0, "y": 0.0, "z": 0.1, "yaw": 0.0,
                        "initial_speed": 0.0},
        "num_checkpoints": 4, "default_friction": 1.0,
    }


def project_for_road(road_def: dict[str, Any],
                     project_name: str | None = None) -> dict[str, Any]:
    """Wrap a generated road_definition in a full project dict."""
    proj = load_base_project()
    proj["road_definition"] = road_def
    if project_name:
        proj["name"] = project_name
    return proj
