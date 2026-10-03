"""
Learning-convergence analysis for Phase 6 benchmarks.

Consumes a benchmark results dict (phases of frozen-policy evaluations on a
fixed seed set) and produces a verdict report:

    improvement vs baseline (delta mean_reward, delta completion_rate)
    cross-seed mean/std per phase
    stability — no phase regresses beyond the documented tolerance
    verdict: improved | no_improvement | regressed | unstable | insufficient_data

Thresholds are inputs, not magic constants — they come from the benchmark
config so the verdict is always reproducible from its declared criteria.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np


def _phase_mean_reward(phase: Dict[str, Any]) -> float:
    return float(phase.get("eval", {}).get("aggregate", {}).get("mean_reward", 0.0))


def _phase_completion(phase: Dict[str, Any]) -> float:
    return float(phase.get("eval", {}).get("aggregate", {}).get("completion_rate", 0.0))


def convergence_report(
    results: Dict[str, Any],
    min_reward_delta: Optional[float] = None,
    stability_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyzes a benchmark `results` dict:
        {"phases": [{"name", "eval": {"aggregate": {...}, "per_seed": {...}}}],
         "config": {"thresholds": {...}}}

    `min_reward_delta` / `stability_tolerance` override the config's
    thresholds when provided explicitly.
    """
    cfg = dict(results.get("config") or {})
    thresholds = dict(cfg.get("thresholds") or {})
    if min_reward_delta is None:
        min_reward_delta = float(thresholds.get("min_reward_delta", 0.0))
    if stability_tolerance is None:
        stability_tolerance = float(thresholds.get("stability_tolerance", 0.0))

    phases: List[Dict[str, Any]] = list(results.get("phases") or [])
    if len(phases) < 2:
        return {
            "verdict": "insufficient_data",
            "phases_analyzed": len(phases),
            "thresholds": {
                "min_reward_delta": min_reward_delta,
                "stability_tolerance": stability_tolerance,
            },
        }

    baseline = phases[0]
    final = phases[-1]
    baseline_r = _phase_mean_reward(baseline)
    final_r = _phase_mean_reward(final)

    improvement = {
        "delta_mean_reward": round(final_r - baseline_r, 4),
        "delta_completion_rate": round(
            _phase_completion(final) - _phase_completion(baseline), 4),
    }

    # Cross-seed stats on the final phase + every phase's seed spread.
    def seed_stats(phase):
        per_seed = list(phase.get("eval", {}).get("per_seed", {}).values())
        arr = np.asarray(per_seed, dtype=np.float64) if per_seed else np.zeros(0)
        return {
            "n_seeds": int(arr.shape[0]),
            "mean": round(float(arr.mean()), 4) if arr.size else 0.0,
            "std": round(float(arr.std()), 4) if arr.size else 0.0,
            "min": round(float(arr.min()), 4) if arr.size else 0.0,
            "max": round(float(arr.max()), 4) if arr.size else 0.0,
        }

    cross_seed = {
        "baseline": seed_stats(baseline),
        "final": seed_stats(final),
        "per_phase": {p["name"]: seed_stats(p) for p in phases},
    }

    # Stability: each post-baseline phase must not regress vs the previous
    # phase beyond the tolerance.
    regressions = []
    prev = baseline_r
    for ph in phases[1:]:
        r = _phase_mean_reward(ph)
        if r < prev - stability_tolerance:
            regressions.append({
                "phase": ph["name"],
                "mean_reward": r,
                "previous_mean_reward": prev,
                "drop": round(prev - r, 4),
            })
        prev = r
    stable = not regressions

    if final_r < baseline_r - stability_tolerance:
        verdict = "regressed"
    elif not stable:
        verdict = "unstable"
    elif improvement["delta_mean_reward"] >= min_reward_delta \
            and improvement["delta_mean_reward"] > 0:
        verdict = "improved"
    else:
        verdict = "no_improvement"

    return {
        "verdict": verdict,
        "stable": stable,
        "regressions": regressions,
        "improvement": improvement,
        "baseline_mean_reward": round(baseline_r, 4),
        "final_mean_reward": round(final_r, 4),
        "cross_seed": cross_seed,
        "phases_analyzed": len(phases),
        "thresholds": {
            "min_reward_delta": min_reward_delta,
            "stability_tolerance": stability_tolerance,
        },
    }
