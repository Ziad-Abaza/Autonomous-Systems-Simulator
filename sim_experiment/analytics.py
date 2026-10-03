"""
Experiment analytics — metrics extraction, smoothing, run comparison.

Reads the existing metrics.jsonl artifact (no new storage, no second
metrics system) and produces chart-ready series: raw + smoothed values
aligned by timestep. Comparison across runs or experiments is a pure
read operation; experiment and run data stay immutable.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple

Series = List[Tuple[int, float]]


def iter_metrics(run_dir: str, scope: Optional[str] = None):
    """Yields (timestep, metrics) pairs from a run's metrics.jsonl."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if scope is None or rec.get("scope") == scope:
                yield int(rec.get("timestep", 0)), rec.get("metrics", {})


def load_metrics_series(run_dir: str, scope: str, metric: str) -> Series:
    """Sorted (timestep, value) series for one metric in one scope."""
    series = []
    for ts, metrics in iter_metrics(run_dir, scope):
        v = metrics.get(metric)
        if isinstance(v, (int, float)):
            series.append((ts, float(v)))
    return sorted(series, key=lambda kv: kv[0])


def smooth(series: Series, window: int = 10) -> Series:
    """Centered-free trailing moving average preserving timesteps."""
    if window <= 1 or len(series) <= 1:
        return list(series)
    out: Series = []
    acc = 0.0
    buf: List[float] = []
    for ts, v in series:
        buf.append(v)
        acc += v
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append((ts, acc / len(buf)))
    return out


def series_stats(series: Series) -> Dict[str, float]:
    if not series:
        return {"max": 0.0, "final": 0.0, "mean": 0.0, "count": 0}
    vals = [v for _, v in series]
    return {
        "max": max(vals),
        "min": min(vals),
        "final": vals[-1],
        "mean": sum(vals) / len(vals),
        "count": len(vals),
    }


def compare_runs(
    run_dirs: List[str],
    metric: str,
    scope: str = "episode",
    smooth_window: int = 10,
) -> Dict[str, Any]:
    """
    Chart-ready comparison of the same metric across runs:
    [{label, run_id, raw, smoothed, max, min, final, mean, count}].
    """
    entries = []
    for rd in run_dirs:
        raw = load_metrics_series(rd, scope, metric)
        if not raw:
            continue
        stats = series_stats(raw)
        entries.append({
            "label": os.path.basename(rd.rstrip("/\\")),
            "run_id": os.path.basename(rd.rstrip("/\\")),
            "run_dir": os.path.abspath(rd),
            "raw": [[t, v] for t, v in raw],
            "smoothed": [[t, v] for t, v in smooth(raw, smooth_window)],
            **stats,
        })
    return {"metric": metric, "scope": scope,
            "smooth_window": smooth_window, "series": entries}


def compare_experiments(
    experiment_dirs: List[str],
    metric: str,
    scope: str = "episode",
    smooth_window: int = 10,
) -> Dict[str, Any]:
    """Compares a metric across ALL runs of each experiment."""
    run_dirs = []
    labels = []
    for exp_dir in experiment_dirs:
        runs_dir = os.path.join(exp_dir, "runs")
        if not os.path.isdir(runs_dir):
            continue
        exp_label = os.path.basename(exp_dir.rstrip("/\\"))
        for rid in sorted(os.listdir(runs_dir)):
            rd = os.path.join(runs_dir, rid)
            if os.path.isdir(rd):
                run_dirs.append(rd)
                labels.append(f"{exp_label}/{rid}")
    result = compare_runs(run_dirs, metric, scope, smooth_window)
    for entry, label in zip(result["series"], labels):
        entry["label"] = label
    return result


def metric_summary(run_dir: str) -> Dict[str, Any]:
    """
    One-line run summary: final/best metrics, timesteps, throughput,
    termination reason histogram — for comparison tables.
    """
    ep_rewards = load_metrics_series(run_dir, "episode", "reward")
    ep_lengths = load_metrics_series(run_dir, "episode", "length")
    eval_reward = load_metrics_series(run_dir, "evaluation", "mean_reward")
    eval_completion = load_metrics_series(run_dir, "evaluation", "completion_rate")

    reasons: Dict[str, int] = {}
    for _, metrics in iter_metrics(run_dir, "episode"):
        r = metrics.get("termination_reason")
        if r:
            reasons[r] = reasons.get(r, 0) + 1

    run_metrics: Dict[str, Any] = {}
    for _, metrics in iter_metrics(run_dir, "run"):
        run_metrics.update(metrics)

    # run.json is the authoritative source for status/timesteps
    run_status: Dict[str, Any] = {}
    run_json = os.path.join(run_dir, "run.json")
    if os.path.exists(run_json):
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                run_status = json.load(f)
        except json.JSONDecodeError:
            pass

    ep_stats = series_stats(ep_rewards)
    return {
        "run_id": os.path.basename(run_dir.rstrip("/\\")),
        "episodes": int(ep_stats.get("count", 0)),
        "mean_return": round(ep_stats.get("mean", 0.0), 4),
        "best_return": round(ep_stats.get("max", 0.0), 4),
        "final_return": round(ep_stats.get("final", 0.0), 4),
        "mean_length": round(series_stats(ep_lengths).get("mean", 0.0), 2),
        "best_eval_mean_reward": round(series_stats(eval_reward).get("max", 0.0), 4),
        "final_eval_mean_reward": round(series_stats(eval_reward).get("final", 0.0), 4),
        "final_eval_completion": round(series_stats(eval_completion).get("final", 0.0), 4),
        "total_timesteps": int(run_status.get("current_timestep",
                                             run_metrics.get("total_timesteps", 0))),
        "status": run_status.get("status", ""),
        "wall_clock_time": run_metrics.get("wall_clock_time", run_metrics.get("sps", 0.0)),
        "sps": run_metrics.get("sps", 0.0),
        "termination_reasons": reasons,
    }


def list_run_dirs(experiment_dir: str) -> List[str]:
    runs_dir = os.path.join(experiment_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    return [os.path.join(runs_dir, d) for d in sorted(os.listdir(runs_dir))
            if os.path.isdir(os.path.join(runs_dir, d))]
