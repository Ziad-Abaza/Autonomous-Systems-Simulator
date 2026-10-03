"""Experiment matrix tests — config validation + dry-run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "agentRL" / \
    "experiments" / "configs"


def test_experiment_config_loads():
    from agentRL.envs.track_registry import TrackRegistry
    from agentRL.experiments.matrix import load_config, EXPERIMENT_IDS

    reg = TrackRegistry.default()
    known_tracks = set(reg.list())
    configs = list(CONFIGS_DIR.glob("*.json"))
    assert len(configs) >= 8, "expected >=8 experiment configs"
    for path in configs:
        cfg = load_config(path)
        assert cfg["id"].startswith("E")
        tracks = set()
        if "track" in cfg:
            tracks.add(cfg["track"])
        for ph in cfg.get("phases", []):
            tracks.add(ph["track_id"])
        for t in cfg.get("tracks", []):
            tracks.add(t)
        for t in cfg.get("holdouts", []):
            tracks.add(t)
        assert tracks <= known_tracks, \
            f"{cfg['id']}: unknown tracks {tracks - known_tracks}"
        algos = set(cfg.get("algos", [cfg.get("algo")]))
        assert algos <= {"sac", "ppo"}, f"{cfg['id']}: bad algos {algos}"
    assert set(EXPERIMENT_IDS) >= {"E001", "E002", "E003", "E004", "E005"}


def test_track_sampler_rotates():
    from agentRL.envs.factory import EnvFactory
    from agentRL.envs.track_registry import TrackRegistry
    from agentRL.train.trainer import TrackSampler

    reg = TrackRegistry.default()
    sampler = TrackSampler(EnvFactory(),
                           [reg.load("smoke"), reg.load("oval")],
                           seed=0)
    ids = {sampler.next()._sampler_track for _ in range(20)}
    assert ids == {"smoke", "oval"}
    # same env instance reused per track (cached)
    e1 = sampler.next()
    seen = set()
    for _ in range(20):
        e = sampler.next()
        seen.add(id(e))
    assert len(seen) == 2


def test_e001_dry_run(tmp_path):
    from agentRL.experiments.matrix import run_experiment

    out = run_experiment("E001", run_dir=str(tmp_path / "e001"),
                         steps_override=400, eval_episodes=1,
                         algos=["sac"])
    run_dir = tmp_path / "e001" / "sac"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert "sac" in out
