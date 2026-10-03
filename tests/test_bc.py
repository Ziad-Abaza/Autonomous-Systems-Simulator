"""Phase 6 — Behavior Cloning: data loading, training, checkpoints, eval."""
import json
import os

import numpy as np
import pytest


def _ep(eid, ret=1.0, length=3, obs_dim=4, act=0.5, reason="collision", seed=1):
    steps = []
    for t in range(length):
        steps.append({
            "obs": [0.1 * t + act] * obs_dim,
            "action": [act, 1.0 - act],
            "reward": ret / length,
            "terminated": (t == length - 1),
            "truncated": False,
            "termination_reason": reason if t == length - 1 else "",
        })
    return {
        "episode_id": eid, "seed": seed, "env_fingerprint": "fp_a",
        "scenario_id": "scen", "total_return": ret, "length": length,
        "termination_reason": reason, "steps": steps,
    }


def _write_dataset(ds_dir, episodes, manifest_extra=None):
    os.makedirs(ds_dir, exist_ok=True)
    with open(os.path.join(ds_dir, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    manifest = {
        "dataset_format": "transitions_v1",
        "schema_hash": "abc123",
        "env_fingerprint": "fp_a",
        "episodes": len(episodes),
        "steps": sum(e["length"] for e in episodes),
        "filters": {}, "skipped": {},
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(ds_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def _dataset(tmp_path, n=12, obs_dim=4, fp="fp_a"):
    ds = str(tmp_path / "ds")
    eps = [_ep(f"e{i}", obs_dim=obs_dim, act=0.5 + 0.01 * i, seed=i)
           for i in range(n)]
    for e in eps:
        e["env_fingerprint"] = fp
    _write_dataset(ds, eps)
    return ds


# ------------------------------------------------------------------- data loading

def test_load_transitions_shapes(tmp_path):
    from sim_experiment.bc.bc_data import load_transitions
    ds = _dataset(tmp_path, n=8, obs_dim=4)
    data = load_transitions(ds, split="train", seed=7, obs_dim=4,
                            expected_env_fingerprint="fp_a")
    assert data["obs"].shape[1] == 4
    assert data["actions"].shape[0] == data["obs"].shape[0]
    assert data["obs"].shape[0] > 0


def test_load_transitions_fingerprint_mismatch(tmp_path):
    from sim_experiment.bc.bc_data import load_transitions, BCDataError
    ds = _dataset(tmp_path)
    with pytest.raises(BCDataError, match="fingerprint"):
        load_transitions(ds, split="train", seed=1, obs_dim=4,
                         expected_env_fingerprint="fp_DIFFERENT")


def test_load_transitions_obs_dim_mismatch(tmp_path):
    from sim_experiment.bc.bc_data import load_transitions, BCDataError
    ds = _dataset(tmp_path, obs_dim=4)
    with pytest.raises(BCDataError, match="obs"):
        load_transitions(ds, split="train", seed=1, obs_dim=8)


def test_load_transitions_invalid_dataset(tmp_path):
    from sim_experiment.bc.bc_data import load_transitions, BCDataError
    ds = str(tmp_path / "bad")
    ep = _ep("e1")
    del ep["steps"][0]["action"]
    _write_dataset(ds, [ep])
    with pytest.raises(BCDataError):
        load_transitions(ds, split="train", seed=1, obs_dim=4)


# ------------------------------------------------------------------- BC training

def test_bc_runner_learns_constant_policy(tmp_path):
    """BC on a synthetic 'constant action' dataset reduces loss and saves ckpt."""
    from sim_experiment.bc.bc_model import BCPolicy  # noqa
    from sim_experiment.bc.bc_data import load_transitions
    from sim_experiment.bc.bc_runner import BCRunner

    ds = _dataset(tmp_path, n=16)
    data = load_transitions(ds, split="train", seed=1, obs_dim=4)
    val = load_transitions(ds, split="val", seed=1, obs_dim=4)

    runner = BCRunner(obs_dim=4, act_dim=2, action_mode="continuous",
                      hidden=(64, 64), lr=1e-2, batch_size=64, seed=3)
    m1 = runner.train_epochs(data, val, epochs=30)
    assert m1["final_train_loss"] < m1["initial_train_loss"]

    ck = str(tmp_path / "bc.pt")
    runner.save_checkpoint(ck, epoch=30)
    assert os.path.exists(ck)

    # Resume: model + optimizer + epoch restore
    r2 = BCRunner(obs_dim=4, act_dim=2, action_mode="continuous", seed=3)
    payload = r2.load_checkpoint(ck)
    assert payload["epoch"] == 30
    # Same params -> identical forward output on the same input
    x = np.zeros(4, dtype=np.float32)
    import torch
    with torch.no_grad():
        a1 = runner.policy(torch.tensor(x).unsqueeze(0))
        a2 = r2.policy(torch.tensor(x).unsqueeze(0))
    assert torch.allclose(a1, a2)


def test_bc_runner_deterministic_seed(tmp_path):
    from sim_experiment.bc.bc_data import load_transitions
    from sim_experiment.bc.bc_runner import BCRunner
    ds = _dataset(tmp_path, n=10)
    data = load_transitions(ds, split="train", seed=2, obs_dim=4)
    r1 = BCRunner(obs_dim=4, act_dim=2, action_mode="continuous", seed=9)
    m1 = r1.train_epochs(data, None, epochs=5)
    r2 = BCRunner(obs_dim=4, act_dim=2, action_mode="continuous", seed=9)
    m2 = r2.train_epochs(data, None, epochs=5)
    assert m1["final_train_loss"] == m2["final_train_loss"]


def test_bc_eval_policy(tmp_path):
    """BCPolicy acts as a deterministic evaluable policy."""
    from sim_experiment.bc.bc_runner import BCRunner
    runner = BCRunner(obs_dim=4, act_dim=2, action_mode="continuous", seed=1)
    act_fn = runner.eval_action_fn()
    a = act_fn(np.zeros(4, dtype=np.float32))
    assert np.asarray(a).shape == (2,)


# ------------------------------------------------------------ E2E via contract

def test_bc_trainer_end_to_end(tmp_path):
    """Full contract-driven BC run: dataset validation -> train -> ckpt -> eval."""
    import subprocess, sys
    from sim_env.templates import EnvironmentTemplateManager
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator

    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")

    # 1. Produce a real dataset from a short PPO run.
    m_rl = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=96,
                                rollout_length=48, num_envs=1,
                                eval_frequency=0, checkpoint_frequency=0,
                                algorithm_config={"trajectory_episodes": 4}),
        evaluation=EvaluationConfig(eval_seeds=[5], num_episodes=1),
        name="bc_source", random_seed=42,
    )
    exp_dir = mgr.create(m_rl)
    orch = LocalTrainingOrchestrator(experiments_root=root)
    run_id = orch.launch(m_rl, exp_dir, trainer="ppo")
    s = orch.wait(exp_dir, run_id, timeout_s=300)
    assert s["status"] == "COMPLETED", s.get("error")

    rl_dir = os.path.join(exp_dir, "runs", run_id)
    from sim_experiment.dataset import export_dataset
    ds_dir = os.path.join(str(tmp_path), "bc_dataset")
    with open(os.path.join(rl_dir, "contract.json")) as f:
        src_contract = json.load(f)
    export_dataset(rl_dir, ds_dir,
                   env_fingerprint=src_contract["environment_fingerprint"])

    # 2. BC run against that dataset via run override.
    m_bc = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="bc", total_timesteps=1,
                                num_envs=1, eval_frequency=0,
                                checkpoint_frequency=0,
                                algorithm_config={"bc_epochs": 5,
                                                  "batch_size": 64}),
        evaluation=EvaluationConfig(eval_seeds=[5], num_episodes=1),
        name="bc_run", random_seed=42,
    )
    exp_dir2 = mgr.create(m_bc)
    run_id2 = orch.launch(m_bc, exp_dir2, trainer="bc",
                          run_overrides={"bc_dataset_dir": ds_dir})
    s2 = orch.wait(exp_dir2, run_id2, timeout_s=300)
    rd2 = os.path.join(exp_dir2, "runs", run_id2)
    if s2["status"] != "COMPLETED":
        err = ""
        for f in ("stdout.log", "stderr.log"):
            p = os.path.join(rd2, "logs", f)
            if os.path.exists(p):
                err += open(p, encoding="utf-8", errors="replace").read()[-2000:]
        pytest.fail(f"bc run failed: {s2.get('error')}\n{err}")

    assert os.path.exists(os.path.join(rd2, "checkpoints", "policy_final.pt"))
    from sim_experiment.metrics import MetricsReader
    rows = MetricsReader(os.path.join(rd2, "metrics.jsonl")).read_all()
    assert any(r["scope"] == "bc" for r in rows)
    assert any(r["scope"] == "run" for r in rows)
