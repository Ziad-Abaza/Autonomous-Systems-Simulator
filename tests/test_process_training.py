"""Phase 6 — vector-env integration in PPO/SAC/DQN runners + contract."""
import numpy as np
import pytest

from sim_experiment.vec_env import SyncVectorEnv, is_vec_env
from sim_experiment.process_env import ProcessVectorEnv
from sim_experiment.trainer_contract import validate_contract, build_contract


def _scripted_worker(conn, env_dict, scenario_dict, seed):
    """Module-level worker for spawn pickling (duplicate of test_process_env's)."""
    class _Env:
        def __init__(self, spec, seed):
            self.term_at = spec.get("term_at", 4)
            self.obs_dim = spec.get("obs_dim", 8)
            self.t = 0

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.zeros(self.obs_dim, dtype=np.float32), {}

        def step(self, action):
            self.t += 1
            term = self.t >= self.term_at
            return (np.ones(self.obs_dim, dtype=np.float32) * self.t,
                    1.0, term, False,
                    {"step": self.t, "termination_reason": "collision" if term else ""})

    env = _Env(env_dict, seed)
    conn.send("ready")
    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            break
        if cmd == "close":
            break
        if cmd == "reset":
            conn.send(("ok", env.reset(seed=payload)[0]))
        elif cmd == "step":
            conn.send(("ok", env.step(payload)))
        elif cmd == "set_scenario":
            conn.send(("ok", None))
        elif cmd == "ping":
            conn.send(("ok", "pong"))
    conn.close()


class ScriptedEnv:
    def __init__(self, obs_dim=8, term_at=4, reward=1.0):
        self.term_at = term_at
        self.reward = reward
        self.obs_dim = obs_dim
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        term = self.t >= self.term_at
        return (np.ones(self.obs_dim, dtype=np.float32) * self.t,
                self.reward, term, False,
                {"step": self.t, "termination_reason": "collision" if term else "",
                 "lateral_offset": 0.0, "speed": 1.0,
                 "checkpoints_passed": 0, "is_colliding": term, "is_on_road": True})


def _vec(n=2, **kw):
    return SyncVectorEnv([ScriptedEnv(**kw) for _ in range(n)])


# ------------------------------------------------------------------ PPO vec

def test_ppo_runner_accepts_sync_vec_env():
    from sim_client.agents.ppo_baseline import PPORunner
    runner = PPORunner(env=_vec(2), num_steps=16, num_epochs=1,
                       batch_size=8, seed=11)
    metrics = runner.train(total_timesteps=32)
    assert metrics["total_timesteps"] == 32
    assert metrics["episodes_completed"] >= 2  # term_at=4 -> each env finishes eps


def test_ppo_runner_accepts_process_vec_env():
    from sim_client.agents.ppo_baseline import PPORunner
    pool = ProcessVectorEnv(env_dict={"term_at": 4}, scenario_dict=None,
                            seeds=[1, 2], worker_fn=_scripted_worker)
    try:
        runner = PPORunner(env=pool, num_steps=16, num_epochs=1,
                           batch_size=8, seed=11)
        metrics = runner.train(total_timesteps=32)
        assert metrics["total_timesteps"] == 32
        assert metrics["episodes_completed"] >= 2
    finally:
        pool.close()


def test_ppo_vec_set_envs_same_count():
    from sim_client.agents.ppo_baseline import PPORunner
    runner = PPORunner(env=_vec(2), num_steps=16, num_epochs=1,
                       batch_size=8, seed=11)
    runner.set_envs(_vec(2))
    assert runner._envs_dirty is True


def test_ppo_vec_set_envs_count_mismatch():
    from sim_client.agents.ppo_baseline import PPORunner
    runner = PPORunner(env=_vec(2), num_steps=16, num_epochs=1,
                       batch_size=8, seed=11)
    with pytest.raises(ValueError):
        runner.set_envs(_vec(3))


def test_ppo_vec_list_mode_rejects_vec_swap():
    from sim_client.agents.ppo_baseline import PPORunner
    runner = PPORunner(env=[ScriptedEnv()], num_steps=16, num_epochs=1,
                       batch_size=8, seed=11)
    with pytest.raises(ValueError):
        runner.set_envs(_vec(1))


def test_ppo_vec_deterministic_seeds():
    """Same seed -> identical metrics across two vec runs."""
    from sim_client.agents.ppo_baseline import PPORunner
    r1 = PPORunner(env=_vec(2), num_steps=16, num_epochs=1, batch_size=8, seed=5)
    m1 = r1.train(32)
    r2 = PPORunner(env=_vec(2), num_steps=16, num_epochs=1, batch_size=8, seed=5)
    m2 = r2.train(32)
    assert m1["episode_returns"] == m2["episode_returns"]


# ------------------------------------------------------------------ SAC/DQN vec

def test_sac_runner_accepts_vec_env():
    from sim_client.agents.sac_baseline import SACRunner
    vec = _vec(2, term_at=6)
    runner = SACRunner(env=vec, obs_dim=8, act_dim=3,
                       action_low=np.zeros(3), action_high=np.ones(3),
                       warmup_steps=4, batch_size=4, updates_per_step=1,
                       update_interval=8, seed=3)
    metrics = runner.train(total_timesteps=24)
    assert metrics["total_timesteps"] >= 24
    assert metrics["episodes_completed"] >= 1


def test_dqn_runner_accepts_vec_env():
    from sim_client.agents.dqn_baseline import DQNRunner
    vec = _vec(2, term_at=6)
    runner = DQNRunner(env=vec, obs_dim=8, num_actions=5,
                       warmup_steps=4, batch_size=4, train_freq=1,
                       update_interval=8, seed=3)
    metrics = runner.train(total_timesteps=24)
    assert metrics["total_timesteps"] >= 24
    assert metrics["episodes_completed"] >= 1


# ----------------------------------------------------------------- contract

def _minimal_contract(env_mode, curriculum=None):
    return {
        "contract_version": "1.0",
        "experiment_id": "exp_x", "run_id": "run_x", "seed": 1,
        "environment_fingerprint": "fp",
        "training": {"algorithm": "ppo", "num_envs": 2},
        "evaluation": {},
        "paths": {"run_dir": "r", "environment_json": "e",
                  "metrics_file": "m", "checkpoints_dir": "c"},
        "env_mode": env_mode,
        "curriculum": curriculum,
    }


def test_contract_accepts_process_mode():
    assert validate_contract(_minimal_contract("process")) == []


def test_contract_process_allows_curriculum():
    errors = validate_contract(_minimal_contract("process", curriculum={"stages": []}))
    assert not any("curriculum" in e for e in errors)


def test_contract_still_rejects_tcp_curriculum():
    contract = _minimal_contract("tcp", curriculum={"stages": []})
    contract["tcp"] = {"host": "127.0.0.1", "ports": [9000]}
    errors = validate_contract(contract)
    assert any("curriculum" in e for e in errors)


# ------------------------------------------------------- end-to-end process run

def test_ppo_process_mode_end_to_end(tmp_path):
    """Real contract-driven PPO run with env_mode='process' — orchestrator
    launches the trainer subprocess which spawns its own env workers."""
    from sim_env.templates import EnvironmentTemplateManager
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    from sim_experiment.run import RunStatus
    from sim_experiment.metrics import MetricsReader
    import os, json

    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    manifest = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(
            algorithm="ppo", total_timesteps=64, rollout_length=64,
            num_envs=2, eval_frequency=0, checkpoint_frequency=0,
        ),
        evaluation=EvaluationConfig(eval_seeds=[5], num_episodes=1),
        name="processtest", random_seed=42,
    )
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=root)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo", env_mode="process")
    summary = orch.wait(exp_dir, run_id, timeout_s=300)

    rd = os.path.join(exp_dir, "runs", run_id)
    if summary["status"] != RunStatus.COMPLETED:
        err = ""
        for f in ("stdout.log", "stderr.log"):
            p = os.path.join(rd, "logs", f)
            if os.path.exists(p):
                err += open(p).read()[-2000:]
        pytest.fail(f"run did not complete: {summary.get('error')}\n{err}")

    contract = json.load(open(os.path.join(rd, "contract.json")))
    assert contract["env_mode"] == "process"
    rows = MetricsReader(os.path.join(rd, "metrics.jsonl")).read_all()
    assert any(r["scope"] == "run" for r in rows)
