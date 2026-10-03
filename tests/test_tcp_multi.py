"""Phase 6 — SimServerMulti: one process, N envs, one port, N clients."""
import threading
import time

import numpy as np
import pytest

from sim_env import SimulationEnvironment


def _make_env(seed=1):
    return SimulationEnvironment(seed=seed)


@pytest.fixture
def multi_server():
    from sim_net.multi_server import SimServerMulti
    srv = SimServerMulti(env_factories=[lambda: _make_env(1), lambda: _make_env(2)],
                         host="127.0.0.1", port=9881)
    assert srv.start() is True
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            srv.poll_and_process()
            time.sleep(0.001)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.15)
    yield srv
    stop.set()
    t.join(timeout=2.0)
    srv.stop()


def _client():
    from sim_client.gym_env import SimGymEnv
    return SimGymEnv(host="127.0.0.1", port=9881)


def test_multi_two_clients_independent(multi_server):
    c1, c2 = _client(), _client()
    try:
        o1, _ = c1.reset(seed=10)
        o2, _ = c2.reset(seed=20)
        # step client1 only — client2 must remain at step 0
        for _ in range(5):
            c1.step([0.0, 0.5, 0.0])
        assert multi_server.clients[1].env.current_step == 0
        assert multi_server.clients[0].env.current_step == 5
    finally:
        c1.close()
        c2.close()


def test_multi_scenario_isolated_per_client(multi_server):
    from sim_env.scenario_designer import ScenarioDefinition
    wet = ScenarioDefinition.get_standard_scenarios()["wet_adverse_weather"].to_dict()
    c1, c2 = _client(), _client()
    try:
        c1.set_scenario(wet, seed=1, reset=True)
        c2.reset(seed=1)
        assert multi_server.clients[0].env.scenario_def.name == "Wet Track Adverse Weather"
        e2 = multi_server.clients[1].env
        assert e2.scenario_def is None or e2.scenario_def.name != "Wet Track Adverse Weather"
    finally:
        c1.close()
        c2.close()


def test_multi_client_slot_reclaimed_on_disconnect(multi_server):
    c1 = _client()
    c1.reset(seed=1)
    c1.close()
    time.sleep(0.2)
    # a fresh client can connect and gets a working env slot
    c2 = _client()
    try:
        o, info = c2.reset(seed=3)
        assert len(o) > 0
    finally:
        c2.close()


def test_multi_extra_client_rejected(multi_server):
    c1, c2 = _client(), _client()
    c1.reset(seed=1)
    c2.reset(seed=2)
    # Server is full — the third client is rejected at connect/handshake.
    with pytest.raises((RuntimeError, ConnectionError)):
        _client()
    c1.close()
    c2.close()


# ------------------------------------------------------------ env_mode=tcp_multi

def test_tcp_multi_end_to_end(tmp_path):
    """Orchestrator launches one sim process hosting 2 envs; PPO trains."""
    import os, json
    from sim_env.templates import EnvironmentTemplateManager
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    from sim_experiment.metrics import MetricsReader

    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    project = EnvironmentTemplateManager.create_project_from_template("lane_following")
    manifest = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(
            algorithm="ppo", total_timesteps=64, rollout_length=32,
            num_envs=2, eval_frequency=0, checkpoint_frequency=0,
        ),
        evaluation=EvaluationConfig(eval_seeds=[5], num_episodes=1),
        name="tcpmulti", random_seed=42,
    )
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=root)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo", env_mode="tcp_multi")
    summary = orch.wait(exp_dir, run_id, timeout_s=300)
    rd = os.path.join(exp_dir, "runs", run_id)
    if summary["status"] != "COMPLETED":
        err = ""
        for f in ("stdout.log", "stderr.log"):
            p = os.path.join(rd, "logs", f)
            if os.path.exists(p):
                err += open(p, encoding="utf-8", errors="replace").read()[-2000:]
        pytest.fail(f"tcp_multi run failed: {summary.get('error')}\n{err}")

    contract = json.load(open(os.path.join(rd, "contract.json")))
    assert contract["env_mode"] == "tcp_multi"
    rows = MetricsReader(os.path.join(rd, "metrics.jsonl")).read_all()
    assert any(r["scope"] == "run" for r in rows)
