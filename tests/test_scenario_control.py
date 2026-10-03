"""Phase 6 — SimulationEnvironment.set_scenario + TCP SET_SCENARIO (2.1)."""
import json
import socket
import threading
import time

import numpy as np
import pytest

from sim_env.scenario_designer import ScenarioDefinition
from sim_env.templates import EnvironmentTemplateManager
from sim_experiment.headless import build_env_from_dicts


def _project():
    return EnvironmentTemplateManager.create_project_from_template("lane_following")


def _scen_dict(name="basic_lane_following"):
    return ScenarioDefinition.get_standard_scenarios()[name].to_dict()


def _scenario_entities(env):
    return [e for e in env.entities
            if getattr(e, "_scenario_spawned", False)]


# ------------------------------------------------------------- env.set_scenario

def test_reset_does_not_accumulate_scenario_entities():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("obstacle_evasion"), seed=1)
    env.reset(seed=1)
    n1 = len(_scenario_entities(env))
    assert n1 == 2  # cone + barrier from obstacle_evasion
    env.reset(seed=2)
    n2 = len(_scenario_entities(env))
    assert n2 == n1  # no accumulation across resets


def test_set_scenario_clears_previous_obstacles():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("obstacle_evasion"), seed=1)
    env.reset(seed=1)
    assert len(_scenario_entities(env)) == 2
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("basic_lane_following")))
    env.reset(seed=1)
    assert len(_scenario_entities(env)) == 0


def test_set_scenario_applies_new_obstacles():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("basic_lane_following"), seed=1)
    env.reset(seed=1)
    assert len(_scenario_entities(env)) == 0
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("obstacle_evasion")))
    env.reset(seed=1)
    assert len(_scenario_entities(env)) == 2


def test_set_scenario_updates_surface_friction():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("basic_lane_following"), seed=1)
    env.reset(seed=1)
    f1 = env.active_surface_friction
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("wet_adverse_weather")))
    env.reset(seed=1)
    f2 = env.active_surface_friction
    assert f2 == pytest.approx(f1 * 0.65, rel=1e-6)


def test_set_scenario_deterministic_reset():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("basic_lane_following"), seed=1)
    env.reset(seed=1)
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("wet_adverse_weather")))
    o1, _ = env.reset(seed=77)
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("basic_lane_following")))
    env.reset(seed=3)
    env.set_scenario(ScenarioDefinition.from_dict(_scen_dict("wet_adverse_weather")))
    o2, _ = env.reset(seed=77)
    a1 = np.asarray(o1["vector"] if isinstance(o1, dict) else o1)
    a2 = np.asarray(o2["vector"] if isinstance(o2, dict) else o2)
    assert np.allclose(a1, a2)


# ------------------------------------------------------------- TCP SET_SCENARIO

def _start_server(env, port):
    from sim_net import SimulationServer
    server = SimulationServer(env, host="127.0.0.1", port=port)
    assert server.start() is True
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            server.poll_and_process()
            time.sleep(0.002)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.1)
    return server, stop, t


@pytest.fixture
def tcp_env():
    env = build_env_from_dicts(_project().to_dict(), _scen_dict("basic_lane_following"), seed=1)
    server, stop, t = _start_server(env, 9878)
    yield env
    stop.set()
    t.join(timeout=2.0)
    server.stop()


def _client():
    from sim_client import SimulationClient
    c = SimulationClient(host="127.0.0.1", port=9878)
    c.connect()
    return c


def test_tcp_scenario_update_success(tcp_env):
    c = _client()
    try:
        obs, info = c.set_scenario(_scen_dict("wet_adverse_weather"), seed=5, reset=True)
        assert obs is not None and len(obs) > 0
        assert tcp_env.scenario_def.name == "Wet Track Adverse Weather"
    finally:
        c.close()


def test_tcp_scenario_update_mid_episode_rejected(tcp_env):
    c = _client()
    try:
        c.reset(seed=1)
        c.step([0.0, 0.5, 0.0])
        with pytest.raises(RuntimeError, match="scenario_update_rejected|episode"):
            c.set_scenario(_scen_dict("wet_adverse_weather"), reset=False)
    finally:
        c.close()


def test_tcp_scenario_update_forced_mid_episode(tcp_env):
    c = _client()
    try:
        c.reset(seed=1)
        c.step([0.0, 0.5, 0.0])
        obs, info = c.set_scenario(_scen_dict("wet_adverse_weather"), seed=9, reset=True)
        assert tcp_env.current_step == 0  # reset applied
        assert tcp_env.scenario_def.name == "Wet Track Adverse Weather"
    finally:
        c.close()


def test_tcp_scenario_update_unknown_id_rejected(tcp_env):
    c = _client()
    try:
        with pytest.raises(RuntimeError, match="unknown_scenario"):
            c.set_scenario("nonexistent_scenario_xyz", reset=True)
    finally:
        c.close()


def test_tcp_scenario_update_invalid_dict_rejected(tcp_env):
    c = _client()
    try:
        bad = dict(_scen_dict("wet_adverse_weather"))
        bad["surface_friction_mult"] = "not_a_number"
        with pytest.raises(RuntimeError, match="invalid_scenario"):
            c.set_scenario(bad, reset=True)
    finally:
        c.close()


def test_tcp_scenario_update_deterministic_reset(tcp_env):
    c = _client()
    try:
        o1, _ = c.set_scenario(_scen_dict("wet_adverse_weather"), seed=42, reset=True)
        c.step([0.0, 0.5, 0.0])
        o2, _ = c.set_scenario(_scen_dict("wet_adverse_weather"), seed=42, reset=True)
        assert np.allclose(np.asarray(o1), np.asarray(o2))
    finally:
        c.close()


def test_tcp_scenario_update_protocol_2_0_rejected(tcp_env):
    """A client that only negotiates protocol 2.0 cannot SET_SCENARIO."""
    from sim_net.protocol import ProtocolEncoder, MessageType

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(("127.0.0.1", 9878))
    buf = ""

    def rpc(msg_type, payload):
        nonlocal buf
        sock.sendall(ProtocolEncoder.encode(msg_type, payload))
        while "\n" not in buf:
            buf += sock.recv(16384).decode("utf-8")
        line, buf = buf.split("\n", 1)
        return ProtocolEncoder.decode(line)

    try:
        mt, _ = rpc(MessageType.HANDSHAKE, {"protocol_versions": ["2.0"]})
        assert mt == MessageType.HANDSHAKE_ACK
        mt, payload = rpc("SET_SCENARIO", {"scenario_id": "wet_adverse_weather", "reset": True})
        assert mt == MessageType.ERROR
        assert "protocol" in str(payload).lower() or "unsupported" in str(payload).lower()
    finally:
        sock.close()


def test_tcp_set_scenario_capability_advertised(tcp_env):
    c = _client()
    try:
        contract = c.discover_contract()
        caps = contract.get("capabilities") or {}
        assert caps.get("scenario_update") is True
    finally:
        c.close()


# ----------------------------------------------------- curriculum over TCP E2E

def test_curriculum_advances_over_tcp(tmp_path):
    """Real curriculum run on a live headless TCP simulator: stage transition
    drives SET_SCENARIO through the wire and the run completes."""
    import os
    from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
    from sim_experiment.manager import ExperimentManager
    from sim_experiment.orchestrator import LocalTrainingOrchestrator
    from sim_env.curriculum import CurriculumDefinition, CurriculumStage

    project = _project()
    project.curriculum = CurriculumDefinition(name="tcp_curr", stages=[
        CurriculumStage(stage_id=1, name="s1", description="",
                        scenario_id="basic_lane_following",
                        target_metric="mean_return",
                        advancement_threshold=-1e9, min_episodes=1,
                        environment_overrides={"time_limit": 0.4}),
        CurriculumStage(stage_id=2, name="s2", description="",
                        scenario_id="wet_adverse_weather",
                        target_metric="mean_return",
                        advancement_threshold=1e18, min_episodes=1),
    ])
    manifest = ExperimentManifest.from_project(
        project=project, scenario=project.scenario_def,
        training=TrainingConfig(algorithm="ppo", total_timesteps=64,
                                rollout_length=32, batch_size=32, epochs=1,
                                eval_frequency=32, checkpoint_frequency=0,
                                num_envs=1),
        evaluation=EvaluationConfig(eval_seeds=[7], num_episodes=1),
        name="tcp_curr_test", random_seed=42,
    )
    root = str(tmp_path / "experiments")
    mgr = ExperimentManager(root_dir=root)
    exp_dir = mgr.create(manifest)
    orch = LocalTrainingOrchestrator(experiments_root=root)
    run_id = orch.launch(manifest, exp_dir, trainer="ppo", env_mode="tcp")
    summary = orch.wait(exp_dir, run_id, timeout_s=300)
    rd = os.path.join(exp_dir, "runs", run_id)
    if summary["status"] != "COMPLETED":
        err = ""
        for f in ("stdout.log", "stderr.log"):
            p = os.path.join(rd, "logs", f)
            if os.path.exists(p):
                err += open(p, encoding="utf-8", errors="replace").read()[-2000:]
        pytest.fail(f"tcp curriculum run failed: {summary.get('error')}\n{err}")

    state = json.load(open(os.path.join(rd, "curriculum_state.json")))
    assert state["stage_index"] == 1  # advanced past stage 0 via SET_SCENARIO
    assert state["history"][0]["advanced"] is True
