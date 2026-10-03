"""HANDSHAKE contract correctness — authored agents must expose their
compiled pipeline's effective spaces, not the legacy env defaults.

Regression for the audit finding: a TCP client connecting to an env whose
AgentDefinition has non-default obs/action shapes previously received the
legacy spaces (23-dim obs / [-1,1]x[0,1]x[0,1] action) regardless of the
actual authored contract.
"""
import threading
import time
import numpy as np
import pytest

from sim_env import SimulationEnvironment
from sim_env.agent import AgentDefinition
from sim_env.observation_designer import (
    ObservationSpaceDefinition, ObservationChannelConfig)
from sim_env.action_designer import ActionSpaceDefinition, ActionChannelConfig
from sim_net import SimulationServer
from sim_client import SimulationClient


def _serve(env, port):
    server = SimulationServer(env, host="127.0.0.1", port=port)
    assert server.start() is True
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            server.poll_and_process()
            time.sleep(0.003)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.1)
    return stop


def _custom_agent():
    """Authored agent with a deliberately non-default contract:
    6-dim obs vector and a 2-channel action space."""
    agent = AgentDefinition.create_default_vehicle_agent(agent_id="hs_agent")
    agent.observation_space = ObservationSpaceDefinition(channels=[
        ObservationChannelConfig(
            name=n, shape=[1], source_sensor="vehicle_state", source_key=src)
        for n, src in [
            ("speed", "speed"), ("yaw_rate", "yaw_rate"),
            ("distance_from_center", "distance_from_center"),
            ("heading_error", "heading_error"),
            ("distance_to_checkpoint", "distance_to_checkpoint"),
            ("steering_angle", "steering_angle"),
        ]
    ], flatten_vector=True)
    agent.action_space = ActionSpaceDefinition(
        space_type="continuous",
        channels=[
            ActionChannelConfig(name="steering", min_val=-0.5, max_val=0.5),
            ActionChannelConfig(name="throttle", min_val=0.0, max_val=0.8),
        ])
    return agent


def test_handshake_reports_authored_spaces():
    env = SimulationEnvironment(agent=_custom_agent())
    stop = _serve(env, 9891)
    try:
        client = SimulationClient(host="127.0.0.1", port=9891)
        spec = client.connect()
        assert spec["contract_source"] == "authored_agent"
        assert spec["vector_dim"] == 6
        assert spec["action_space"]["type"] == "continuous"
        assert spec["action_space"]["continuous_low"] == [-0.5, 0.0]
        assert spec["action_space"]["continuous_high"] == [0.5, 0.8]
        assert "authored_observation_schema" in spec
        # and the obs actually produced matches
        obs, _ = client.reset()
        assert len(obs) == 6
    finally:
        stop.set()


def test_handshake_no_agent_reports_legacy():
    env = SimulationEnvironment(agent=None)
    stop = _serve(env, 9892)
    try:
        client = SimulationClient(host="127.0.0.1", port=9892)
        spec = client.connect()
        assert spec["contract_source"] == "legacy"
        assert spec["vector_dim"] == env.observation_schema.compute_vector_dim()
    finally:
        stop.set()


def test_gym_env_space_matches_authored():
    """SimGymEnv must size its spaces from the authored contract."""
    env = SimulationEnvironment(agent=_custom_agent())
    stop = _serve(env, 9893)
    try:
        from sim_client.gym_env import SimGymEnv
        g = SimGymEnv(host="127.0.0.1", port=9893)
        assert g.action_space.shape == (2,)
        assert np.allclose(g.action_space.low, [-0.5, 0.0])
        assert np.allclose(g.action_space.high, [0.5, 0.8])
        assert g.observation_space.shape == (6,)
        obs, _ = g.reset()
        assert np.asarray(obs).shape == (6,)
        g.close()
    finally:
        stop.set()


def test_handshake_default_agent_full_space():
    """Default authored agent: spaces must equal the compiled pipeline's."""
    env = SimulationEnvironment(
        agent=AgentDefinition.create_default_vehicle_agent())
    stop = _serve(env, 9894)
    try:
        client = SimulationClient(host="127.0.0.1", port=9894)
        spec = client.connect()
        authored = env.agent.observation_space.export_schema()
        assert spec["vector_dim"] == authored["vector_dimension"] == 23
        obs, _ = client.reset()
        assert len(obs) == 23
        # obs space channels match authored names
        names = spec["observation_schema"]["channels"]
        assert names == [c["name"] for c in authored["channels"]]
    finally:
        stop.set()
