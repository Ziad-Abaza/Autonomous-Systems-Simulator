"""
Integration test for External AI Protocol Contract Discovery (Phase 3N).
Verifies that external clients can dynamically discover environment contract,
schemas, action channels, observation dimensions, and sensor suites without hardcoding.
"""

import threading
import time
import pytest
from sim_env import SimulationEnvironment
from sim_net import SimulationServer
from sim_client import SimulationClient
from sim_env.agent import AgentDefinition


def test_contract_discovery_protocol():
    agent = AgentDefinition.create_default_vehicle_agent(agent_id="test_ai_agent")
    env = SimulationEnvironment(agent=agent)
    server = SimulationServer(env, host="127.0.0.1", port=9877)
    assert server.start() is True

    stop_event = threading.Event()

    def server_worker():
        while not stop_event.is_set():
            server.poll_and_process()
            time.sleep(0.005)

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()
    time.sleep(0.1)

    try:
        client = SimulationClient(host="127.0.0.1", port=9877)
        # Connect & discover — negotiated version is the max mutual
        # (2.1 since SET_SCENARIO landed; assert the contract field, not
        # a hardcoded literal).
        from sim_net.protocol import PROTOCOL_VERSION
        spec = client.connect()
        assert spec.get("protocol_version") == PROTOCOL_VERSION

        contract = client.discover_contract()
        assert contract.get("protocol_version") == PROTOCOL_VERSION
        assert contract["physics_hz"] == 60.0
        assert "action_schema" in contract
        assert "observation_schema" in contract
        assert "reward_schema" in contract
        assert "termination_schema" in contract
        assert "sensors" in contract

        # Verify discovery content
        obs_schema = contract["observation_schema"]
        assert obs_schema["vector_dimension"] == 23

        act_schema = contract["action_schema"]
        assert act_schema["num_channels"] == 3
        channel_names = [c["name"] for c in act_schema["channels"]]
        assert "steering" in channel_names
        assert "throttle" in channel_names
        assert "brake" in channel_names

        client.close()
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        server.stop()
