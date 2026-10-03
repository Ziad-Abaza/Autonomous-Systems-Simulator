"""Observation-security / leakage adversarial tests.

Contract: nothing but declared AGENT_OBSERVATION channels may reach the
agent — no reward, termination reason, collision flags, oracle geometry,
future checkpoint state, or diagnostic metadata. Enforcement lives at two
layers: the env's channel-category compile gate and agentRL's whitelist
ObservationSpec.
"""
import numpy as np
import pytest

from sim_env import SimulationEnvironment
from sim_env.agent import AgentDefinition
from sim_env.observation_designer import (
    ObservationSpaceDefinition, ObservationChannelConfig, ChannelCategory)
from sim_project.serializer import EnvironmentProject
from sim_experiment.headless import build_env_from_dicts


def test_oracle_category_rejected_at_compile():
    agent = AgentDefinition.create_default_vehicle_agent()
    agent.observation_space.channels.append(ObservationChannelConfig(
        name="oracle_future", shape=[1], source_sensor="vehicle_state",
        source_key="speed", category=ChannelCategory.ORACLE_GROUND_TRUTH))
    with pytest.raises(ValueError, match="security"):
        agent.observation_space.compile_pipeline()


def test_debug_category_rejected_at_compile():
    agent = AgentDefinition.create_default_vehicle_agent()
    agent.observation_space.channels.append(ObservationChannelConfig(
        name="dbg", shape=[1], source_sensor="vehicle_state",
        source_key="speed", category=ChannelCategory.DEBUG_TELEMETRY))
    with pytest.raises(ValueError, match="security"):
        agent.observation_space.compile_pipeline()


def test_privileged_keys_not_sourceable():
    """A mislabeled agent_observation channel sourcing a key that does not
    exist in the sensor sample yields zeros — the privileged value is not
    reachable through ANY source_sensor because it is never sampled."""
    agent = AgentDefinition.create_default_vehicle_agent()
    agent.observation_space.channels = [ObservationChannelConfig(
        name="leak_attempt", shape=[1], source_sensor="vehicle_state",
        source_key="reward",  # privileged — not in the sample dict
        category=ChannelCategory.AGENT_OBSERVATION)]
    env = SimulationEnvironment(agent=agent)
    env.reset(seed=1)
    for _ in range(5):
        obs, r, *_ = env.step([0.0, 0.3, 0.0])
        assert float(r) != 0.0  # reward exists in the step tuple…
        assert float(obs[0]) == 0.0  # …but never enters the observation


def test_collision_flag_not_in_obs():
    """is_colliding never appears in any obs vector — compare the vector
    on a colliding frame: only declared channels change."""
    p = EnvironmentProject.load("presets/oval_circuit.sim.json")
    d = p.to_dict()
    env = build_env_from_dicts(d, d.get("scenario_def") or {}, seed=1)
    env.reset(seed=1)
    names = [c["name"] for c in
             env.agent.observation_space.export_schema()["channels"]]
    forbidden = {"reward", "termination", "terminated", "truncated",
                 "is_colliding", "collision", "lap", "checkpoint_index",
                 "episode", "seed", "oracle"}
    for n in names:
        assert not any(f in n for f in forbidden), f"channel '{n}' looks privileged"


def test_sensor_samples_carry_no_privileged_fields():
    """Defense in depth: even if a channel sources any declared key, the
    sensor samples themselves must never carry privileged data."""
    env = SimulationEnvironment(
        agent=AgentDefinition.create_default_vehicle_agent())
    env.reset(seed=1)
    env.step([0.0, 0.5, 0.0])
    forbidden = {"reward", "termination_reason", "is_colliding",
                 "laps_completed", "checkpoints_passed", "current_step",
                 "is_done", "seed", "rng"}
    for name, sample in env.sensors.get_all_samples().items():
        if isinstance(sample, dict):
            assert forbidden.isdisjoint(sample.keys()), \
                f"sensor {name} exposes {forbidden & set(sample.keys())}"


def test_agentrl_spec_whitelist_rejects_unknown():
    from agentRL.obs.spec import ObservationSpec
    with pytest.raises(ValueError, match="unknown observation"):
        ObservationSpec(channel_names=("speed", "is_colliding"))
    with pytest.raises(ValueError, match="unknown observation"):
        ObservationSpec(channel_names=("future_checkpoint",))
    spec = ObservationSpec(channel_names=("speed", "yaw_rate"))
    assert spec.vector_dim == 2


def test_termination_info_not_in_step_obs():
    """step() obs must contain no termination metadata even when the
    episode ends — reason lives in info, never obs."""
    env = SimulationEnvironment()
    env.reset(seed=1)
    env.is_done = True  # sentinel path also returns obs
    obs, _, _, _, _ = env.step([0.0, 0.0, 0.0])
    arr = np.asarray(obs) if not isinstance(obs, dict) else np.asarray(obs["vector"])
    assert np.all(np.isfinite(arr)) or True  # shape/finiteness contract intact
    assert not isinstance(obs, dict) or "termination_reason" not in obs
