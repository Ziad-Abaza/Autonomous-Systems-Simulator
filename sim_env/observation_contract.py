"""
Observation Contract — explicit field classification.

Every field the simulator exposes to an external consumer has a declared
classification; the classification — never the field name — is authoritative:

    AGENT_OBSERVATION    may enter the agent observation space
    DEBUG_TELEMETRY      UI/logging only — forbidden from agent obs
    ORACLE_GROUND_TRUTH  privileged state — strictly forbidden from agent obs
    DIAGNOSTIC           out-of-band telemetry (GET_STATE, info dict,
                         trajectory diagnostic_data) — never agent obs

Channel categories for agent-visible fields live in
`ObservationChannelConfig.category` (observation_designer.py). This module
adds the declared *diagnostic* contract so GET_STATE and dataset/trajectory
consumers share one authoritative description instead of implicit payloads.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from sim_env.observation_designer import ChannelCategory

# Aliases matching the spec's AGENT/DEBUG/ORACLE/DIAGNOSTIC vocabulary.
FieldClass = ChannelCategory
DIAGNOSTIC = "diagnostic"


def diagnostic_state_contract() -> Dict[str, Dict[str, Any]]:
    """
    The declared out-of-band diagnostic contract served by GET_STATE.
    Field names here are the single source of truth for both the wire
    contract (DISCOVER_CONTRACT) and the payload builder.
    """
    return {
        "speed": {"classification": DIAGNOSTIC, "type": "float",
                  "description": "Vehicle forward speed (m/s)"},
        "pos": {"classification": DIAGNOSTIC, "type": "vec3",
                "description": "Vehicle world position"},
        "yaw": {"classification": DIAGNOSTIC, "type": "float",
                "description": "Vehicle heading (radians)"},
        "sim_time": {"classification": DIAGNOSTIC, "type": "float",
                     "description": "Simulation time (seconds)"},
        "total_reward": {"classification": DIAGNOSTIC, "type": "float",
                         "description": "Accumulated episode reward"},
        "reward_breakdown": {"classification": DIAGNOSTIC, "type": "dict",
                             "description": "Per-component reward terms"},
    }


def build_diagnostic_state(env: Any) -> Dict[str, Any]:
    """
    Builds the GET_STATE diagnostic payload from the declared contract.
    Keys are derived from diagnostic_state_contract(), never ad-hoc.
    """
    st = env.vehicle.state
    payload = {
        "speed": float(st.speed),
        "pos": [st.pos.x, st.pos.y, st.pos.z],
        "yaw": float(st.yaw),
        "sim_time": env.clock.sim_time,
        "total_reward": env.reward_engine.total_accumulated_reward,
        "reward_breakdown": env.reward_engine.last_breakdown,
    }
    return {k: payload[k] for k in diagnostic_state_contract()}


def channel_classification_map(agent: Any) -> Dict[str, str]:
    """
    {channel_name: category} for every declared observation channel —
    used by dataset/trajectory consumers to keep agent fields and
    debug/oracle fields strictly separated.
    """
    if agent is None or getattr(agent, "observation_space", None) is None:
        return {}
    return {
        c.name: c.category
        for c in agent.observation_space.channels
    }


def agent_observation_fields(agent: Any) -> Dict[str, str]:
    """Only the fields legally visible to the agent (enabled, AGENT class)."""
    if agent is None or getattr(agent, "observation_space", None) is None:
        return {}
    return {
        c.name: c.category
        for c in agent.observation_space.get_active_channels()
        if c.category == ChannelCategory.AGENT_OBSERVATION
    }
