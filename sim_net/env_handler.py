"""
Environment command handler — the shared message-dispatch core used by both
SimulationServer (single client) and SimServerMulti (one env per client).

Holds a SimulationEnvironment plus per-connection negotiated protocol state;
translates wire messages into env calls and response payloads.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np

from sim_net.protocol import (
    MessageType, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
    SET_SCENARIO_MIN_VERSION,
)
from sim_env.environment import SimulationEnvironment
from sim_env.observation_contract import (
    build_diagnostic_state, diagnostic_state_contract, channel_classification_map,
)


def _version_key(v: str) -> tuple:
    """Sortable key for dotted protocol version strings."""
    parts = []
    for p in str(v).split('.'):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _negotiate_protocol(client_versions) -> Optional[str]:
    """Picks the highest mutually supported protocol version, or None."""
    if not client_versions:
        return PROTOCOL_VERSION  # Legacy client: assume it speaks the current protocol
    mutual = [v for v in client_versions if v in SUPPORTED_PROTOCOL_VERSIONS]
    if not mutual:
        return None
    return max(mutual, key=_version_key)


class SimulationCommandHandler:
    """
    Per-connection command dispatch bound to one SimulationEnvironment.
    `client_version` is set by HANDSHAKE and gates versioned commands.
    """

    def __init__(self, env: SimulationEnvironment):
        self.env = env
        self.client_version: Optional[str] = None

    def handle(self, msg_type: str, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if msg_type == MessageType.HANDSHAKE:
            negotiated = _negotiate_protocol(
                payload.get('protocol_versions') or payload.get('client_protocol_versions')
            )
            if negotiated is None:
                return MessageType.ERROR, {
                    'error': 'protocol_version_mismatch: no mutually supported protocol version',
                    'supported_versions': list(SUPPORTED_PROTOCOL_VERSIONS)
                }
            self.client_version = negotiated
            spec = {
                'protocol_version': negotiated,
                'supported_versions': list(SUPPORTED_PROTOCOL_VERSIONS),
                'action_space': self.env.action_config.to_dict(),
                'observation_schema': self.env.observation_schema.to_dict(),
                'vector_dim': self.env.observation_schema.compute_vector_dim(),
                'track_name': self.env.road_def.name,
                'track_length': self.env.track.spline.total_length,
                'physics_hz': self.env.clock.physics_hz,
                'dt': self.env.clock.dt,
            }
            return MessageType.HANDSHAKE_ACK, spec

        elif msg_type == MessageType.DISCOVER_CONTRACT:
            # Full declarative contract discovery
            if self.env.agent is not None:
                act_schema = self.env.agent.action_space.export_schema()
                obs_schema = self.env.agent.observation_space.export_schema()
                reward_graph = self.env.agent.reward_function.export_graph()
                term_rules = [r.to_dict() for r in self.env.agent.termination_rules.rules if r.enabled]
            else:
                act_schema = {
                    'space_type': self.env.action_config.type,
                    'num_channels': 3,
                    'continuous_low': self.env.action_config.continuous_low,
                    'continuous_high': self.env.action_config.continuous_high
                }
                obs_schema = {
                    'vector_dimension': self.env.observation_schema.compute_vector_dim(),
                    'flatten_vector': self.env.observation_schema.flatten_vector
                }
                reward_graph = {'weights': self.env.reward_engine.config.to_dict()}
                term_rules = [{'rules': self.env.termination_engine.config.to_dict()}]

            contract = {
                'protocol_version': PROTOCOL_VERSION,
                'supported_versions': list(SUPPORTED_PROTOCOL_VERSIONS),
                'environment_id': self.env.road_def.name,
                'environment_version': getattr(self.env, 'environment_version', '1.0.0'),
                'physics_hz': self.env.clock.physics_hz,
                'dt': self.env.clock.dt,
                'action_schema': act_schema,
                'observation_schema': obs_schema,
                'reward_schema': reward_graph,
                'termination_schema': term_rules,
                'sensors': [s.to_dict() for s in self.env.sensors.sensors.values()],
                'observation_field_classes': channel_classification_map(self.env.agent),
                'diagnostic_fields': diagnostic_state_contract(),
                'scenario': {
                    'name': self.env.scenario_def.name if self.env.scenario_def else self.env.scenario.name,
                    'weather': self.env.scenario_def.weather if self.env.scenario_def else self.env.scenario.weather,
                    'friction_mult': self.env.scenario_def.surface_friction_mult if self.env.scenario_def else self.env.scenario.surface_friction_mult
                },
                'capabilities': {
                    'scenario_update': True,
                    'episode_state': self._episode_state(),
                }
            }
            return MessageType.CONTRACT_ACK, contract

        elif msg_type == MessageType.RESET:
            seed = payload.get('seed', None)
            options = payload.get('options', None)
            obs, info = self.env.reset(seed=seed, options=options)
            return MessageType.RESET_ACK, {
                'obs': obs if not isinstance(obs, np.ndarray) else obs.tolist(),
                'info': info
            }

        elif msg_type == MessageType.STEP:
            action = payload.get('action', [0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = self.env.step(action)
            return MessageType.STEP_ACK, {
                'obs': obs if not isinstance(obs, np.ndarray) else obs.tolist(),
                'reward': float(reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'info': info
            }

        elif msg_type == MessageType.GET_STATE:
            # Payload keys come from the declared diagnostic contract —
            # diagnostic fields are never agent observations.
            return MessageType.STATE_ACK, build_diagnostic_state(self.env)

        elif msg_type == MessageType.SET_SCENARIO:
            return self._handle_set_scenario(payload)

        return MessageType.ERROR, {'error': f"Unknown message type: {msg_type}"}

    def _episode_state(self) -> str:
        """
        Episode lifecycle from the wire's perspective:
        - 'mid_episode': stepped at least once, not yet terminated/truncated
        - 'terminated' : last episode ended (done)
        - 'idle'       : freshly reset or never stepped
        """
        if getattr(self.env, 'is_done', False):
            return 'terminated'
        if getattr(self.env, 'current_step', 0) > 0:
            return 'mid_episode'
        return 'idle'

    def _handle_set_scenario(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        SET_SCENARIO — validated scenario update for a running headless env.

        State machine:
            mid_episode + reset != true  -> ERROR scenario_update_rejected
            otherwise                    -> apply scenario; when reset is true
                                            env.reset(seed) returns fresh obs.
        """
        if _version_key(self.client_version or "0.0") < _version_key(SET_SCENARIO_MIN_VERSION):
            return MessageType.ERROR, {
                'error': (
                    f"unsupported_in_protocol_version: SET_SCENARIO requires "
                    f"protocol >= {SET_SCENARIO_MIN_VERSION} "
                    f"(negotiated {self.client_version or 'none'})"),
                'supported_versions': list(SUPPORTED_PROTOCOL_VERSIONS),
            }

        force_reset = bool(payload.get('reset', False))
        state = self._episode_state()
        if state == 'mid_episode' and not force_reset:
            return MessageType.ERROR, {
                'error': (
                    'scenario_update_rejected: episode is active; finish the '
                    'episode first or send reset=true to apply the scenario '
                    'and reset atomically'),
                'episode_state': state,
            }

        from sim_env.scenario_designer import ScenarioDefinition
        scen_dict = payload.get('scenario')
        scen_id = payload.get('scenario_id')
        try:
            if scen_dict is not None:
                scenario_def = ScenarioDefinition.from_dict(scen_dict)
            elif scen_id:
                lib = ScenarioDefinition.get_standard_scenarios()
                if scen_id not in lib:
                    return MessageType.ERROR, {
                        'error': f"unknown_scenario: '{scen_id}' is not in the "
                                 f"standard scenario library",
                        'known': sorted(lib.keys()),
                    }
                scenario_def = lib[scen_id]
            else:
                return MessageType.ERROR, {
                    'error': "invalid_scenario: provide 'scenario' (dict) or 'scenario_id'",
                }
        except Exception as exc:
            return MessageType.ERROR, {
                'error': f"invalid_scenario: {type(exc).__name__}: {exc}",
            }

        try:
            self.env.set_scenario(scenario_def)
        except Exception as exc:
            return MessageType.ERROR, {
                'error': f"scenario_apply_failed: {type(exc).__name__}: {exc}",
            }

        ack: Dict[str, Any] = {
            'scenario': {
                'scenario_id': scenario_def.scenario_id,
                'name': scenario_def.name,
                'weather': scenario_def.weather,
                'friction_mult': scenario_def.surface_friction_mult,
            },
            'episode_state': 'applied',
        }
        if force_reset:
            obs, info = self.env.reset(seed=payload.get('seed'))
            ack['obs'] = obs if not isinstance(obs, np.ndarray) else obs.tolist()
            ack['info'] = info
            ack['episode_state'] = 'reset'
        return MessageType.SET_SCENARIO_ACK, ack
