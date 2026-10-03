"""
High-frequency TCP simulation server exposing the environment to external AI models.
Handles lockstep or asynchronous stepping, disconnections, and multiple client lifecycle events.
"""

from __future__ import annotations
import socket
import select
import threading
import queue
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np

from sim_net.protocol import (
    MessageType, ProtocolEncoder, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
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


class SimulationServer:
    """
    TCP server hosting the SimulationEnvironment for external AI/RL connections.
    """
    def __init__(self, env: SimulationEnvironment, host: str = "127.0.0.1", port: int = 8765):
        self.env = env
        self.host = host
        self.port = port

        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_addr = None

        self.is_running = False
        self.is_client_connected = False
        self.total_steps_served = 0
        self.last_latency_ms = 0.0

        self._rx_buffer = ""
        self._lock = threading.Lock()
        # Protocol version negotiated for the currently connected client
        # (reset on disconnect). Gates version-sensitive commands.
        self._client_version: Optional[str] = None

    def start(self) -> bool:
        """Starts listening on host:port."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TCP_NODELAY disables Nagle's algorithm for low-latency simulation packets
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.setblocking(False)
            self.is_running = True
            return True
        except Exception as e:
            print(f"[SimServer] Failed to bind to {self.host}:{self.port}: {e}")
            self.is_running = False
            return False

    def stop(self) -> None:
        self.is_running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None
        self.is_client_connected = False

    def poll_and_process(self) -> bool:
        """
        Polls non-blocking for client connections and incoming messages.
        Returns True if a step action was processed this frame.
        """
        if not self.is_running or not self.server_socket:
            return False

        # 1. Check for incoming connection if no client connected
        if not self.client_socket:
            readable, _, _ = select.select([self.server_socket], [], [], 0)
            if readable:
                try:
                    sock, addr = self.server_socket.accept()
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setblocking(False)
                    self.client_socket = sock
                    self.client_addr = addr
                    self.is_client_connected = True
                    self._rx_buffer = ""
                    print(f"[SimServer] External AI connected from {addr}")
                except Exception as e:
                    print(f"[SimServer] Error accepting client: {e}")
            return False

        # 2. Check for incoming data from connected client
        readable, _, exceptional = select.select([self.client_socket], [], [self.client_socket], 0)
        if exceptional:
            self._disconnect_client()
            return False

        if readable:
            try:
                chunk = self.client_socket.recv(16384)
                if not chunk:
                    self._disconnect_client()
                    return False
                self._rx_buffer += chunk.decode('utf-8', errors='ignore')
            except BlockingIOError:
                pass
            except Exception as e:
                print(f"[SimServer] Client read error: {e}")
                self._disconnect_client()
                return False

        # 3. Process complete messages separated by newline
        step_processed = False
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                t0 = time.perf_counter()
                msg_type, payload = ProtocolEncoder.decode(line)
                response = self._handle_message(msg_type, payload)
                if response:
                    resp_bytes = ProtocolEncoder.encode(response[0], response[1])
                    self.client_socket.sendall(resp_bytes)
                if msg_type == MessageType.STEP:
                    step_processed = True
                    self.total_steps_served += 1
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                print(f"[SimServer] Error handling message '{line[:50]}': {e}")
                err_bytes = ProtocolEncoder.encode(MessageType.ERROR, {'error': str(e)})
                try:
                    self.client_socket.sendall(err_bytes)
                except Exception:
                    pass

        return step_processed

    def _handle_message(self, msg_type: str, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if msg_type == MessageType.HANDSHAKE:
            negotiated = _negotiate_protocol(
                payload.get('protocol_versions') or payload.get('client_protocol_versions')
            )
            if negotiated is None:
                return MessageType.ERROR, {
                    'error': 'protocol_version_mismatch: no mutually supported protocol version',
                    'supported_versions': list(SUPPORTED_PROTOCOL_VERSIONS)
                }
            self._client_version = negotiated
            # Return environment spec
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
                                            (or the episode just terminated),
                                            env.reset(seed) returns fresh obs.

        Scenario source (exactly one): payload['scenario'] (a serialized
        ScenarioDefinition dict) or payload['scenario_id'] (resolved against
        the built-in standard-scenario library).
        """
        if _version_key(self._client_version or "0.0") < _version_key(SET_SCENARIO_MIN_VERSION):
            return MessageType.ERROR, {
                'error': (
                    f"unsupported_in_protocol_version: SET_SCENARIO requires "
                    f"protocol >= {SET_SCENARIO_MIN_VERSION} "
                    f"(negotiated {self._client_version or 'none'})"),
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

    def _disconnect_client(self) -> None:
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        self.client_socket = None
        self.client_addr = None
        self.is_client_connected = False
        self._client_version = None
        print("[SimServer] External AI disconnected.")
