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

from sim_net.protocol import MessageType, ProtocolEncoder
from sim_env.environment import SimulationEnvironment


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
            # Return environment spec
            spec = {
                'protocol_version': '2.0',
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
                'protocol_version': '2.0',
                'environment_id': self.env.road_def.name,
                'environment_version': getattr(self.env, 'environment_version', '1.0.0'),
                'physics_hz': self.env.clock.physics_hz,
                'dt': self.env.clock.dt,
                'action_schema': act_schema,
                'observation_schema': obs_schema,
                'reward_schema': reward_graph,
                'termination_schema': term_rules,
                'sensors': [s.to_dict() for s in self.env.sensors.sensors.values()],
                'scenario': {
                    'name': self.env.scenario_def.name if self.env.scenario_def else self.env.scenario.name,
                    'weather': self.env.scenario_def.weather if self.env.scenario_def else self.env.scenario.weather,
                    'friction_mult': self.env.scenario_def.surface_friction_mult if self.env.scenario_def else self.env.scenario.surface_friction_mult
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
            st = self.env.vehicle.state
            return MessageType.STATE_ACK, {
                'speed': float(st.speed),
                'pos': [st.pos.x, st.pos.y, st.pos.z],
                'yaw': float(st.yaw),
                'sim_time': self.env.clock.sim_time,
                'total_reward': self.env.reward_engine.total_accumulated_reward,
                'reward_breakdown': self.env.reward_engine.last_breakdown,
            }

        return MessageType.ERROR, {'error': f"Unknown message type: {msg_type}"}

    def _disconnect_client(self) -> None:
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        self.client_socket = None
        self.client_addr = None
        self.is_client_connected = False
        print("[SimServer] External AI disconnected.")
