"""
Lightweight client connecting an external AI process to the simulation server over TCP.
"""

from __future__ import annotations
import socket
import json
import time
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
from sim_net.protocol import MessageType, ProtocolEncoder


class SimulationClient:
    """
    Client for interacting with the simulation environment from an external AI process.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.env_spec: Dict[str, Any] = {}
        self._rx_buffer = ""

    def connect(self) -> Dict[str, Any]:
        """Connects to simulator and exchanges handshake."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

        # Send Handshake declaring supported protocol versions for negotiation
        from sim_net.protocol import SUPPORTED_PROTOCOL_VERSIONS
        self._send(MessageType.HANDSHAKE, {'protocol_versions': list(SUPPORTED_PROTOCOL_VERSIONS)})
        msg_type, payload = self._receive()
        if msg_type != MessageType.HANDSHAKE_ACK:
            raise ConnectionError(f"Handshake failed, unexpected response: {msg_type}")

        self.env_spec = payload
        return payload

    def discover_contract(self) -> Dict[str, Any]:
        """Discovers rich declarative environment contract (Phase 3)."""
        self._send(MessageType.DISCOVER_CONTRACT, {})
        msg_type, payload = self._receive()
        if msg_type != MessageType.CONTRACT_ACK:
            raise RuntimeError(f"Contract discovery failed: {payload}")
        return payload

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sends reset command and returns (observation, info)."""
        payload = {'seed': seed, 'options': options}
        self._send(MessageType.RESET, payload)
        msg_type, resp = self._receive()
        if msg_type != MessageType.RESET_ACK:
            raise RuntimeError(f"Reset failed: {resp}")
        obs = np.array(resp['obs'], dtype=np.float32)
        info = resp.get('info', {})
        return obs, info

    def step(self, action: Union[np.ndarray, List[float], int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action and returns (observation, reward, terminated, truncated, info)."""
        act_payload = action.tolist() if isinstance(action, np.ndarray) else action
        self._send(MessageType.STEP, {'action': act_payload})
        msg_type, resp = self._receive()
        if msg_type != MessageType.STEP_ACK:
            raise RuntimeError(f"Step failed: {resp}")
        obs = np.array(resp['obs'], dtype=np.float32)
        reward = float(resp['reward'])
        terminated = bool(resp['terminated'])
        truncated = bool(resp['truncated'])
        info = resp.get('info', {})
        return obs, reward, terminated, truncated, info

    def set_scenario(
        self,
        scenario: Union[Dict[str, Any], str, Any],
        seed: Optional[int] = None,
        reset: bool = False,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Sends SET_SCENARIO (protocol >= 2.1).

        `scenario`: serialized ScenarioDefinition dict, a standard-library
        scenario_id string, or a ScenarioDefinition object.
        `reset=True` applies the scenario and resets atomically (allowed
        mid-episode); without it the server rejects while an episode is
        active. Returns (obs, info) — obs is None when no reset occurred.
        """
        if isinstance(scenario, str):
            payload = {'scenario_id': scenario}
        else:
            scen_dict = scenario.to_dict() if hasattr(scenario, 'to_dict') else dict(scenario)
            payload = {'scenario': scen_dict}
        payload['seed'] = seed
        payload['reset'] = bool(reset)
        self._send(MessageType.SET_SCENARIO, payload)
        msg_type, resp = self._receive()
        if msg_type != MessageType.SET_SCENARIO_ACK:
            raise RuntimeError(f"SET_SCENARIO failed: {resp}")
        obs = np.array(resp['obs'], dtype=np.float32) if 'obs' in resp else None
        return obs, resp

    def get_state(self) -> Dict[str, Any]:
        self._send(MessageType.GET_STATE, {})
        _, resp = self._receive()
        return resp

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        if not self.sock:
            raise ConnectionError("Not connected to simulation server.")
        data = ProtocolEncoder.encode(msg_type, payload)
        self.sock.sendall(data)

    def _receive(self) -> Tuple[str, Dict[str, Any]]:
        if not self.sock:
            raise ConnectionError("Not connected to simulation server.")
        while "\n" not in self._rx_buffer:
            chunk = self.sock.recv(16384)
            if not chunk:
                raise ConnectionResetError("Connection closed by simulator.")
            self._rx_buffer += chunk.decode('utf-8')

        line, self._rx_buffer = self._rx_buffer.split("\n", 1)
        return ProtocolEncoder.decode(line)
