"""
High-frequency TCP simulation server exposing the environment to external AI models.
Handles lockstep or asynchronous stepping, disconnections, and multiple client lifecycle events.
"""

from __future__ import annotations
import socket
import select
import threading
import time
from typing import Optional, Dict, Any, Tuple

from sim_net.protocol import MessageType, ProtocolEncoder
from sim_net.env_handler import SimulationCommandHandler
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
        # Message dispatch lives in the shared command handler so the same
        # env semantics serve SimulationServer and SimServerMulti.
        self._handler = SimulationCommandHandler(env)

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
        return self._handler.handle(msg_type, payload)

    def _disconnect_client(self) -> None:
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        self.client_socket = None
        self.client_addr = None
        self.is_client_connected = False
        self._handler.client_version = None
        print("[SimServer] External AI disconnected.")
