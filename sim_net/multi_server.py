"""
Multi-client TCP simulation server: one process, one port, N environments.

Each connected client is bound to its own SimulationEnvironment instance
(env-per-connection). Clients are fully independent — resets, steps, and
SET_SCENARIO on one connection never touch another env. A connection slot
is freed when the client disconnects and its env is returned to the pool
for the next client.

Single-threaded, non-blocking, select()-driven: like SimulationServer, it is
polled from the host application's main loop via poll_and_process().
"""

from __future__ import annotations
import socket
import select
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from sim_net.protocol import MessageType, ProtocolEncoder
from sim_net.env_handler import SimulationCommandHandler
from sim_env.environment import SimulationEnvironment


class _ClientSlot:
    __slots__ = ("sock", "addr", "rx_buffer", "handler", "env", "connected_at")

    def __init__(self, sock: socket.socket, addr, env: SimulationEnvironment):
        self.sock = sock
        self.addr = addr
        self.rx_buffer = ""
        self.env = env
        self.handler = SimulationCommandHandler(env)
        self.connected_at = time.time()


class SimServerMulti:
    """
    TCP server hosting `num_envs` environments on a single port, one env per
    client connection. Built for env_mode='tcp_multi' — real serialization
    boundary with a single simulator process.
    """

    def __init__(self, env_factories: List[Callable[[], SimulationEnvironment]],
                 host: str = "127.0.0.1", port: int = 8765):
        if not env_factories:
            raise ValueError("SimServerMulti requires at least one env factory")
        self.host = host
        self.port = port
        self.envs: List[SimulationEnvironment] = [f() for f in env_factories]
        self._free_envs: Deque[SimulationEnvironment] = deque(self.envs)

        self.server_socket: Optional[socket.socket] = None
        self.clients: List[_ClientSlot] = []
        self.is_running = False
        self.total_steps_served = 0
        self.last_latency_ms = 0.0

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def is_client_connected(self) -> bool:
        return bool(self.clients)

    def start(self) -> bool:
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.num_envs)
            self.server_socket.setblocking(False)
            self.is_running = True
            return True
        except Exception as e:
            print(f"[SimServerMulti] Failed to bind to {self.host}:{self.port}: {e}")
            self.is_running = False
            return False

    def stop(self) -> None:
        self.is_running = False
        for slot in list(self.clients):
            self._drop_client(slot)
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

    def poll_and_process(self) -> bool:
        """
        Polls the accept socket and every connected client once.
        Returns True if at least one STEP was processed.
        """
        if not self.is_running or not self.server_socket:
            return False

        # 1. Accept pending connections while env slots remain
        try:
            readable, _, _ = select.select([self.server_socket], [], [], 0)
        except Exception:
            readable = []
        if readable:
            try:
                sock, addr = self.server_socket.accept()
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setblocking(False)
                if self._free_envs:
                    slot = _ClientSlot(sock, addr, self._free_envs.popleft())
                    self.clients.append(slot)
                    print(f"[SimServerMulti] Client connected from {addr} "
                          f"({len(self.clients)}/{self.num_envs} slots used)")
                else:
                    # Full: tell the client, then hang up.
                    try:
                        sock.sendall(ProtocolEncoder.encode(
                            MessageType.ERROR,
                            {'error': f'server_full: all {self.num_envs} env slots busy'}))
                    except Exception:
                        pass
                    sock.close()
                    print(f"[SimServerMulti] Rejected client {addr}: server full")
            except Exception as e:
                print(f"[SimServerMulti] Accept error: {e}")

        # 2. Read + dispatch for each connected client
        step_processed = False
        for slot in list(self.clients):
            try:
                readable, _, exceptional = select.select(
                    [slot.sock], [], [slot.sock], 0)
            except Exception:
                self._drop_client(slot)
                continue
            if exceptional:
                self._drop_client(slot)
                continue
            if not readable:
                continue
            try:
                chunk = slot.sock.recv(16384)
                if not chunk:
                    self._drop_client(slot)
                    continue
                slot.rx_buffer += chunk.decode('utf-8', errors='ignore')
            except BlockingIOError:
                continue
            except Exception:
                self._drop_client(slot)
                continue

            while "\n" in slot.rx_buffer:
                line, slot.rx_buffer = slot.rx_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    t0 = time.perf_counter()
                    msg_type, payload = ProtocolEncoder.decode(line)
                    response = slot.handler.handle(msg_type, payload)
                    if response:
                        slot.sock.sendall(
                            ProtocolEncoder.encode(response[0], response[1]))
                    if msg_type == MessageType.STEP:
                        step_processed = True
                        self.total_steps_served += 1
                    self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    print(f"[SimServerMulti] Error handling message "
                          f"'{line[:50]}': {e}")
                    try:
                        slot.sock.sendall(ProtocolEncoder.encode(
                            MessageType.ERROR, {'error': str(e)}))
                    except Exception:
                        self._drop_client(slot)
                        break

        return step_processed

    def _drop_client(self, slot: _ClientSlot) -> None:
        try:
            slot.sock.close()
        except Exception:
            pass
        if slot in self.clients:
            self.clients.remove(slot)
            # Env returns to the pool for the next client — reset it so a
            # recycled env never carries over episode state from the
            # disconnected client.
            try:
                slot.env.reset()
            except Exception:
                pass
            self._free_envs.append(slot.env)
            print(f"[SimServerMulti] Client {slot.addr} disconnected "
                  f"({len(self.clients)} slots used)")
