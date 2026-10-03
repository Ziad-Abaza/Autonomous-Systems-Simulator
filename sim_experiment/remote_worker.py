"""
Remote training workers — versioned, authenticated, LAN-oriented.

WorkerService exposes a LocalTrainingOrchestrator over a newline-delimited
JSON TCP channel. A BatchScheduler can then dispatch runs to a worker
running on another process/host on a LAN — the worker executes trainer
subprocesses against its own (shared or replica) experiments_root.

    RemoteWorkerAdapter (client)  ──TCP/NDJSON──>  WorkerService
                                                       └── LocalTrainingOrchestrator
                                                                └── trainer subprocess

Security model: a shared-token gate for LAN use — NOT internet exposure.
Every message carries {"v", "type", "token"}; unauthenticated messages are
rejected before dispatch; protocol version mismatches fail cleanly.
"""

from __future__ import annotations
import hmac
import json
import os
import select
import socket
import socketserver
import threading
import time
import uuid
from typing import Any, Dict, Optional

from sim_experiment.manifest import ExperimentManifest
from sim_experiment.orchestrator import LocalTrainingOrchestrator

WORKER_PROTOCOL_VERSION = "1.0"
SUPPORTED_WORKER_VERSIONS = ("1.0",)


class WorkerService:
    """TCP NDJSON worker service wrapping a LocalTrainingOrchestrator."""

    def __init__(self, host: str, port: int, experiments_root: str, token: str):
        if not token:
            raise ValueError("WorkerService requires a non-empty shared token")
        self.host = host
        self._port = port
        self._token = token
        self.worker_id = f"worker_{uuid.uuid4().hex[:12]}"
        self.orch = LocalTrainingOrchestrator(experiments_root=experiments_root)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def port(self) -> int:
        return self._sock.getsockname()[1] if self._sock else -1

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self._port))
        self._sock.listen(8)
        self._sock.setblocking(False)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ------------------------------------------------------------ protocol

    def _capabilities(self) -> Dict[str, Any]:
        return {
            "type": "remote",
            "trainers": sorted(self.orch.trainer_modules()),
            "max_envs_per_run": None,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "worker_id": self.worker_id,
            "heartbeat": True,
        }

    def _handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        # Auth gate first — never dispatch unauthenticated messages.
        token = msg.get("token", "")
        if not hmac.compare_digest(str(token), self._token):
            return {"type": "ERROR", "error": {
                "type": "auth_failed", "message": "Invalid worker token"}}

        ver = msg.get("v", WORKER_PROTOCOL_VERSION)
        if ver not in SUPPORTED_WORKER_VERSIONS:
            return {"type": "ERROR", "error": {
                "type": "protocol_version_mismatch",
                "supported": list(SUPPORTED_WORKER_VERSIONS)}}

        mtype = msg.get("type")

        def _root_checked(exp_dir_field: str) -> Optional[Dict[str, Any]]:
            """Returns an ERROR response if msg[exp_dir_field] escapes root."""
            exp_dir = os.path.abspath(str(msg.get(exp_dir_field, "")))
            root = os.path.abspath(self.orch.experiments_root)
            try:
                inside = os.path.commonpath([root, exp_dir]) == root
            except ValueError:
                inside = False
            if not inside:
                return {"type": "ERROR", "error": {
                    "type": "invalid_experiment_dir",
                    "message": f"experiment_dir escapes worker root: "
                               f"{msg.get(exp_dir_field)!r}"}}
            return None

        try:
            if mtype == "HELLO":
                return {"type": "HELLO_ACK",
                        "protocol_version": WORKER_PROTOCOL_VERSION,
                        "supported": list(SUPPORTED_WORKER_VERSIONS),
                        "capabilities": self._capabilities()}

            if mtype == "REGISTER":
                return {"type": "REGISTER_ACK",
                        "worker_id": self.worker_id,
                        "capabilities": self._capabilities()}

            if mtype == "HEARTBEAT":
                return {"type": "HEARTBEAT_ACK",
                        "alive": True,
                        "worker_id": self.worker_id}

            if mtype == "STATUS":
                return {"type": "STATUS_ACK",
                        "alive": True,
                        "worker_id": self.worker_id,
                        "protocol_version": WORKER_PROTOCOL_VERSION,
                        "capabilities": self._capabilities()}

            if mtype == "LAUNCH":
                # Root containment: the experiment dir must live under this
                # worker's experiments root — remote callers cannot point
                # launches at arbitrary filesystem locations.
                bad = _root_checked("experiment_dir")
                if bad:
                    return bad
                manifest = ExperimentManifest.from_dict(msg["manifest"])
                run_id = self.orch.launch(
                    manifest, os.path.abspath(msg["experiment_dir"]),
                    trainer=msg.get("trainer", "ppo"),
                    env_mode=msg.get("env_mode", "inprocess"),
                    run_overrides=msg.get("run_overrides"))
                return {"type": "LAUNCH_ACK", "run_id": run_id}

            if mtype == "POLL":
                bad = _root_checked("experiment_dir")
                if bad:
                    return bad
                summary = self.orch.poll(
                    os.path.abspath(msg["experiment_dir"]), msg["run_id"])
                return {"type": "POLL_ACK", "summary": summary}

            if mtype == "CANCEL":
                bad = _root_checked("experiment_dir")
                if bad:
                    return bad
                self.orch.cancel(
                    os.path.abspath(msg["experiment_dir"]), msg["run_id"])
                return {"type": "CANCEL_ACK"}

            return {"type": "ERROR", "error": {
                "type": "unknown_type", "message": f"Unknown message type {mtype}"}}
        except Exception as e:
            return {"type": "ERROR", "error": {
                "type": "launch_rejected" if mtype == "LAUNCH" else "service_error",
                "message": str(e)}}

    # ------------------------------------------------------------ accept loop

    def _accept_loop(self) -> None:
        while self._running and self._sock:
            try:
                readable, _, _ = select.select([self._sock], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not readable:
                continue
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,),
                             daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        """One request/response round per connection (newline JSON)."""
        try:
            conn.settimeout(30.0)
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    return
                buf += chunk
                if len(buf) > (64 << 20):
                    return
            line = buf.split(b"\n", 1)[0]
            msg = json.loads(line.decode("utf-8"))
            resp = self._handle(msg)
            conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


class RemoteWorkerAdapter:
    """
    Scheduler worker adapter over TCP. One short-lived connection per RPC.
    `capabilities()` performs a HELLO handshake (cached).
    """

    def __init__(self, host: str, port: int, token: str, timeout: float = 30.0):
        self.host = host
        self.port = int(port)
        self._token = token
        self.timeout = timeout
        self._caps: Optional[Dict[str, Any]] = None
        self.worker_id: Optional[str] = None

    def _rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"v": WORKER_PROTOCOL_VERSION, "token": self._token, **payload}
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            raise RuntimeError("Remote worker closed connection without reply")
        resp = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if resp.get("type") == "ERROR":
            err = resp.get("error", {})
            raise RuntimeError(
                f"Remote worker error [{err.get('type', 'unknown')}]: "
                f"{err.get('message', '')}")
        return resp

    def handshake(self) -> Dict[str, Any]:
        resp = self._rpc({"type": "HELLO"})
        caps = resp.get("capabilities") or {}
        if caps.get("worker_id"):
            self.worker_id = caps["worker_id"]
        return resp

    def register(self) -> Dict[str, Any]:
        """Explicit worker registration — stable identity for lease tracking."""
        resp = self._rpc({"type": "REGISTER"})
        self.worker_id = resp.get("worker_id")
        return resp

    def heartbeat(self) -> Dict[str, Any]:
        """Liveness check; raises if the worker is unreachable."""
        resp = self._rpc({"type": "HEARTBEAT"})
        if resp.get("worker_id"):
            self.worker_id = resp["worker_id"]
        return {"alive": bool(resp.get("alive")),
                "worker_id": resp.get("worker_id")}

    def capabilities(self) -> Dict[str, Any]:
        if self._caps is None:
            self._caps = self.handshake()["capabilities"]
        return dict(self._caps)

    def launch(self, manifest, experiment_dir: str, trainer: str,
               env_mode: str, run_overrides: Optional[Dict[str, Any]]) -> str:
        resp = self._rpc({
            "type": "LAUNCH",
            "manifest": manifest.to_dict(),
            "experiment_dir": experiment_dir,
            "trainer": trainer,
            "env_mode": env_mode,
            "run_overrides": run_overrides or {},
        })
        return resp["run_id"]

    def poll(self, experiment_dir: str, run_id: str) -> Dict[str, Any]:
        resp = self._rpc({"type": "POLL", "experiment_dir": experiment_dir,
                          "run_id": run_id})
        return resp["summary"]

    def cancel(self, experiment_dir: str, run_id: str) -> None:
        self._rpc({"type": "CANCEL", "experiment_dir": experiment_dir,
                   "run_id": run_id})
