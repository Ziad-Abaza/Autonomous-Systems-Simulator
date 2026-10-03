"""
Process-isolated vectorized environments (Phase 6).

`ProcessVectorEnv` runs each environment in its own OS process
(`multiprocessing` spawn context — required on Windows/macOS) and steps
all envs concurrently: commands are pipelined to every worker before any
result is read, so worker processes execute steps in parallel. This is
real parallelism around the GIL — physics/sensor cost is paid in the
workers, not the trainer process.

Worker protocol (per-env Pipe):

    parent -> worker : ("reset", seed)        -> ("ok", obs)
                       ("step", action)       -> ("ok", (obs, reward,
                                                     terminated,
                                                     truncated, info))
                       ("set_scenario", dict) -> ("ok", None)
                       ("ping", None)         -> ("ok", "pong")
                       ("close", None)        -> worker exits

Failure semantics: a dead worker's pipe raises EOFError/OSError on the
next recv, which is surfaced as `EnvWorkerCrash(worker_index)`. The
crash does not corrupt sibling workers — remaining envs stay usable and
the exception propagates to the trainer so the run fails explicitly
(`worker_crash` is a retryable scheduler error type).
"""

from __future__ import annotations
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from sim_experiment.vec_env import VectorEnv, StepResult


class EnvWorkerCrash(RuntimeError):
    """Raised when an environment worker process dies or its pipe breaks."""

    def __init__(self, worker_index: int, message: str):
        self.worker_index = worker_index
        super().__init__(f"env worker {worker_index} crashed: {message}")


def _env_worker(conn, env_dict, scenario_dict, seed) -> None:
    """
    Default worker main: builds a real SimulationEnvironment in the child
    process and serves the command loop until 'close' or pipe EOF.
    """
    try:
        from sim_experiment.headless import build_env_from_dicts
        from sim_env.scenario_designer import ScenarioDefinition

        env = build_env_from_dicts(env_dict, scenario_dict, seed=seed)
        conn.send("ready")
        while True:
            try:
                cmd, payload = conn.recv()
            except EOFError:
                break
            try:
                if cmd == "close":
                    break
                if cmd == "reset":
                    obs, _info = env.reset(seed=payload)
                    conn.send(("ok", obs))
                elif cmd == "step":
                    conn.send(("ok", env.step(payload)))
                elif cmd == "set_scenario":
                    if hasattr(env, "set_scenario"):
                        env.set_scenario(ScenarioDefinition.from_dict(payload))
                    else:
                        # Rebuild path: same env_dict + new scenario, same seed
                        env = build_env_from_dicts(env_dict, payload, seed=seed)
                    conn.send(("ok", None))
                elif cmd == "ping":
                    conn.send(("ok", "pong"))
                else:
                    conn.send(("err", f"unknown command: {cmd}"))
            except Exception as exc:  # surface worker-side env errors
                conn.send(("err", f"{type(exc).__name__}: {exc}"))
    except Exception:
        # Env build/startup failure: report before dying so the parent can
        # raise a useful EnvWorkerCrash instead of a bare EOF.
        try:
            conn.send(("init_err", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ProcessVectorEnv(VectorEnv):
    """
    One worker process per environment, synchronous batched API.

    `env_dict`/`scenario_dict` are the serialized project/scenario dicts
    consumed by `build_env_from_dicts` (same as the inprocess path).
    `seeds[i]` is the build-time seed of worker i — reset seeds are
    supplied explicitly per `reset_all`/`reset_at` call by the trainer,
    preserving the deterministic per-(env,reset) seed contract.
    """

    kind = "process"

    def __init__(
        self,
        env_dict: Dict[str, Any],
        scenario_dict: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        worker_fn: Optional[Callable] = None,
        startup_timeout: float = 120.0,
    ):
        seeds = list(seeds or [0])
        if not seeds:
            raise ValueError("ProcessVectorEnv requires at least one seed")
        self.num_envs = len(seeds)
        self._closed = False
        self._dead: List[bool] = [False] * self.num_envs
        self._procs: List[mp.Process] = []
        self._conns: List[Any] = []

        ctx = mp.get_context("spawn")
        target = worker_fn or _env_worker
        for i, seed in enumerate(seeds):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=target,
                args=(child_conn, env_dict, scenario_dict, seed),
                daemon=True,
                name=f"sim-env-{i}",
            )
            proc.start()
            # Parent must drop its copy of the child end or EOF is never
            # observed on the parent side.
            child_conn.close()
            self._procs.append(proc)
            self._conns.append(parent_conn)

        # Wait for each worker to finish env construction.
        deadline = time.time() + startup_timeout
        for i, conn in enumerate(self._conns):
            while time.time() < deadline:
                if conn.poll(0.25):
                    break
                if not self._procs[i].is_alive():
                    break
            try:
                msg = conn.recv()
            except (EOFError, OSError) as exc:
                self.close()
                raise EnvWorkerCrash(i, f"startup failed: {exc}")
            if isinstance(msg, tuple) and msg and msg[0] == "init_err":
                detail = msg[1]
                self.close()
                raise EnvWorkerCrash(i, f"env build failed:\n{detail}")

    # ------------------------------------------------------------- plumbing

    def _mark_dead(self, i: int, exc: Exception) -> EnvWorkerCrash:
        self._dead[i] = True
        return EnvWorkerCrash(i, f"{type(exc).__name__}: {exc}")

    def _check_live(self, i: int) -> None:
        if self._closed:
            raise EnvWorkerCrash(i, "pool is closed")
        if self._dead[i]:
            raise EnvWorkerCrash(i, "worker already dead")
        if not self._procs[i].is_alive() and self._procs[i].exitcode is not None:
            raise self._mark_dead(i, RuntimeError(
                f"process exited (code {self._procs[i].exitcode})"))

    def _rpc_one(self, i: int, cmd: str, payload: Any = None) -> Any:
        self._check_live(i)
        conn = self._conns[i]
        try:
            conn.send((cmd, payload))
            status, value = conn.recv()
        except (EOFError, BrokenPipeError, OSError, ConnectionResetError) as exc:
            raise self._mark_dead(i, exc)
        if status == "err":
            raise EnvWorkerCrash(i, str(value))
        return value

    # ----------------------------------------------------------------- API

    def reset_all(self, seeds: List[Optional[int]]) -> List[Any]:
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        for i, conn in enumerate(self._conns):
            self._check_live(i)
            conn.send(("reset", seeds[i]))
        return [self._recv_result(i) for i in range(self.num_envs)]

    def reset_at(self, index: int, seed: Optional[int] = None) -> Any:
        return self._rpc_one(index, "reset", seed)

    def step_all(self, actions: List[Any]) -> List[StepResult]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        for i, conn in enumerate(self._conns):
            self._check_live(i)
            conn.send(("step", actions[i]))
        return [self._recv_result(i) for i in range(self.num_envs)]

    def _recv_result(self, i: int) -> Any:
        conn = self._conns[i]
        try:
            status, value = conn.recv()
        except (EOFError, BrokenPipeError, OSError, ConnectionResetError) as exc:
            raise self._mark_dead(i, exc)
        if status == "err":
            raise EnvWorkerCrash(i, str(value))
        return value

    def set_scenario(self, scenario_dict: Optional[Dict[str, Any]]) -> None:
        for i, conn in enumerate(self._conns):
            self._check_live(i)
            conn.send(("set_scenario", scenario_dict))
        for i in range(self.num_envs):
            self._recv_result(i)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i, conn in enumerate(self._conns):
            try:
                if self._procs[i].is_alive():
                    conn.send(("close", None))
            except Exception:
                pass
        deadline = time.time() + 5.0
        for proc in self._procs:
            remaining = max(0.0, deadline - time.time())
            proc.join(timeout=remaining)
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
