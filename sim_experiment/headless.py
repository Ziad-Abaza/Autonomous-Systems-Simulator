"""
Headless environment construction and pooling.

`build_env_from_dicts` is the single factory used by trainers, evaluators,
and the CLI to materialize a SimulationEnvironment from serialized
project/scenario dictionaries — no UI or render dependencies.

`HeadlessEnvPool` manages N fully independent in-process environments
(independent seeds, episodes, agent state). `HeadlessSimProcessPool`
launches N simulator OS processes (main.py --headless --port) when
stronger isolation or TCP access is needed.
"""

from __future__ import annotations
import os
import socket
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

from sim_env.environment import SimulationEnvironment
from sim_env.scenario_designer import ScenarioDefinition
from sim_project.serializer import EnvironmentProject


def build_env_from_dicts(
    env_dict: Dict[str, Any],
    scenario_dict: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> SimulationEnvironment:
    """Materializes a SimulationEnvironment from serialized dictionaries."""
    project = EnvironmentProject.from_dict(env_dict)
    scenario = (
        ScenarioDefinition.from_dict(scenario_dict)
        if scenario_dict else project.scenario_def
    )
    env = SimulationEnvironment(
        road_def=project.road_def,
        vehicle_config=project.vehicle_config,
        action_config=project.action_config,
        observation_schema=project.observation_schema,
        reward_config=project.reward_config,
        termination_config=project.termination_config,
        randomization_config=project.randomization_config,
        scenario_config=project.scenario_config,
        agent=project.agent,
        episode_config=project.episode_config,
        scenario_def=scenario,
        seed=seed,
    )
    for ent in project.entities:
        env.add_entity(ent)
    return env


class HeadlessEnvPool:
    """
    N independent in-process environments with independent seeds and
    episodes. Environments share no mutable state; each is reset with a
    derived seed (base_seed + index).
    """

    def __init__(
        self,
        env_dict: Dict[str, Any],
        scenario_dict: Optional[Dict[str, Any]] = None,
        num_envs: int = 1,
        base_seed: int = 42,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.envs: List[SimulationEnvironment] = [
            build_env_from_dicts(env_dict, scenario_dict, seed=base_seed + i)
            for i in range(num_envs)
        ]

    def __len__(self) -> int:
        return len(self.envs)

    def reset_all(self) -> List[Any]:
        return [env.reset(seed=env.clock.seed) for env in self.envs]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HeadlessSimProcessPool:
    """
    Launches N headless simulator OS processes (main.py --headless --port P)
    for TCP-driven parallel training. Process isolation: each simulator has
    its own Python process, socket, and RNG state.
    """

    def __init__(self, num_envs: int, base_port: Optional[int] = None, repo_root: Optional[str] = None):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.repo_root = repo_root or os.path.dirname(os.path.abspath(__file__)) + os.sep + ".."
        self.repo_root = os.path.abspath(self.repo_root)
        base = base_port if base_port is not None else _free_port()
        self.ports = [base + i for i in range(num_envs)]
        self.procs: List[subprocess.Popen] = []

    def start(self, timeout_s: float = 30.0) -> List[int]:
        """Spawns all simulators and waits for their ports to accept TCP."""
        for port in self.ports:
            log_path = os.path.join(self.repo_root, "logs")
            os.makedirs(log_path, exist_ok=True)
            log = open(os.path.join(log_path, f"sim_{port}.log"), "w", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, "main.py", "--headless", "--port", str(port)],
                cwd=self.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self.procs.append(proc)

        deadline = time.time() + timeout_s
        for port, proc in zip(self.ports, self.procs):
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Headless sim on port {port} exited early (code {proc.returncode})")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                self.stop()
                raise TimeoutError(f"Headless sim on port {port} did not open in {timeout_s}s")
        return self.ports

    def stop(self) -> None:
        for proc in self.procs:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.procs.clear()
