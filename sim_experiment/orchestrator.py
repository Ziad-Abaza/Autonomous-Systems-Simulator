"""
Local Training Orchestrator.

Launches external trainer processes for experiment runs on the local
machine (Windows-friendly, no Docker, no cloud). Safety model:

- Trainers run via structured subprocess invocation (arg list, never a
  shell string). Trainer modules are restricted to the
  sim_experiment.trainers.* namespace.
- Run directories must live inside the experiments root.
- stdout/stderr are captured to logs/ under the run dir.
- The orchestrator is the single writer of run.json status; the trainer
  reports through contract artifacts (metrics.jsonl, registry.jsonl,
  run_result.json).
- Cancellation uses graceful terminate() then kill() escalation.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

from sim_experiment.manifest import ExperimentManifest
from sim_experiment.run import RunManager, RunStatus
from sim_experiment.metrics import MetricsReader
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trainer_contract import build_contract, validate_contract
from sim_experiment.curriculum_runtime import validate_curriculum
from sim_experiment.capabilities import check_compatibility
from sim_experiment.headless import HeadlessSimProcessPool

TRAINER_MODULES = {
    "ppo": "sim_experiment.trainers.ppo_trainer",
    "sac": "sim_experiment.trainers.sac_trainer",
    "dqn": "sim_experiment.trainers.dqn_trainer",
    "dummy": "sim_experiment.trainers.dummy_trainer",
}

_TRAINER_PREFIX = "sim_experiment.trainers."


class LocalTrainingOrchestrator:
    """Creates runs, launches trainer subprocesses, monitors and cancels them."""

    def __init__(self, experiments_root: str = "experiments", python_exe: Optional[str] = None):
        self.experiments_root = os.path.abspath(experiments_root)
        self.python_exe = python_exe or sys.executable
        # Repo root (dir containing the sim_experiment package) — required
        # on sys.path for `-m sim_experiment.trainers.*` to resolve.
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.run_manager = RunManager()
        self._procs: Dict[str, subprocess.Popen] = {}          # run_dir -> Popen
        self._sim_pools: Dict[str, HeadlessSimProcessPool] = {}  # run_dir -> pool

    @staticmethod
    def trainer_modules() -> Dict[str, str]:
        """Registered trainer short-name -> module mapping."""
        return dict(TRAINER_MODULES)

    # ------------------------------------------------------------ validation

    def _resolve_trainer(self, trainer: str) -> str:
        module = TRAINER_MODULES.get(trainer, trainer)
        if not module.startswith(_TRAINER_PREFIX):
            raise ValueError(
                f"Trainer '{trainer}' is not allowed: modules must live under {_TRAINER_PREFIX}"
            )
        # Module must not contain traversal / odd chars
        if not all(c.isalnum() or c in "._" for c in module):
            raise ValueError(f"Invalid trainer module name: {module!r}")
        return module

    def _check_run_dir(self, run_dir: str) -> str:
        rd = os.path.abspath(run_dir)
        if os.path.commonpath([self.experiments_root, rd]) != self.experiments_root:
            raise ValueError(f"Run dir escapes experiments root: {run_dir!r}")
        return rd

    # ------------------------------------------------------------ launch

    def launch(
        self,
        manifest: ExperimentManifest,
        experiment_dir: str,
        trainer: str = "ppo",
        env_mode: str = "inprocess",
        run_id: Optional[str] = None,
        resume_from: Optional[Dict[str, Any]] = None,
        run_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Creates a run, materializes the trainer contract, launches the
        external trainer subprocess, and returns run_id.

        `run_overrides` applies per-run (never per-experiment) variations —
        used by the batch scheduler: {"seed", "scenario_dict",
        "algorithm_config"}. The manifest itself is never mutated.
        """
        run_overrides = dict(run_overrides or {})
        run_seed = int(run_overrides.get("seed", manifest.random_seed))
        module = self._resolve_trainer(trainer)
        experiment_dir = os.path.abspath(experiment_dir)

        # Algorithm/environment compatibility is checked BEFORE any process
        # launch — incompatible combinations are configuration errors.
        compat_errors = check_compatibility(manifest, trainer)
        if compat_errors:
            raise ValueError(
                "Incompatible experiment/trainer combination: "
                + "; ".join(compat_errors)
            )

        # Curriculum is a deterministic configuration surface: invalid
        # definitions and unsupported env modes fail before any process or
        # run directory is created (non-retryable configuration errors).
        if manifest.curriculum_configuration:
            curr_errors = validate_curriculum(
                manifest.curriculum_configuration,
                known_scenario_ids=[manifest.scenario_id],
            )
            if curr_errors:
                raise ValueError("Invalid curriculum: " + "; ".join(curr_errors))
            # Curriculum works in every env_mode: inprocess rebuilds stage
            # envs, process workers re-set their scenario, and TCP sims
            # running protocol >= 2.1 accept SET_SCENARIO updates.

        rmg = self.run_manager

        if run_id is None:
            run = rmg.create_run(
                experiment_dir,
                seed=run_seed,
                experiment_id=manifest.experiment_id,
                trainer=module,
                env_mode=env_mode,
                num_envs=manifest.training.num_envs,
                resume_from=resume_from,
            )
            run_id = run.run_id
        rd = self._check_run_dir(rmg.run_dir(experiment_dir, run_id))

        # TCP modes: spawn headless simulator workers first. 'tcp' spawns
        # one process per env; 'tcp_multi' spawns a single process hosting
        # all envs on one port (per-client env binding).
        tcp_ports: List[int] = []
        if env_mode in ("tcp", "tcp_multi"):
            pool = HeadlessSimProcessPool(
                manifest.training.num_envs,
                shared_process=(env_mode == "tcp_multi"))
            tcp_ports = pool.start()
            self._sim_pools[rd] = pool

        contract = build_contract(
            manifest, experiment_dir, rd, run_id,
            env_mode=env_mode, tcp_ports=tcp_ports, resume=resume_from,
        )
        # Per-run overrides (batch scheduler): seed / scenario / alg config.
        contract["seed"] = run_seed
        if run_overrides.get("scenario_dict"):
            contract["scenario"] = run_overrides["scenario_dict"]
            contract["scenario_id"] = run_overrides["scenario_dict"].get(
                "scenario_id", contract["scenario_id"])
        if run_overrides.get("algorithm_config"):
            contract["training"]["algorithm_config"] = {
                **contract["training"].get("algorithm_config", {}),
                **run_overrides["algorithm_config"],
            }
        errors = validate_contract(contract)
        if errors:
            self._cleanup_sim_pool(rd)
            raise ValueError("Invalid trainer contract: " + "; ".join(errors))

        with open(os.path.join(rd, "contract.json"), "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2)

        rmg.set_status(experiment_dir, run_id, RunStatus.QUEUED)
        rmg.set_status(experiment_dir, run_id, RunStatus.STARTING)

        stdout_log = open(os.path.join(rd, "logs", "stdout.log"), "w", encoding="utf-8")
        stderr_log = open(os.path.join(rd, "logs", "stderr.log"), "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [self.python_exe, "-m", module, "--run-dir", rd],
                cwd=self.repo_root,
                stdout=stdout_log,
                stderr=stderr_log,
            )
        except Exception as e:
            stdout_log.close()
            stderr_log.close()
            self._cleanup_sim_pool(rd)
            rmg.set_status(experiment_dir, run_id, RunStatus.FAILED,
                           error={"type": "launch_failure", "message": str(e)})
            raise

        self._procs[rd] = proc
        rmg.set_status(experiment_dir, run_id, RunStatus.RUNNING, pid=proc.pid)
        return run_id

    # ------------------------------------------------------------ monitoring

    def poll(self, experiment_dir: str, run_id: str) -> Dict[str, Any]:
        """
        Monitors a run: syncs latest metrics/progress/checkpoint refs from
        contract artifacts; finalizes status when the process exits.
        Returns the run summary dict.
        """
        rmg = self.run_manager
        experiment_dir = os.path.abspath(experiment_dir)
        run = rmg.load_run(experiment_dir, run_id)
        rd = rmg.run_dir(experiment_dir, run_id)

        if run.status in RunStatus.TERMINAL:
            return run.to_dict()

        # Sync metrics tail -> run progress
        metrics_path = os.path.join(rd, "metrics.jsonl")
        reader = MetricsReader(metrics_path)
        latest = reader.latest()
        episodes = reader.by_scope("episode")
        if latest:
            rmg.update_progress(
                experiment_dir, run_id,
                timestep=latest.get("timestep", run.current_timestep),
                episode=len(episodes),
                latest_metrics=latest.get("metrics", {}),
            )

        # Sync artifact registry -> run refs
        registry = ArtifactRegistry(rd)
        for entry in registry.entries("checkpoint"):
            rmg.add_artifact_ref(experiment_dir, run_id, "checkpoint", entry["path"])
        for entry in registry.entries("evaluation"):
            rmg.add_artifact_ref(experiment_dir, run_id, "evaluation", entry["path"])
        for entry in registry.entries("replay"):
            rmg.add_artifact_ref(experiment_dir, run_id, "replay", entry["path"])

        # Wall-clock timeout enforcement
        run = rmg.load_run(experiment_dir, run_id)
        proc = self._procs.get(rd)
        manifest_max_wall = self._load_training_max_wall(experiment_dir)
        if (manifest_max_wall > 0 and run.started_at
                and time.time() - run.started_at > manifest_max_wall):
            self._terminate_proc(rd)
            self._cleanup_sim_pool(rd)
            rmg.set_status(experiment_dir, run_id, RunStatus.INTERRUPTED,
                           error={"type": "timeout",
                                  "message": f"Exceeded max_wall_seconds={manifest_max_wall}"},
                           exit_code=-1)
            return rmg.load_run(experiment_dir, run_id).to_dict()

        # Process exit detection
        if proc is not None:
            code = proc.poll()
            if code is not None:
                self._finalize_run(experiment_dir, run_id, rd, code)
        else:
            # Unknown process (e.g. orchestrator restarted): mark INTERRUPTED
            # only when metrics stop progressing AND run.json is stale — for
            # now, leave RUNNING state as-is; callers re-poll or cancel.
            pass

        return rmg.load_run(experiment_dir, run_id).to_dict()

    def wait(self, experiment_dir: str, run_id: str, timeout_s: float = 600.0, poll_s: float = 0.25) -> Dict[str, Any]:
        """Polls until the run reaches a terminal status or timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            summary = self.poll(experiment_dir, run_id)
            if summary["status"] in RunStatus.TERMINAL:
                return summary
            time.sleep(poll_s)
        raise TimeoutError(f"Run {run_id} did not reach a terminal state in {timeout_s}s")

    def cancel(self, experiment_dir: str, run_id: str) -> Dict[str, Any]:
        """Cancels a live run: graceful terminate, kill escalation."""
        rmg = self.run_manager
        experiment_dir = os.path.abspath(experiment_dir)
        run = rmg.load_run(experiment_dir, run_id)
        rd = rmg.run_dir(experiment_dir, run_id)
        if run.status in RunStatus.TERMINAL:
            return run.to_dict()

        self._terminate_proc(rd)
        self._cleanup_sim_pool(rd)
        rmg.set_status(experiment_dir, run_id, RunStatus.CANCELLED)
        return rmg.load_run(experiment_dir, run_id).to_dict()

    # ------------------------------------------------------------ internals

    def _terminate_proc(self, rd: str, grace_s: float = 5.0) -> None:
        proc = self._procs.get(rd)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=grace_s)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._procs.pop(rd, None)

    def _cleanup_sim_pool(self, rd: str) -> None:
        pool = self._sim_pools.pop(rd, None)
        if pool is not None:
            pool.stop()

    def _finalize_run(self, experiment_dir: str, run_id: str, rd: str, exit_code: int) -> None:
        rmg = self.run_manager
        self._cleanup_sim_pool(rd)
        self._procs.pop(rd, None)

        result_path = os.path.join(rd, "run_result.json")
        result: Dict[str, Any] = {}
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except json.JSONDecodeError:
                result = {}

        status = result.get("status")
        if status == "completed":
            rmg.set_status(experiment_dir, run_id, RunStatus.COMPLETED, exit_code=exit_code)
        elif status in (rmg_status for rmg_status in (RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.INTERRUPTED)):
            rmg.set_status(experiment_dir, run_id, status,
                           error=result.get("error"), exit_code=exit_code)
        elif exit_code == 0:
            rmg.set_status(experiment_dir, run_id, RunStatus.COMPLETED, exit_code=exit_code)
        else:
            rmg.set_status(
                experiment_dir, run_id, RunStatus.FAILED,
                error=result.get("error") or {
                    "type": "trainer_crash",
                    "message": f"Trainer exited with code {exit_code} without run_result.json",
                },
                exit_code=exit_code,
            )

    def _load_training_max_wall(self, experiment_dir: str) -> float:
        manifest_path = os.path.join(experiment_dir, "experiment.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return float(json.load(f).get("training", {}).get("max_wall_seconds", 0.0))
        except (OSError, json.JSONDecodeError):
            return 0.0
