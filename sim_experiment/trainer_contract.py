"""
External Trainer Contract (versioned).

The contract is the machine-readable interface materialized into a run
directory before the external trainer process launches. The trainer reads
contract.json to discover the environment, schemas, seed, training
configuration, and output paths — and reports back through:

    metrics.jsonl          structured metrics stream (see metrics.py)
    checkpoints/           model artifacts + artifacts/registry.jsonl
    evaluation/            evaluation result files
    trajectories/          optional trajectory datasets
    run_result.json        final {"status": "...", ...} written at exit

contract_version allows the orchestrator and trainers to detect
incompatible protocol revisions explicitly.
"""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional

from sim_experiment.manifest import ExperimentManifest
from sim_experiment.curriculum_runtime import curriculum_fingerprint

TRAINER_CONTRACT_VERSION = "1.0"


def build_contract(
    manifest: ExperimentManifest,
    experiment_dir: str,
    run_dir: str,
    run_id: str,
    env_mode: str = "inprocess",
    tcp_ports: Optional[List[int]] = None,
    tcp_host: str = "127.0.0.1",
    resume: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Materializes the trainer contract for one run."""
    run_dir = os.path.abspath(run_dir)
    return {
        "contract_version": TRAINER_CONTRACT_VERSION,
        "experiment_id": manifest.experiment_id,
        "experiment_fingerprint": manifest.experiment_fingerprint,
        "run_id": run_id,
        "environment_name": manifest.environment_name,
        "environment_version": manifest.environment_version,
        "environment_fingerprint": manifest.environment_fingerprint,
        "scenario_id": manifest.scenario_id,
        "scenario": manifest.scenario_configuration,
        "seed": manifest.random_seed,
        "simulator_version": manifest.simulator_version,
        "protocol_version": manifest.protocol_version,
        "observation_schema": manifest.observation_schema,
        "action_schema": manifest.action_schema,
        "curriculum": manifest.curriculum_configuration,
        "curriculum_fingerprint": curriculum_fingerprint(manifest.curriculum_configuration),
        "training": manifest.training.to_dict(),
        "evaluation": manifest.evaluation.to_dict(),
        "env_mode": env_mode,
        "tcp": {"host": tcp_host, "ports": list(tcp_ports or [])},
        "paths": {
            "experiment_dir": os.path.abspath(experiment_dir),
            "run_dir": run_dir,
            "environment_json": os.path.join(os.path.abspath(experiment_dir), "environment.json"),
            "scenario_json": os.path.join(os.path.abspath(experiment_dir), "scenario.json"),
            "metrics_file": os.path.join(run_dir, "metrics.jsonl"),
            "checkpoints_dir": os.path.join(run_dir, "checkpoints"),
            "evaluation_dir": os.path.join(run_dir, "evaluation"),
            "trajectories_dir": os.path.join(run_dir, "trajectories"),
            "replays_dir": os.path.join(run_dir, "replays"),
            "logs_dir": os.path.join(run_dir, "logs"),
            "run_result": os.path.join(run_dir, "run_result.json"),
        },
        "resume": dict(resume) if resume else None,
        "capabilities": ["metrics_jsonl", "checkpoints", "evaluation", "trajectories", "replays"],
    }


_REQUIRED_KEYS = (
    "contract_version", "experiment_id", "run_id", "seed",
    "environment_fingerprint", "training", "evaluation", "paths", "env_mode",
)
_REQUIRED_PATHS = ("run_dir", "environment_json", "metrics_file", "checkpoints_dir")


def validate_contract(contract: Dict[str, Any]) -> List[str]:
    """Structural validation; returns a list of problems (empty = valid)."""
    errors: List[str] = []
    for k in _REQUIRED_KEYS:
        if k not in contract:
            errors.append(f"missing key: {k}")
    if contract.get("contract_version") != TRAINER_CONTRACT_VERSION:
        errors.append(
            f"unsupported contract_version: {contract.get('contract_version')!r} "
            f"(expected {TRAINER_CONTRACT_VERSION})"
        )
    paths = contract.get("paths", {})
    if isinstance(paths, dict):
        for k in _REQUIRED_PATHS:
            if k not in paths:
                errors.append(f"missing paths.{k}")
    else:
        errors.append("paths is not an object")
    env_mode = contract.get("env_mode")
    if env_mode not in ("inprocess", "tcp", "process"):
        errors.append(f"unknown env_mode: {env_mode!r}")
    if env_mode == "tcp":
        tcp = contract.get("tcp", {})
        if not tcp.get("ports"):
            errors.append("env_mode 'tcp' requires tcp.ports")
        # Curriculum over TCP requires simulators running protocol >= 2.1
        # (SET_SCENARIO). The contract no longer rejects the combination;
        # the runtime fails explicitly if a simulator doesn't support it.
    return errors
