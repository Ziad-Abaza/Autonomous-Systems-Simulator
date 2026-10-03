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
    """Materializes the trainer contract for one run.

    All paths are written *relative to run_dir* so the contract (and the
    whole run directory) stays portable — exports and copies never carry
    stale absolute paths from the machine that created them.
    """
    run_dir = os.path.abspath(run_dir)

    def _rel(path: str) -> str:
        return os.path.relpath(os.path.abspath(path), run_dir).replace(
            os.sep, "/")

    experiment_dir = os.path.abspath(experiment_dir)
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
            "experiment_dir": _rel(experiment_dir),
            "run_dir": ".",
            "environment_json": _rel(os.path.join(experiment_dir, "environment.json")),
            "scenario_json": _rel(os.path.join(experiment_dir, "scenario.json")),
            "metrics_file": "metrics.jsonl",
            "checkpoints_dir": "checkpoints",
            "evaluation_dir": "evaluation",
            "trajectories_dir": "trajectories",
            "replays_dir": "replays",
            "logs_dir": "logs",
            "run_result": "run_result.json",
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
    if env_mode not in ("inprocess", "tcp", "tcp_multi", "process"):
        errors.append(f"unknown env_mode: {env_mode!r}")
    if env_mode in ("tcp", "tcp_multi"):
        tcp = contract.get("tcp", {})
        if not tcp.get("ports"):
            errors.append(f"env_mode '{env_mode}' requires tcp.ports")
        # Curriculum over TCP requires simulators running protocol >= 2.1
        # (SET_SCENARIO). The contract no longer rejects the combination;
        # the runtime fails explicitly if a simulator doesn't support it.
    return errors


def resolve_contract_paths(contract: Dict[str, Any],
                           run_dir: Optional[str] = None) -> Dict[str, Any]:
    """Resolves the contract's run-relative paths to absolute paths.

    ``run_dir`` defaults to the directory containing contract.json's own
    ``paths.run_dir`` resolved against the caller — pass the directory
    that holds contract.json. Both relative (portable, current format)
    and absolute (legacy) path values are handled.
    """
    base = os.path.abspath(run_dir) if run_dir else os.getcwd()
    paths = dict(contract.get("paths", {}))
    resolved: Dict[str, str] = {}
    for key, val in paths.items():
        if isinstance(val, str):
            resolved[key] = val if os.path.isabs(val) \
                else os.path.abspath(os.path.join(base, val))
        else:
            resolved[key] = val
    out = dict(contract)
    out["paths"] = resolved
    return out
