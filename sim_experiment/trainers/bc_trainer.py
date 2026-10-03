"""
BC external trainer entrypoint. Usage:

    python -m sim_experiment.trainers.bc_trainer --run-dir <run_dir>

Behavior Cloning over transitions_v1 datasets. Reads contract.json
(bc.dataset_dir supplied by the launcher), validates dataset compatibility
against the contract's environment fingerprint, trains a supervised policy
on the deterministic train split, evaluates it on the contract environment,
and writes metrics.jsonl / checkpoints / run_result.json.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sim_experiment.metrics import MetricsWriter
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trainers._harness import (
    write_result, load_contract, build_eval_env, obs_to_vec,
    action_bounds, env_action_type, run_periodic_eval,
    resolve_resume_checkpoint,
)


def run_training(run_dir: str) -> int:
    contract, err = load_contract(run_dir)
    if contract is None:
        write_result(run_dir, "failed", error={"type": "invalid_contract", "message": err})
        return 2

    training = contract["training"]
    paths = contract["paths"]
    seed = int(contract["seed"])

    if training.get("algorithm") != "bc":
        write_result(run_dir, "failed",
                     error={"type": "unsupported_algorithm",
                            "message": f"bc_trainer cannot run algorithm "
                                       f"'{training.get('algorithm')}'"})
        return 2

    dataset_dir = (contract.get("bc") or {}).get("dataset_dir")
    if not dataset_dir:
        write_result(run_dir, "failed",
                     error={"type": "invalid_contract",
                            "message": "algorithm 'bc' requires bc.dataset_dir "
                                       "in the trainer contract"})
        return 2
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.abspath(dataset_dir)

    alg = training.get("algorithm_config", {})
    epochs = int(alg.get("bc_epochs", 20))
    batch_size = int(alg.get("batch_size", training.get("batch_size", 256)))
    lr = float(alg.get("bc_lr", training.get("learning_rate", 3e-4)))
    hidden = tuple(alg.get("hidden_sizes", (64, 64)))
    val_ratio = float(alg.get("val_ratio", 0.2))
    action_mode = "discrete" if env_action_type(contract) == "discrete" else "continuous"

    if action_mode == "discrete":
        num_actions = int(alg.get("num_actions",
                                  (contract.get("action_schema") or {}).get(
                                      "num_actions", 5)))
        act_dim = num_actions
    else:
        low, _ = action_bounds(contract)
        act_dim = int(low.shape[0])

    writer = MetricsWriter(paths["metrics_file"])
    registry = ArtifactRegistry(run_dir)

    from sim_experiment.bc.bc_data import load_transitions, BCDataError
    from sim_experiment.bc.bc_runner import BCRunner

    # Compatibility gates: dataset structure + fingerprint + dims are
    # validated BEFORE any training work — mismatches are config errors.
    try:
        full = load_transitions(
            dataset_dir, split="all", seed=seed,
            expected_env_fingerprint=contract["environment_fingerprint"])
        obs_dim = int(full["obs"].shape[1])
        train_data = load_transitions(
            dataset_dir, split="train", seed=seed, obs_dim=obs_dim,
            expected_env_fingerprint=contract["environment_fingerprint"],
            ratios={"train": 1.0 - val_ratio, "val": val_ratio, "test": 0.0})
        val_data = None
        try:
            val_data = load_transitions(
                dataset_dir, split="val", seed=seed, obs_dim=obs_dim,
                ratios={"train": 1.0 - val_ratio, "val": val_ratio, "test": 0.0})
        except BCDataError:
            pass  # tiny datasets may produce an empty val split
    except BCDataError as e:
        write_result(run_dir, "failed",
                     error={"type": "dataset_incompatible", "message": str(e)})
        return 2

    resume_ckpt = resolve_resume_checkpoint(contract)
    runner = BCRunner(
        obs_dim=obs_dim, act_dim=act_dim, action_mode=action_mode,
        hidden=hidden, lr=lr, batch_size=batch_size, seed=seed,
        resume_checkpoint=resume_ckpt)

    start = time.perf_counter()

    def on_epoch(rec: Dict[str, Any]) -> None:
        writer.write("bc", timestep=int(rec["epoch"]), metrics={
            "epoch": int(rec["epoch"]),
            "train_loss": float(rec["train_loss"]),
            "val_loss": rec["val_loss"],
        })

    metrics = runner.train_epochs(train_data, val_data, epochs=epochs,
                                  on_epoch=on_epoch)
    wall = time.perf_counter() - start

    final_ckpt = os.path.join(paths["checkpoints_dir"], "policy_final.pt")
    runner.save_checkpoint(final_ckpt, epoch=runner.epoch_offset, extra={
        "dataset_dir": dataset_dir,
        "transitions": metrics["transitions"],
        "final_train_loss": metrics["final_train_loss"],
        "final_val_loss": metrics["final_val_loss"],
    })
    registry.register("checkpoint", os.path.relpath(final_ckpt, run_dir),
                      step=runner.epoch_offset,
                      metadata={"algorithm": "bc", "final": True,
                                "experiment_id": contract["experiment_id"],
                                "run_id": contract["run_id"],
                                "env_fingerprint": contract["environment_fingerprint"]})

    # Frozen-policy evaluation on the contract environment (in-process so
    # replay capture and deterministic resets are available).
    eval_env = None
    try:
        if contract["evaluation"].get("enabled", True):
            eval_env = build_eval_env(contract)
            act_fn = runner.eval_action_fn()
            run_periodic_eval(
                contract, eval_env, act_fn, runner.epoch_offset,
                paths, registry, writer, algorithm="bc")
    finally:
        if eval_env is not None and hasattr(eval_env, "close"):
            eval_env.close()

    writer.write("run", timestep=runner.epoch_offset, metrics={
        "total_timesteps": runner.epoch_offset,
        "epochs": runner.epoch_offset,
        "transitions": metrics["transitions"],
        "final_train_loss": metrics["final_train_loss"],
        "final_val_loss": metrics["final_val_loss"],
        "wall_clock_time": round(wall, 2),
    })
    writer.close()

    write_result(run_dir, "completed",
                 total_timesteps=runner.epoch_offset,
                 episodes_completed=0,
                 wall_clock_time=round(wall, 2),
                 sps=round(metrics["transitions"] / max(1e-4, wall), 1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return run_training(parser.parse_args().run_dir)


if __name__ == "__main__":
    sys.exit(main())
