"""
Dummy contract-compliant trainer used for orchestration testing.
Simulates a training loop: emits episode/update metrics, writes a
checkpoint + registry entry, and finishes with run_result.json.

Controlled via contract training fields:
  total_timesteps        -> number of simulated steps
  algorithm_config:
    tick_seconds         -> sleep per simulated step (default 0.0)
    fail_at_step         -> raise exception at this step (tests failure path)
    emit_checkpoint_at   -> write checkpoint at this step (default: end)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sim_experiment.metrics import MetricsWriter
from sim_experiment.artifacts import ArtifactRegistry
from sim_experiment.trainer_contract import validate_contract, resolve_contract_paths


def _write_result(run_dir: str, status: str, **kwargs) -> None:
    result = {"status": status, **kwargs}
    path = os.path.join(run_dir, "run_result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    rd = args.run_dir

    try:
        with open(os.path.join(rd, "contract.json"), "r", encoding="utf-8") as f:
            contract = json.load(f)
        errors = validate_contract(contract)
        if errors:
            _write_result(rd, "failed", error={"type": "invalid_contract", "message": "; ".join(errors)})
            return 2
        contract = resolve_contract_paths(contract, rd)

        training = contract["training"]
        total = int(training.get("total_timesteps", 100))
        cfg = training.get("algorithm_config", {})
        tick = float(cfg.get("tick_seconds", 0.0))
        fail_at = cfg.get("fail_at_step")
        ckpt_at = int(cfg.get("emit_checkpoint_at", total))

        paths = contract["paths"]
        writer = MetricsWriter(paths["metrics_file"], buffer_size=1)
        registry = ArtifactRegistry(rd)

        for step in range(1, total + 1):
            if fail_at is not None and step == int(fail_at):
                raise RuntimeError(f"Simulated trainer failure at step {step}")
            writer.write("step", timestep=step, metrics={"progress": step / total})
            if step % 10 == 0:
                writer.write("episode", timestep=step,
                             metrics={"reward": float(step) / 10.0, "length": 10})
            if step == ckpt_at:
                ckpt_path = os.path.join(paths["checkpoints_dir"], f"dummy_{step}.ckpt")
                with open(ckpt_path, "wb") as f:
                    f.write(b"dummy weights")
                registry.register("checkpoint", os.path.relpath(ckpt_path, rd),
                                  step=step, metadata={"algorithm": "dummy"})
            if tick > 0:
                time.sleep(tick)

        writer.write("run", timestep=total, metrics={"done": 1.0})
        writer.close()
        _write_result(rd, "completed", total_timesteps=total)
        return 0

    except Exception as e:
        _write_result(rd, "failed",
                      error={"type": "trainer_exception",
                             "message": str(e),
                             "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    sys.exit(main())
