"""
Experiment Manager.

Application/domain service owning experiment persistence under an
experiments root directory. The UI and CLI call this service; they never
manipulate manifest files directly.

Directory layout:

    experiments/
    └── <experiment_id>/
        ├── experiment.json      # immutable after launch
        ├── environment.json     # full environment snapshot
        ├── scenario.json        # scenario snapshot
        └── runs/
            └── <run_id>/        # managed by RunManager
"""

from __future__ import annotations
import json
import os
import shutil
from typing import Dict, Any, List, Optional

from sim_experiment.manifest import ExperimentManifest
from sim_env.versioning import EnvironmentVersionManager


def default_experiments_root() -> str:
    """Repository experiments root — resolved from this file's location so
    callers are independent of the process working directory."""
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments")


class ExperimentManager:
    """CRUD + lifecycle service for experiment manifests."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir or default_experiments_root())
        os.makedirs(self.root_dir, exist_ok=True)

    # ------------------------------------------------------------ paths

    def experiment_dir(self, experiment_id: str) -> str:
        if not experiment_id or os.path.basename(experiment_id) != experiment_id:
            raise ValueError(f"Invalid experiment_id: {experiment_id!r}")
        return os.path.join(self.root_dir, experiment_id)

    def _manifest_path(self, experiment_id: str) -> str:
        return os.path.join(self.experiment_dir(experiment_id), "experiment.json")

    # ------------------------------------------------------------ CRUD

    def validate(self, manifest: ExperimentManifest) -> List[str]:
        """Returns a list of structural problems; empty means valid."""
        errors: List[str] = []
        if not manifest.experiment_id:
            errors.append("experiment_id is empty")
        if not manifest.environment:
            errors.append("manifest has no environment snapshot")
        if not manifest.environment_fingerprint:
            errors.append("environment_fingerprint is empty")
        else:
            fp = EnvironmentVersionManager.compute_fingerprint(manifest.environment)
            if fp != manifest.environment_fingerprint:
                errors.append(
                    "environment_fingerprint does not match environment snapshot"
                )
        expected_fp = manifest._compute_fingerprint()
        if manifest.experiment_fingerprint and manifest.experiment_fingerprint != expected_fp:
            errors.append("experiment_fingerprint does not match identity payload")
        if not manifest.training.algorithm:
            errors.append("training.algorithm is empty")
        if manifest.training.total_timesteps <= 0:
            errors.append("training.total_timesteps must be positive")
        return errors

    def create(self, manifest: ExperimentManifest) -> str:
        """
        Persists a new experiment (manifest + environment + scenario snapshots).
        Rejects invalid manifests and re-creation of a launched experiment.
        Returns the experiment directory.
        """
        errors = self.validate(manifest)
        if errors:
            raise ValueError("Invalid experiment manifest: " + "; ".join(errors))

        exp_dir = self.experiment_dir(manifest.experiment_id)
        existing = self._try_load(manifest.experiment_id)
        if existing is not None and existing.launched:
            raise RuntimeError(
                f"Experiment {manifest.experiment_id} is immutable (already launched)."
            )
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "runs"), exist_ok=True)

        self._write_json(self._manifest_path(manifest.experiment_id), manifest.to_dict())
        self._write_json(os.path.join(exp_dir, "environment.json"), manifest.environment)
        self._write_json(os.path.join(exp_dir, "scenario.json"), manifest.scenario_configuration)
        return exp_dir

    def save(self, manifest: ExperimentManifest) -> None:
        """Saves manifest metadata changes (allowed only before launch)."""
        existing = self.load(manifest.experiment_id)
        if existing.launched:
            raise RuntimeError(
                f"Experiment {manifest.experiment_id} is immutable (already launched)."
            )
        errors = self.validate(manifest)
        if errors:
            raise ValueError("Invalid experiment manifest: " + "; ".join(errors))
        self._write_json(self._manifest_path(manifest.experiment_id), manifest.to_dict())

    def _try_load(self, experiment_id: str) -> Optional[ExperimentManifest]:
        path = self._manifest_path(experiment_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return ExperimentManifest.from_dict(json.load(f))

    def load(self, experiment_id: str) -> ExperimentManifest:
        m = self._try_load(experiment_id)
        if m is None:
            raise FileNotFoundError(f"No experiment '{experiment_id}' in {self.root_dir}")
        return m

    def list_experiments(self, include_archived: bool = True) -> List[Dict[str, Any]]:
        """Lightweight summaries of all experiments (no full env snapshots)."""
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(self.root_dir):
            return out
        for entry in sorted(os.listdir(self.root_dir)):
            path = self._manifest_path(entry)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Legacy pre-manifest experiment.json files (written by the
            # standalone training scripts, not the sim_experiment manager)
            # have no manifest_version — skip them so they don't appear as
            # unloadable phantom rows in the UI.
            if "manifest_version" not in d:
                continue
            if d.get("archived") and not include_archived:
                continue
            out.append({
                "experiment_id": d.get("experiment_id", entry),
                "name": d.get("name", ""),
                "environment_name": d.get("environment_name", ""),
                "environment_version": d.get("environment_version", ""),
                "environment_fingerprint": d.get("environment_fingerprint", ""),
                "scenario_id": d.get("scenario_id", ""),
                "algorithm": d.get("training", {}).get("algorithm", ""),
                "random_seed": d.get("random_seed", 0),
                "created_at": d.get("created_at", 0.0),
                "launched": d.get("launched", False),
                "archived": d.get("archived", False),
            })
        return out

    def clone(self, experiment_id: str) -> ExperimentManifest:
        """Returns an unlaunched, unsaved copy suitable for editing."""
        clone = self.load(experiment_id)
        clone.launched = False
        clone.archived = False
        return clone

    def mark_launched(self, experiment_id: str) -> None:
        """Marks the experiment immutable once its first run launches."""
        m = self.load(experiment_id)
        if m.launched:
            return
        m.launched = True
        self._write_json(self._manifest_path(experiment_id), m.to_dict())

    def archive(self, experiment_id: str) -> None:
        """Archives an experiment (metadata flag; files are preserved)."""
        m = self.load(experiment_id)
        m.archived = True
        # archive flag is lifecycle metadata, safe to update post-launch
        self._write_json(self._manifest_path(experiment_id), m.to_dict())

    def export(self, experiment_id: str, dest_dir: str) -> str:
        """Copies the complete self-describing experiment directory."""
        src = self.experiment_dir(experiment_id)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"No experiment '{experiment_id}'")
        os.makedirs(os.path.dirname(os.path.abspath(dest_dir)) or ".", exist_ok=True)
        if os.path.abspath(src) == os.path.abspath(dest_dir):
            raise ValueError("Export destination equals source directory")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src, dest_dir)
        return dest_dir

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic on Windows + POSIX
