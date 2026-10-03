"""
Reproducibility verification.

Given an experiment directory, verifies that everything needed to
reproduce the experiment is present and internally consistent:

- experiment.json loads and validates
- environment.json snapshot fingerprint matches the recorded fingerprint
- scenario.json matches the manifest's scenario_configuration
- observation/action schemas are present and non-empty
- simulator version is known to this build
- protocol version is supported by this build

It reports honest status: an experiment whose environment snapshot or
versions cannot be verified is reported as such — never silently assumed.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, List

from sim_version import SIMULATOR_VERSION
from sim_net.protocol import SUPPORTED_PROTOCOL_VERSIONS
from sim_env.versioning import EnvironmentVersionManager
from sim_experiment.manifest import ExperimentManifest, MANIFEST_VERSION

KNOWN_SIMULATOR_VERSIONS = {SIMULATOR_VERSION}


def check_reproducibility(experiment_dir: str) -> Dict[str, Any]:
    """Runs all reproducibility checks; returns a structured report."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "", fatal: bool = False) -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "fatal": fatal})
        return ok

    manifest_path = os.path.join(experiment_dir, "experiment.json")
    if not check("experiment.json exists", os.path.exists(manifest_path), manifest_path, fatal=True):
        return _report(checks)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = ExperimentManifest.from_dict(json.load(f))
        check("manifest parses", True)
    except Exception as e:
        check("manifest parses", False, str(e), fatal=True)
        return _report(checks)

    check("manifest_version supported",
          manifest.manifest_version == MANIFEST_VERSION,
          f"{manifest.manifest_version} (build supports {MANIFEST_VERSION})")

    env_path = os.path.join(experiment_dir, "environment.json")
    if check("environment.json exists", os.path.exists(env_path), fatal=True):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                env_dict = json.load(f)
            fp = EnvironmentVersionManager.compute_fingerprint(env_dict)
            check("environment fingerprint matches manifest",
                  fp == manifest.environment_fingerprint,
                  f"computed {fp[:12]} vs recorded {manifest.environment_fingerprint[:12]}",
                  fatal=True)
        except Exception as e:
            check("environment fingerprint matches manifest", False, str(e), fatal=True)

    scen_path = os.path.join(experiment_dir, "scenario.json")
    if check("scenario.json exists", os.path.exists(scen_path)):
        try:
            with open(scen_path, "r", encoding="utf-8") as f:
                scen_dict = json.load(f)
            check("scenario matches manifest",
                  scen_dict == manifest.scenario_configuration,
                  "scenario.json vs manifest.scenario_configuration")
        except Exception as e:
            check("scenario matches manifest", False, str(e))

    check("observation schema present", bool(manifest.observation_schema.get("channels")))
    check("action schema present", bool(manifest.action_schema.get("channels")))
    check("termination config present", bool(manifest.termination_configuration.get("rules")))
    check("episode config present", bool(manifest.episode_configuration))

    check("simulator version known",
          manifest.simulator_version in KNOWN_SIMULATOR_VERSIONS,
          f"{manifest.simulator_version} (build knows {sorted(KNOWN_SIMULATOR_VERSIONS)})")
    check("protocol version supported",
          manifest.protocol_version in SUPPORTED_PROTOCOL_VERSIONS,
          f"{manifest.protocol_version} (supports {list(SUPPORTED_PROTOCOL_VERSIONS)})")

    check("training config present",
          bool(manifest.training.algorithm) and manifest.training.total_timesteps > 0,
          f"algorithm={manifest.training.algorithm}")

    return _report(checks)


def _report(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    fatal_failures = [c for c in checks if not c["ok"] and c["fatal"]]
    soft_failures = [c for c in checks if not c["ok"] and not c["fatal"]]
    return {
        "reproducible": not fatal_failures and not soft_failures,
        "exact_reproduction_guaranteed": False,  # external factors (torch nondeterminism, HW)
        "checks": checks,
        "fatal_failures": [c["name"] for c in fatal_failures],
        "warnings": [c["name"] for c in soft_failures],
    }
