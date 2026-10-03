"""
Batch experiment expansion.

Deterministic parameter expansion: creates multiple run specifications
from one experiment by varying seed and/or scenario. This is deliberately
NOT a hyperparameter optimization engine — expansion is a simple,
inspectable cross product.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from sim_experiment.manifest import ExperimentManifest


def expand_run_specs(
    manifest: ExperimentManifest,
    seeds: Optional[List[int]] = None,
    scenario_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic cross product of seeds × scenario_ids.
    Empty/None axes use the manifest's own values (single element).
    Returns one spec per run: {"seed": s, "scenario_id": sc}.
    """
    seed_list = list(seeds) if seeds else [manifest.random_seed]
    scen_list = list(scenario_ids) if scenario_ids else [manifest.scenario_id]
    specs: List[Dict[str, Any]] = []
    for sc in scen_list:
        for s in seed_list:
            specs.append({"seed": int(s), "scenario_id": sc})
    return specs
