"""
Trainer Capability Declaration + Compatibility Validation.

Every trainer declares what it can actually run. The experiment platform
validates compatibility BEFORE launching a process — an incompatible
combination (e.g. DQN + continuous action space) is a deterministic
configuration error, never a mid-training crash.

Capabilities are deliberately declarative data so remote workers can
advertise the same shape in a later phase.
"""

from __future__ import annotations
from typing import Dict, Any, List

# Trainer short-name -> capability block.
#   algorithms          supported training.algorithm values
#   action_types        "continuous" | "discrete"
#   observation_types   "vector" | "image"
#   multi_env           supports num_envs > 1
#   recurrent           supports recurrent policies (none currently)
#   checkpoint_format   "torch" | "raw"
#   evaluation          can produce evaluation artifacts
TRAINER_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "algorithms": ["ppo"],
        "action_types": ["continuous"],
        "observation_types": ["vector"],
        "multi_env": True,
        "recurrent": False,
        "checkpoint_format": "torch",
        "evaluation": True,
    },
    "sac": {
        "algorithms": ["sac"],
        "action_types": ["continuous"],
        "observation_types": ["vector"],
        "multi_env": True,
        "recurrent": False,
        "checkpoint_format": "torch",
        "evaluation": True,
    },
    "dqn": {
        "algorithms": ["dqn"],
        "action_types": ["discrete"],
        "observation_types": ["vector"],
        "multi_env": True,
        "recurrent": False,
        "checkpoint_format": "torch",
        "evaluation": True,
    },
    "bc": {
        "algorithms": ["bc"],
        "action_types": ["continuous", "discrete"],
        "observation_types": ["vector"],
        "multi_env": True,
        "recurrent": False,
        "checkpoint_format": "torch",
        "evaluation": True,
    },
    "dummy": {
        "algorithms": ["dummy"],
        "action_types": ["continuous", "discrete"],
        "observation_types": ["vector", "image"],
        "multi_env": True,
        "recurrent": False,
        "checkpoint_format": "raw",
        "evaluation": False,
    },
}


def describe_trainer(name: str) -> Dict[str, Any]:
    """Returns the declared capability block for a trainer (empty if unknown)."""
    return dict(TRAINER_CAPABILITIES.get(name, {}))


def _env_action_type(manifest) -> str:
    """Derives the environment's action type from the manifest action schema."""
    schema = manifest.action_schema or {}
    st = schema.get("space_type") or schema.get("type")
    if st:
        return str(st)
    # Legacy manifests without a declared type default to continuous.
    return "continuous"


def _env_uses_image_obs(manifest) -> bool:
    schema = manifest.observation_schema or {}
    if schema.get("include_image") or schema.get("include_image_channel"):
        return True
    for ch in schema.get("channels", []):
        if ch.get("type") == "image" or ch.get("channel_type") == "image":
            if ch.get("enabled", True):
                return True
    return False


def check_compatibility(manifest, trainer: str) -> List[str]:
    """
    Validates manifest <-> trainer compatibility. Returns problems;
    empty means compatible. Run before any process launch.
    """
    errors: List[str] = []
    cap = TRAINER_CAPABILITIES.get(trainer)
    if cap is None:
        return [f"Unknown trainer '{trainer}' (no capability declaration)"]

    algorithm = manifest.training.algorithm
    if algorithm not in cap["algorithms"]:
        errors.append(
            f"Trainer '{trainer}' does not support algorithm '{algorithm}' "
            f"(supports: {cap['algorithms']})"
        )

    env_action = _env_action_type(manifest)
    if env_action not in cap["action_types"]:
        errors.append(
            f"Trainer '{trainer}' ({algorithm}) requires a "
            f"{'/'.join(cap['action_types'])} action space; "
            f"environment provides '{env_action}'"
        )

    if _env_uses_image_obs(manifest) and "image" not in cap["observation_types"]:
        errors.append(
            f"Trainer '{trainer}' does not support image observations"
        )

    if int(getattr(manifest.training, "num_envs", 1)) > 1 and not cap["multi_env"]:
        errors.append(f"Trainer '{trainer}' does not support multiple environments")

    return errors
