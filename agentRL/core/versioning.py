"""Agent + schema versioning. Checkpoints refuse silent cross-version loads."""
from __future__ import annotations

AGENT_VERSION = "0.1.0"
CKPT_SCHEMA_V = 1
OBS_SPEC_V = 1
ACT_SPEC_V = 1


class VersionError(RuntimeError):
    """Raised when a checkpoint/spec version is incompatible with this agent."""


def check_schema(found: int, expected: int, what: str) -> None:
    if found != expected:
        raise VersionError(
            f"{what} schema v{found} is incompatible with agent "
            f"{AGENT_VERSION} (expects v{expected})"
        )
