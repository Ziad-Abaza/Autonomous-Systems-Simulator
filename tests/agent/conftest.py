"""Shared fixtures for the agent laboratory."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that spawn sim processes / long runs")

TRACKS = {
    "oval": REPO_ROOT / "tracks" / "basic_driving_proving_ground.sim.json",
    "serpentine": REPO_ROOT / "tracks" / "lane_following_serpentine_circuit.sim.json",
    "smoke": REPO_ROOT / "tracks" / "smoke_test.sim.json",
}


@pytest.fixture(scope="session")
def tracks() -> dict:
    return TRACKS


@pytest.fixture()
def base_seed() -> int:
    return 42
