"""TrackRegistry — named TrackSpecs from files and generators.

File tracks come from tracks/*.sim.json. Generated tracks (gen_*) are
parametric closed loops built by track_gen — deterministic per seed and
usable as guaranteed-unseen holdouts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentRL.envs import track_gen

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS_DIR = REPO_ROOT / "tracks"

FILE_TRACKS: dict[str, str] = {
    "oval": "basic_driving_proving_ground.sim.json",
    "serpentine": "lane_following_serpentine_circuit.sim.json",
    "smoke": "smoke_test.sim.json",
}


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    project: dict[str, Any]
    tags: tuple[str, ...] = ()


class TrackRegistry:
    """Registry of file-backed and generated tracks."""

    def __init__(self) -> None:
        self._file: dict[str, Path] = {}
        self._gen: dict[str, Callable[[], dict[str, Any]]] = {}
        self._tags: dict[str, tuple[str, ...]] = {}

    @classmethod
    def default(cls) -> "TrackRegistry":
        reg = cls()
        for tid, fname in FILE_TRACKS.items():
            path = TRACKS_DIR / fname
            if not path.exists():
                continue  # track file removed from the repo — skip
            reg.register_file(tid, path,
                              tags=("train",) if tid != "smoke" else ("smoke",))
        if "smoke" not in reg._file:
            # authored smoke track was removed upstream; the generated
            # open straight keeps the "smoke" tag usable for tests
            reg.register_generated(
                "smoke",
                lambda: track_gen.project_for_road(
                    track_gen.gen_straight(), project_name="smoke"),
                tags=("smoke",))
        # Generated closed loops — distinct geometries for continual
        # phases and guaranteed-unseen holdouts.
        for i in range(6):
            seed = 10_000 + i
            reg.register_generated(
                f"gen_loop_{i}",
                lambda s=seed, i=i: track_gen.project_for_road(
                    track_gen.gen_loop(seed=s, name=f"gen_loop_{i}"),
                    project_name=f"gen_loop_{i}"),
                tags=("train",) if i < 4 else ("holdout",))
        return reg

    def register_file(self, track_id: str, path: Path,
                      tags: tuple[str, ...] = ("train",)) -> None:
        if not Path(path).exists():
            raise KeyError(f"track file not found: {path}")
        self._file[track_id] = Path(path)
        self._tags[track_id] = tags

    def register_generated(self, track_id: str,
                           builder: Callable[[], dict[str, Any]],
                           tags: tuple[str, ...] = ("generated",)) -> None:
        self._gen[track_id] = builder
        self._tags[track_id] = tags

    def list(self) -> list[str]:
        return sorted(set(self._file) | set(self._gen))

    def load(self, track_id: str) -> TrackSpec:
        if track_id in self._file:
            with open(self._file[track_id], encoding="utf-8") as f:
                return TrackSpec(track_id, json.load(f),
                                 self._tags[track_id])
        if track_id in self._gen:
            return TrackSpec(track_id, self._gen[track_id](),
                             self._tags[track_id])
        raise KeyError(f"unknown track {track_id!r}; known: {self.list()}")
