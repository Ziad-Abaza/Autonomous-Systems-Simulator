"""
Track/Environment Library service.

Treats `*.sim.json` EnvironmentProject files as the studio's document
type. The library is a *browser over directories*, not a second
persistence system — `.sim.json` files remain the single source of
truth. User-facing extras that do not belong in the document schema
(description, favorite flag, last-opened) live in a `.library.json`
sidecar inside the library root.

Library roots:
    user root   — writable, default `<repo>/tracks`
    preset root — bundled read-only templates, `<repo>/presets`

Used by the HOME/library screen; never touched by the runtime sim.
"""
from __future__ import annotations
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sim_project.serializer import EnvironmentProject


@dataclass
class TrackAsset:
    path: str
    file_name: str
    name: str
    description: str = ""
    env_version: str = "1.0.0"
    schema_version: str = "2.0.0"
    point_count: int = 0
    entity_count: int = 0
    is_closed: bool = True
    length_m: float = 0.0
    modified: float = 0.0
    file_size: int = 0
    favorite: bool = False
    last_opened: float = 0.0
    broken: bool = False
    error: str = ""
    readonly: bool = False  # bundled presets


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().lower()
    s = re.sub(r"[\s-]+", "_", s)
    return s or "untitled"


def _approx_length_m(road: Dict[str, Any]) -> float:
    cps = road.get("control_points") or []
    if len(cps) < 2:
        return 0.0
    total = 0.0
    for i in range(len(cps) - 1):
        a, b = cps[i], cps[i + 1]
        total += math.hypot(b.get("x", 0) - a.get("x", 0),
                            b.get("y", 0) - a.get("y", 0))
    if road.get("is_closed"):
        a, b = cps[-1], cps[0]
        total += math.hypot(b.get("x", 0) - a.get("x", 0),
                            b.get("y", 0) - a.get("y", 0))
    return round(total, 1)


class TrackLibrary:
    """Scans and manages `*.sim.json` assets under a root directory."""

    SIDECAR = ".library.json"

    def __init__(self, root: str, readonly: bool = False):
        self.root = os.path.abspath(root)
        self.readonly = readonly
        if not readonly:
            os.makedirs(self.root, exist_ok=True)

    # --------------------------------------------------------- sidecar

    def _sidecar_path(self) -> str:
        return os.path.join(self.root, self.SIDECAR)

    def _load_sidecar(self) -> Dict[str, Any]:
        try:
            with open(self._sidecar_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_sidecar(self, data: Dict[str, Any]) -> None:
        if self.readonly:
            return
        tmp = self._sidecar_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._sidecar_path())

    # ------------------------------------------------------------- scan

    def scan(self) -> List[TrackAsset]:
        side = self._load_sidecar()
        assets: List[TrackAsset] = []
        if not os.path.isdir(self.root):
            return assets
        for fn in sorted(os.listdir(self.root)):
            if not fn.endswith(".sim.json"):
                continue
            path = os.path.join(self.root, fn)
            meta = side.get(fn, {})
            asset = TrackAsset(
                path=path, file_name=fn,
                name=os.path.splitext(os.path.splitext(fn)[0])[0],
                readonly=self.readonly)
            try:
                st = os.stat(path)
                asset.modified, asset.file_size = st.st_mtime, st.st_size
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                road = data.get("road_definition") or {}
                asset.name = str(data.get("name") or asset.name)
                asset.env_version = str(data.get("environment_version", "1.0.0"))
                asset.schema_version = str(data.get("schema_version", "?"))
                asset.point_count = len(road.get("control_points") or [])
                asset.entity_count = len(data.get("entities") or [])
                asset.is_closed = bool(road.get("is_closed", True))
                asset.length_m = _approx_length_m(road)
            except Exception as e:
                asset.broken, asset.error = True, str(e)
            asset.description = str(meta.get("description", ""))
            asset.favorite = bool(meta.get("favorite", False))
            asset.last_opened = float(meta.get("last_opened", 0.0))
            assets.append(asset)
        return assets

    # ------------------------------------------------------------ ops

    def _unique_path(self, name: str) -> str:
        base = _slugify(name)
        path = os.path.join(self.root, f"{base}.sim.json")
        i = 2
        while os.path.exists(path):
            path = os.path.join(self.root, f"{base}_{i}.sim.json")
            i += 1
        return path

    def save_project(self, project: EnvironmentProject,
                     name: Optional[str] = None) -> str:
        """Save a project into the library; returns the file path."""
        if self.readonly:
            raise PermissionError("library is read-only")
        if name:
            project.name = name
        path = getattr(project, "file_path", None)
        if not path or os.path.dirname(os.path.abspath(path)) != self.root:
            path = self._unique_path(project.name)
        project.save(path)
        project.file_path = path
        return path

    def create(self, project: EnvironmentProject,
               description: str = "") -> str:
        if self.readonly:
            raise PermissionError("library is read-only")
        path = self._unique_path(project.name)
        project.save(path)
        project.file_path = path
        if description:
            self.set_description(path, description)
        return path

    def duplicate(self, path: str) -> str:
        proj = EnvironmentProject.load(path)
        proj.name = f"{proj.name} (copy)"
        return self.create(proj)

    def rename(self, path: str, new_name: str) -> None:
        proj = EnvironmentProject.load(path)
        proj.name = new_name
        proj.save(path)
        # keep the file name stable to avoid breaking recents/thumbnails

    def delete(self, path: str) -> None:
        if self.readonly:
            raise PermissionError("library is read-only")
        os.remove(path)
        side = self._load_sidecar()
        side.pop(os.path.basename(path), None)
        self._save_sidecar(side)
        thumb = self.thumb_path_for(path)
        if os.path.exists(thumb):
            try:
                os.remove(thumb)
            except OSError:
                pass

    def set_favorite(self, path: str, fav: bool) -> None:
        self._set_meta(path, "favorite", fav)

    def set_description(self, path: str, desc: str) -> None:
        self._set_meta(path, "description", desc)

    def mark_opened(self, path: str) -> None:
        self._set_meta(path, "last_opened", time.time())

    def _set_meta(self, path: str, key: str, value: Any) -> None:
        if self.readonly:
            return
        side = self._load_sidecar()
        fn = os.path.basename(path)
        side.setdefault(fn, {})[key] = value
        self._save_sidecar(side)

    # -------------------------------------------------------- thumbs dir

    def thumbs_dir(self) -> str:
        d = os.path.join(self.root, ".thumbs")
        if not self.readonly:
            os.makedirs(d, exist_ok=True)
        return d

    def thumb_path_for(self, asset_path: str) -> str:
        base = os.path.splitext(os.path.splitext(
            os.path.basename(asset_path))[0])[0]
        return os.path.join(self.thumbs_dir(), base + ".png")

    # ------------------------------------------------------------ recent

    def recents(self, n: int = 6) -> List[TrackAsset]:
        assets = [a for a in self.scan() if a.last_opened]
        assets.sort(key=lambda a: a.last_opened, reverse=True)
        return assets[:n]
