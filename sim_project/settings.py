"""
Studio application settings — user-facing preferences persisted to a
small JSON file inside the studio data directory
(`<data_root>/studio_settings.json`, gitignored).

Separates *application* preferences (data root, theme, ui scale) from
*environment* documents (*.sim.json) — different lifetimes, different
files.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_SETTINGS: Dict[str, Any] = {
    "data_root": "",            # user-chosen dataset/recording root; "" = <repo>/data
    "theme": "dark",
    "ui_scale": 1.0,
    "recent_files": [],         # absolute paths, most-recent first (max 10)
}


class StudioSettings:
    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for k in DEFAULT_SETTINGS:
                if k in d:
                    self._data[k] = d[k]
        except (OSError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ---- typed accessors ----

    @property
    def data_root(self) -> str:
        v = self._data.get("data_root") or ""
        if v:
            return os.path.abspath(v)
        # Default: <repo>/data — independent of where the settings file
        # itself lives (the file is stored inside this directory).
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(repo, "data")

    @data_root.setter
    def data_root(self, v: str) -> None:
        self._data["data_root"] = v

    @property
    def theme(self) -> str:
        return self._data.get("theme", "dark")

    @theme.setter
    def theme(self, v: str) -> None:
        self._data["theme"] = "light" if v == "light" else "dark"

    @property
    def ui_scale(self) -> float:
        try:
            return max(0.75, min(2.0, float(self._data.get("ui_scale", 1.0))))
        except (TypeError, ValueError):
            return 1.0

    @ui_scale.setter
    def ui_scale(self, v: float) -> None:
        self._data["ui_scale"] = max(0.75, min(2.0, float(v)))

    @property
    def recent_files(self) -> List[str]:
        return [p for p in self._data.get("recent_files", []) if os.path.exists(p)]

    def push_recent(self, path: str, keep: int = 10) -> None:
        path = os.path.abspath(path)
        rec = [p for p in self._data.get("recent_files", []) if p != path]
        self._data["recent_files"] = [path] + rec[: keep - 1]

    # ---- recording/data dirs ----

    def recordings_dir(self) -> str:
        d = os.path.join(self.data_root, "recordings")
        os.makedirs(d, exist_ok=True)
        return d

    def datasets_dir(self) -> str:
        d = os.path.join(self.data_root, "datasets")
        os.makedirs(d, exist_ok=True)
        return d

    def validate_data_root(self, path: str) -> Optional[str]:
        """Returns an error string, or None if usable."""
        if not path or not path.strip():
            return "empty path"
        p = os.path.abspath(path.strip())
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            return f"cannot create directory: {e}"
        probe = os.path.join(p, ".write_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
        except OSError as e:
            return f"directory is not writable: {e}"
        return None
