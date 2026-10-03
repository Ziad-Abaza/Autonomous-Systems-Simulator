"""Runtime path resolution — development vs frozen (PyInstaller) builds.

Two roots:
- resource_root(): read-only bundled assets (presets/, packaged docs).
  Under PyInstaller this is sys._MEIPASS; in development the repo root.
- runtime_root(): writable dirs (data/, tracks/, experiments/, recordings/).
  Under PyInstaller this is the directory containing the .exe; in
  development the repo root.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> str:
    return sys._MEIPASS if is_frozen() else _REPO_ROOT


def runtime_root() -> str:
    return os.path.dirname(sys.executable) if is_frozen() else _REPO_ROOT
