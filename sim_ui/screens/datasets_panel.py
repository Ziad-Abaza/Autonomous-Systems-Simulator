"""
Dataset browser panel — shared between HOME and the workspace DATA tab.

Discovers transitions_v1 datasets (manifest.json + episodes.jsonl) under:
    <data_root>/datasets/**
    <data_root>/recordings/**
    experiments/**/dataset_export*

Pure scan function is unit-testable; drawing is UIContext widgets.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, panel, button, label, list_row, section_label,
    scroll_begin, scroll_end, _draw_text, menu_draw,
)


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def scan_datasets(roots: List[str]) -> List[Dict[str, Any]]:
    """Find dataset directories under the given roots.

    A dataset dir contains manifest.json (and usually episodes.jsonl).
    Episode-recording dirs (episode.json with frames) are included as
    kind="recording".
    """
    found: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            # skip thumbnails/metadata internals
            if ".thumbs" in dirpath.split(os.sep):
                continue
            manifest = os.path.join(dirpath, "manifest.json")
            ep_json = os.path.join(dirpath, "episode.json")
            ep_gz = os.path.join(dirpath, "episode.json.gz")
            if dirpath in seen:
                continue
            if os.path.exists(manifest):
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        m = json.load(f)
                except (OSError, json.JSONDecodeError):
                    m = {}
                seen.add(dirpath)
                is_recording = m.get("kind") == "episode_recording"
                found.append({
                    "kind": "recording" if is_recording else "dataset",
                    "path": dirpath,
                    "name": m.get("name") or os.path.basename(dirpath),
                    "format": m.get("dataset_format", "transitions_v1"),
                    "episodes": 1 if is_recording else m.get("episodes"),
                    "steps": m.get("steps"),
                    "env_name": (m.get("env") or {}).get("name")
                                or m.get("environment_name")
                                or m.get("track_name", ""),
                    "run_id": m.get("run_id", ""),
                    "experiment_id": m.get("experiment_id", ""),
                    "size": _dir_size(dirpath),
                    "modified": os.path.getmtime(dirpath),
                })
                dirnames[:] = []  # don't descend into datasets
            elif os.path.exists(ep_json) or os.path.exists(ep_gz):
                seen.add(dirpath)
                meta = {}
                try:
                    p = ep_json if os.path.exists(ep_json) else ep_gz
                    with open(p, "r", encoding="utf-8") as f:
                        meta = json.load(f).get("metadata", {})
                except Exception:
                    pass
                found.append({
                    "kind": "recording",
                    "path": dirpath,
                    "name": os.path.basename(dirpath),
                    "format": "episode_recording",
                    "episodes": 1,
                    "steps": meta.get("total_steps"),
                    "env_name": meta.get("track_name", ""),
                    "run_id": "", "experiment_id": "",
                    "size": _dir_size(dirpath),
                    "modified": os.path.getmtime(dirpath),
                })
    found.sort(key=lambda d: d["modified"], reverse=True)
    return found


def draw_datasets_browser(ctx: UIContext, body: pygame.Rect, app) -> None:
    x0, y0 = body.x + 24, body.y + 20
    _draw_text(ctx, ctx.fonts.h1, "Datasets & Recordings", T.C.text, x0, y0)
    _draw_text(ctx, ctx.fonts.small,
               f"Searched in {app.settings.data_root} and experiments/",
               T.C.text_dim, x0, y0 + 30)

    roots = [app.settings.data_root,
             os.path.join(os.path.dirname(app.settings.data_root),
                          "experiments")]
    # also experiments dir under repo (independent of data_root)
    roots.append(os.path.abspath(
        os.path.join(os.path.dirname(app.settings.path), "experiments")))
    datasets = scan_datasets(roots)

    list_w = min(560, body.w - 48)
    detail_x = x0 + list_w + 20
    detail_w = body.right - detail_x - 24

    area = pygame.Rect(x0, y0 + 56, list_w, body.bottom - y0 - 70)
    recs = [d for d in datasets if d["kind"] == "recording"]
    dss = [d for d in datasets if d["kind"] == "dataset"]
    sections = (("EPISODE RECORDINGS", recs), ("TRAINING DATASETS", dss))
    content_h = sum(len(g) * 48 + 28 for _, g in sections) + 8
    off = scroll_begin(ctx, "ds_list", area, content_h)
    if not datasets:
        _draw_text(ctx, ctx.fonts.body,
                   "Nothing here yet — record an episode in SIMULATE or "
                   "export a training run's dataset.",
                   T.C.text_faint, area.x + 8, area.y + 16, max_w=area.w - 40)
    yy = 0
    for title, group in sections:
        if not group:
            continue
        r = pygame.Rect(area.x, area.y + yy - off, area.w - 6, 22)
        if area.top <= r.bottom and r.top <= area.bottom:
            _draw_text(ctx, ctx.fonts.caption, title, T.C.text_faint,
                       r.x + 2, r.y + 4)
        yy += 28
        for d in group:
            r = pygame.Rect(area.x, area.y + yy - off, area.w - 6, 44)
            yy += 48
            if r.bottom < area.top or r.top > area.bottom:
                continue
            sel = getattr(app, "dataset_detail", None) == d["path"]
            kind_lbl = ("Dataset" if d["kind"] == "dataset"
                        else "Recording")
            sec = (f"{kind_lbl} · {d.get('episodes') or '?'} eps · "
                   f"{d.get('steps') or '?'} steps · "
                   f"{format_bytes(d['size'])}")
            list_row(ctx, r, d["name"], "ds_select", d["path"],
                     secondary=sec, selected=sel,
                     right=time.strftime("%m-%d %H:%M",
                                         time.localtime(d["modified"])))
    scroll_end(ctx, "ds_list", area)

    # detail pane
    if getattr(app, "dataset_detail", None) and detail_w > 160:
        _draw_dataset_detail(ctx, app, detail_x, y0 + 56, detail_w)


def _draw_dataset_detail(ctx: UIContext, app, x: int, y: int, w: int) -> None:
    path = app.dataset_detail
    h = min(400, ctx.surface.get_height() - y - 20)
    rect = pygame.Rect(x, y, w, h)
    panel(ctx, rect)

    from sim_experiment.dataset_inspect import inspect_dataset
    try:
        rep = inspect_dataset(path) if os.path.exists(
            os.path.join(path, "episodes.jsonl")) else None
    except Exception:
        rep = None

    _draw_text(ctx, ctx.fonts.bold, os.path.basename(path), T.C.text,
               rect.x + 12, rect.y + 10, max_w=rect.w - 24)
    yy = rect.y + 34
    if rep:
        st = rep.get("statistics", {})
        valid = rep.get("valid", False)
        col = T.C.ok if valid else T.C.error
        _draw_text(ctx, ctx.fonts.small,
                   "VALID" if valid else "INVALID", col, rect.x + 12, yy)
        yy += 20
        for k, v in [
            ("Format", str(rep.get("dataset_format"))),
            ("Episodes", str(st.get("episode_count", "?"))),
            ("Steps", str(st.get("step_count", "?"))),
            ("Return mean", f"{st.get('return', {}).get('mean', 0):.2f}"),
            ("Return min/max", f"{st.get('return', {}).get('min', 0):.1f} / "
                               f"{st.get('return', {}).get('max', 0):.1f}"),
            ("Obs dims", str(st.get("obs_dims", []))),
            ("Action dims", str(st.get("action_dims", []))),
        ]:
            _draw_text(ctx, ctx.fonts.small, k, T.C.text_dim, rect.x + 12, yy)
            _draw_text(ctx, ctx.fonts.small, v, T.C.text,
                       rect.x + 130, yy, max_w=rect.w - 140)
            yy += 20
        errs = rep.get("errors") or []
        if errs:
            yy += 6
            _draw_text(ctx, ctx.fonts.small, "Problems:", T.C.error,
                       rect.x + 12, yy)
            yy += 18
            for e in errs[:4]:
                _draw_text(ctx, ctx.fonts.caption, "• " + e, T.C.error,
                           rect.x + 16, yy, max_w=rect.w - 28)
                yy += 15
    else:
        _draw_text(ctx, ctx.fonts.small,
                   "Episode recording (not a transitions_v1 dataset).",
                   T.C.text_dim, rect.x + 12, yy, max_w=rect.w - 24)
        yy += 24

    by = rect.bottom - 40
    # recordings can be opened directly in the replay player
    ep_file = os.path.join(path, "episode.json")
    if os.path.exists(ep_file):
        button(ctx, pygame.Rect(rect.x + 12, by, 110, 28),
               "View Replay", "ds_view_replay", ep_file,
               style="primary")
        x_btn = rect.x + 130
    else:
        x_btn = rect.x + 12
    button(ctx, pygame.Rect(x_btn, by, 120, 28),
           "Open Folder", "ds_open_folder", path)
    if rep:
        button(ctx, pygame.Rect(x_btn + 128, by, 110, 28),
               "Open Source", "ds_open_source", path)
    button(ctx, pygame.Rect(rect.right - 92, by, 80, 28),
           "Delete…", "ds_delete", path, style="danger")
