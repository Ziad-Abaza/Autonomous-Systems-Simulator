"""
Experiments browser panel — HOME section.

Lists experiments from ExperimentManager with status, environment
context, and run info. Detail pane shows manifest fields + runs.
Actions delegate to the app's training service hooks (_train_action),
which already exist and are service-layer clean.
"""
from __future__ import annotations
import os
import time
from typing import Any, Optional

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, panel, button, list_row, section_label,
    scroll_begin, scroll_end, _draw_text,
)


def draw_experiments_browser(ctx: UIContext, body: pygame.Rect, app) -> None:
    x0, y0 = body.x + 24, body.y + 20
    _draw_text(ctx, ctx.fonts.h1, "Experiments", T.C.text, x0, y0)
    _draw_text(ctx, ctx.fonts.small,
               "Created from tracks — open a track and use its TRAIN "
               "settings to create one.", T.C.text_dim, x0, y0 + 30)

    exps = app.exp_mgr.list_experiments()
    list_w = min(560, body.w - 48)
    detail_x = x0 + list_w + 20
    detail_w = body.right - detail_x - 24

    area = pygame.Rect(x0, y0 + 56, list_w, body.bottom - y0 - 70)
    content_h = len(exps) * 48 + 8
    off = scroll_begin(ctx, "exp_list", area, content_h)
    if not exps:
        _draw_text(ctx, ctx.fonts.body,
                   "No experiments yet. Open a track, then create an "
                   "experiment from the TRAIN section.",
                   T.C.text_faint, area.x + 8, area.y + 16,
                   max_w=area.w - 40)
    for i, e in enumerate(exps):
        r = pygame.Rect(area.x, area.y + i * 48 - off, area.w - 6, 44)
        if r.bottom < area.top or r.top > area.bottom:
            continue
        sel = app.train_state.get("selected_experiment") == e["experiment_id"]
        status = "LAUNCHED" if e.get("launched") else "DRAFT"
        if e.get("archived"):
            status = "ARCHIVED"
        sec = (f"{e.get('algorithm','?').upper()} · {e.get('environment_name','?')} "
               f"· seed {e.get('random_seed','?')}")
        list_row(ctx, r, e.get("name") or e["experiment_id"],
                 "exp_select", e["experiment_id"], secondary=sec,
                 selected=sel, right=status)
    scroll_end(ctx, "exp_list", area)

    sel = app.train_state.get("selected_experiment")
    if sel and detail_w > 160:
        _draw_experiment_detail(ctx, app, sel, detail_x, y0 + 56, detail_w)


def _draw_experiment_detail(ctx: UIContext, app, exp_id: str,
                            x: int, y: int, w: int) -> None:
    h = min(440, ctx.surface.get_height() - y - 20)
    rect = pygame.Rect(x, y, w, h)
    panel(ctx, rect)
    try:
        m = app.exp_mgr.load(exp_id)
    except Exception as e:
        _draw_text(ctx, ctx.fonts.body, f"Cannot load: {e}", T.C.error,
                   rect.x + 12, rect.y + 12, max_w=rect.w - 24)
        return

    _draw_text(ctx, ctx.fonts.bold, m.name or exp_id, T.C.text,
               rect.x + 12, rect.y + 10, max_w=rect.w - 24)
    yy = rect.y + 36
    exp_dir = app.exp_mgr.experiment_dir(exp_id)
    runs = app.run_mgr.list_runs(exp_dir)
    fields = [
        ("Algorithm", m.training.algorithm.upper()),
        ("Timesteps", str(m.training.total_timesteps)),
        ("Environment", m.environment_name or "?"),
        ("Env fingerprint", (m.environment_fingerprint or "")[:12] + "…"),
        ("Scenario", str(m.scenario_id)),
        ("Seed", str(m.random_seed)),
        ("Launched", "yes" if m.launched else "no"),
        ("Runs", str(len(runs))),
    ]
    for k, v in fields:
        _draw_text(ctx, ctx.fonts.small, k, T.C.text_dim, rect.x + 12, yy)
        _draw_text(ctx, ctx.fonts.small, v, T.C.text, rect.x + 140, yy,
                   max_w=rect.w - 150)
        yy += 20
    yy += 6
    for r in runs[-5:]:
        rr = pygame.Rect(rect.x + 12, yy, rect.w - 24, 26)
        last = ""
        try:
            st = app.run_mgr.load_run(exp_dir, r["run_id"])
            last = f" · {st.status.value if hasattr(st.status,'value') else st.status}"
        except Exception:
            pass
        _draw_text(ctx, ctx.fonts.small, r["run_id"] + last, T.C.text_dim,
                   rr.x, rr.y, max_w=rr.w - 90)
        yy += 28

    by = rect.bottom - 40
    button(ctx, pygame.Rect(rect.x + 12, by, 110, 28),
           "Launch", "exp_launch", exp_id, style="primary",
           enabled=bool(exp_id))
    button(ctx, pygame.Rect(rect.x + 130, by, 110, 28),
           "Open Track", "exp_open_env", exp_id)
    button(ctx, pygame.Rect(rect.right - 96, by, 84, 28),
           "Export…", "exp_export", exp_id)
