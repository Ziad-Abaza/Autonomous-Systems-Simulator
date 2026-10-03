"""
Vehicle Dynamics Validation panel — workspace DYNAMICS tab.

Runs the deterministic maneuver suite directly against VehicleModel
(physics only, no track/rendering) and presents the measured metrics,
trajectory and time series inside the studio. Every number shown comes
from an actual simulation run via tools.vehicle_dynamics.harness.
"""
from __future__ import annotations
import math
import os
import time
from typing import Any, Dict, Optional

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, button, list_row, scroll_begin, scroll_end, _draw_text,
)
from sim_core.vehicle.vehicle_config import VehicleConfig
from tools.vehicle_dynamics.harness import run_maneuver, RunResult
from tools.vehicle_dynamics.maneuvers import standard_suite

_METRIC_ROWS = [
    ("final_speed_ms", "final speed", "m/s"),
    ("peak_vy_ms", "peak |vy|", "m/s"),
    ("steady_sideslip_deg", "steady sideslip", "deg"),
    ("peak_sideslip_deg", "peak sideslip", "deg"),
    ("peak_yaw_rate_dps", "peak yaw rate", "deg/s"),
    ("steady_yaw_rate_dps", "steady yaw rate", "deg/s"),
    ("peak_ay_ms2", "peak body ay", "m/s^2"),
    ("peak_ax_ms2", "peak body ax", "m/s^2"),
    ("peak_alpha_f_deg", "peak slip front", "deg"),
    ("peak_alpha_r_deg", "peak slip rear", "deg"),
    ("peak_fy_f_n", "peak Fy front", "N"),
    ("peak_fy_r_n", "peak Fy rear", "N"),
    ("mean_curvature_1pm", "mean curvature", "1/m"),
    ("lateral_drift_m", "lateral drift", "m"),
]

_SPEEDS = (10.0, 20.0, 30.0)


def _state(app) -> Dict[str, Any]:
    if not hasattr(app, "dyn_state"):
        app.dyn_state = {
            "runs": {},            # name -> RunResult
            "order": [],           # run order
            "selected": None,      # selected run name
            "speed": 20.0,
            "surface": 1.0,
            "compare_a": None,
            "compare_b": None,
            "running": False,
        }
    return app.dyn_state


def run_selected(app, maneuver_id: Optional[str] = None) -> None:
    ds = _state(app)
    suite = standard_suite(speed_ms=ds["speed"])
    items = suite.values() if maneuver_id is None else [suite[maneuver_id]]
    for m in items:
        r = run_maneuver(m.id, m.script, m.duration_s,
                         config=VehicleConfig(), dt=1.0 / 60.0,
                         initial_speed=m.initial_speed,
                         surface_friction=ds["surface"],
                         pre_roll_s=m.pre_roll_s)
        tag = f"{m.id} @ {ds['speed']:.0f}m/s"
        ds["runs"][tag] = r
        if tag not in ds["order"]:
            ds["order"].append(tag)
        ds["selected"] = tag


def export_runs(app) -> str:
    """Persist all recorded runs to benchmarks/vehicle_dynamics/ui_<ts>/."""
    from tools.vehicle_dynamics.run_validation import (
        build_manifest, write_csv, plot_result)
    ds = _state(app)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    out_dir = os.path.join(repo, "benchmarks", "vehicle_dynamics",
                           f"ui_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    results = {}
    for name, r in ds["runs"].items():
        safe = name.replace(" ", "_").replace("/", "_")
        write_csv(os.path.join(out_dir, f"{safe}.csv"), r)
        plot_result(os.path.join(out_dir, "plots", f"{safe}.png"), r)
        results[name] = r.metrics
    import json as _json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        _json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        _json.dump(build_manifest(VehicleConfig(), 1 / 60, ds["speed"]),
                   f, indent=2)
    return out_dir


# ------------------------------------------------------------------ draw

def draw_dynamics_panel(ctx: UIContext, body: pygame.Rect, app) -> None:
    ds = _state(app)
    rail_w = 300
    rail = pygame.Rect(body.x, body.y, rail_w, body.h)
    main = pygame.Rect(body.x + rail_w + 1, body.y,
                       body.w - rail_w - 1, body.h)

    pygame.draw.rect(ctx.surface, T.C.panel, rail)
    pygame.draw.line(ctx.surface, T.C.border, (rail.right, body.y),
                     (rail.right, body.bottom), 1)
    ctx.block(rail)

    _draw_text(ctx, ctx.fonts.caption, "MANEUVERS", T.C.text_faint,
               rail.x + 12, body.y + 10)

    # speed selector
    y = body.y + 30
    _draw_text(ctx, ctx.fonts.small, "speed", T.C.text_dim,
               rail.x + 12, y + 7)
    x = rail.x + 62
    for v in _SPEEDS:
        b = pygame.Rect(x, y, 56, 26)
        button(ctx, b, f"{v:.0f} m/s", "dyn_speed", v,
               style="primary" if ds["speed"] == v else "default")
        x += 60
    y += 36

    b = pygame.Rect(rail.x + 12, y, rail.w - 24, 28)
    button(ctx, b, "Run full suite", "dyn_run_all", style="primary")
    y += 36

    area = pygame.Rect(rail.x + 8, y, rail.w - 16, body.bottom - y - 8)
    suite = standard_suite(speed_ms=ds["speed"])
    names = [m.id for m in suite.values()]
    content_h = len(names) * 34
    off = scroll_begin(ctx, "dyn_maneuvers", area, content_h)
    for i, mid in enumerate(names):
        r = pygame.Rect(area.x, area.y + i * 34 - off, area.w - 4, 30)
        if r.bottom < area.top or r.top > area.bottom:
            continue
        run_tag = f"{mid} @ {ds['speed']:.0f}m/s"
        sec = "done" if run_tag in ds["runs"] else ""
        list_row(ctx, r, mid, "dyn_run", mid, secondary=sec)
    scroll_end(ctx, "dyn_maneuvers", area)

    # ------------------------------------------------------------- main
    sel = ds["selected"]
    if sel and sel in ds["runs"]:
        _draw_result(ctx, main, ds["runs"][sel], ds, app)
    elif ds["runs"]:
        _draw_compare_or_hint(ctx, main, ds)
    else:
        _draw_text(ctx, ctx.fonts.h1, "Vehicle Dynamics Validation",
                   T.C.text, main.x + 24, main.y + 24)
        _draw_text(ctx, ctx.fonts.small,
                   "Pick a maneuver on the left or run the full suite. "
                   "Physics-only deterministic runs, dt = 1/60 s.",
                   T.C.text_dim, main.x + 24, main.y + 56)


def _draw_compare_or_hint(ctx, main, ds):
    y = main.y + 24
    _draw_text(ctx, ctx.fonts.h1, "Runs", T.C.text, main.x + 24, y)
    y += 36
    for name in ds["order"]:
        r = pygame.Rect(main.x + 24, y, min(460, main.w - 60), 30)
        list_row(ctx, r, name, "dyn_select", name)
        y += 34
    if ds["compare_a"] and ds["compare_b"]:
        _draw_compare_table(ctx, main, ds, y)


def _draw_result(ctx, main, r: RunResult, ds, app) -> None:
    x0 = main.x + 20
    _draw_text(ctx, ctx.fonts.bold, r.name, T.C.text, x0, main.y + 12)

    # compare selectors + export
    bx = main.right - 316
    b = pygame.Rect(bx, main.y + 8, 60, 24)
    button(ctx, b, "Set A", "dyn_cmp_a", r.name,
           style="primary" if ds["compare_a"] == r.name else "default")
    b = pygame.Rect(bx + 68, main.y + 8, 60, 24)
    button(ctx, b, "Set B", "dyn_cmp_b", r.name,
           style="primary" if ds["compare_b"] == r.name else "default")
    b = pygame.Rect(bx + 136, main.y + 8, 90, 24)
    button(ctx, b, "Compare", "dyn_compare")
    b = pygame.Rect(bx + 234, main.y + 8, 70, 24)
    button(ctx, b, "Export", "dyn_export")

    # metrics table (left half)
    y = main.y + 44
    _draw_text(ctx, ctx.fonts.caption, "METRICS", T.C.text_faint, x0, y)
    y += 18
    col_w = min(320, (main.w - 60) // 2)
    for k, lbl, unit in _METRIC_ROWS:
        v = r.metrics.get(k)
        if v is None:
            continue
        _draw_text(ctx, ctx.fonts.small, lbl, T.C.text_dim, x0, y)
        _draw_text(ctx, ctx.fonts.mono, f"{v:10.3f} {unit}",
                   T.C.text, x0 + 170, y)
        y += 20

    # trajectory plot (top right)
    px = x0 + col_w + 30
    pw = main.right - px - 16
    ph = min(240, (main.h - 80) // 2)
    _plot_xy(ctx, pygame.Rect(px, main.y + 44, pw, ph),
             r.series["pos_x"], r.series["pos_y"], "trajectory (m)")

    # time series (bottom right)
    r2 = pygame.Rect(px, main.y + 44 + ph + 26, pw, ph)
    series = [
        ("beta deg", [math.degrees(v) for v in r.series["sideslip"]],
         T.C.accent),
        ("wz deg/s", [math.degrees(v) for v in r.series["yaw_rate"]],
         T.C.ok),
        ("ay m/s2", r.series["ay"], T.C.warn),
    ]
    _plot_series(ctx, r2, series, "sideslip / yaw rate / lat accel vs t")

    if ds["compare_a"] and ds["compare_b"]:
        _draw_compare_table(ctx, main, ds,
                            main.y + 44 + max(ph * 2 + 26, 20 * 15))


def _draw_compare_table(ctx, main, ds, y) -> None:
    a = ds["runs"].get(ds["compare_a"])
    b = ds["runs"].get(ds["compare_b"])
    if not a or not b or y > main.bottom - 40:
        return
    x0 = main.x + 20
    _draw_text(ctx, ctx.fonts.caption,
               f"COMPARE  A={ds['compare_a']}  vs  B={ds['compare_b']}",
               T.C.text_faint, x0, y)
    y += 18
    keys = ["peak_sideslip_deg", "steady_sideslip_deg", "peak_vy_ms",
            "peak_yaw_rate_dps", "peak_ay_ms2", "lateral_drift_m"]
    for k in keys:
        va, vb = a.metrics.get(k), b.metrics.get(k)
        if va is None or vb is None:
            continue
        _draw_text(ctx, ctx.fonts.small, k, T.C.text_dim, x0, y)
        _draw_text(ctx, ctx.fonts.mono, f"{va:9.3f}", T.C.text, x0 + 200, y)
        _draw_text(ctx, ctx.fonts.mono, f"{vb:9.3f}", T.C.text, x0 + 310, y)
        d = vb - va
        col = T.C.ok if abs(d) < 1e-9 else T.C.warn
        _draw_text(ctx, ctx.fonts.mono, f"{d:+9.3f}", col, x0 + 420, y)
        y += 18


# ------------------------------------------------------------------ plots

def _plot_xy(ctx, rect, xs, ys, title):
    pygame.draw.rect(ctx.surface, T.C.bg_deep, rect, border_radius=6)
    pygame.draw.rect(ctx.surface, T.C.border, rect, 1, border_radius=6)
    ctx.block(rect)
    _draw_text(ctx, ctx.fonts.caption, title, T.C.text_faint,
               rect.x + 8, rect.y + 6)
    if len(xs) < 2:
        return
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-6)
    pad = 24
    sx = (rect.w - 2 * pad) / span
    sy = (rect.h - 2 * pad) / span
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    pts = []
    for x, y in zip(xs, ys):
        px = rect.centerx + (x - cx) * sx
        py = rect.centery - (y - cy) * sy
        pts.append((px, py))
    if len(pts) >= 2:
        pygame.draw.lines(ctx.surface, T.C.accent, False, pts, 2)
    pygame.draw.circle(ctx.surface, T.C.ok,
                       (int(pts[0][0]), int(pts[0][1])), 3)
    pygame.draw.circle(ctx.surface, T.C.warn,
                       (int(pts[-1][0]), int(pts[-1][1])), 3)


def _plot_series(ctx, rect, series, title):
    pygame.draw.rect(ctx.surface, T.C.bg_deep, rect, border_radius=6)
    pygame.draw.rect(ctx.surface, T.C.border, rect, 1, border_radius=6)
    ctx.block(rect)
    _draw_text(ctx, ctx.fonts.caption, title, T.C.text_faint,
               rect.x + 8, rect.y + 6)
    n = max((len(s[1]) for s in series), default=0)
    if n < 2:
        return
    lo = min(min(s[1]) for s in series if s[1])
    hi = max(max(s[1]) for s in series if s[1])
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad_l, pad_r, pad_t, pad_b = 34, 10, 22, 10
    x0, x1 = rect.x + pad_l, rect.right - pad_r
    y0, y1 = rect.y + pad_t, rect.bottom - pad_b
    y_zero = y1 - (0.0 - lo) / (hi - lo) * (y1 - y0)
    if y0 <= y_zero <= y1:
        pygame.draw.line(ctx.surface, T.C.border, (x0, y_zero),
                         (x1, y_zero), 1)
    _draw_text(ctx, ctx.fonts.caption, f"{hi:.0f}", T.C.text_faint,
               rect.x + 4, y0 - 2)
    _draw_text(ctx, ctx.fonts.caption, f"{lo:.0f}", T.C.text_faint,
               rect.x + 4, y1 - 8)
    for lbl, vals, col in series:
        pts = []
        for i, v in enumerate(vals):
            px = x0 + i / (n - 1) * (x1 - x0)
            py = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(ctx.surface, col, False, pts, 1)


# ------------------------------------------------------------------ actions

def handle_dynamics_action(app, action: str, payload: Any) -> bool:
    ds = _state(app)
    if action == "dyn_run":
        run_selected(app, payload)
        app._status(f"Ran {payload}", "ok")
    elif action == "dyn_run_all":
        run_selected(app, None)
        app._status("Suite complete", "ok")
    elif action == "dyn_speed":
        ds["speed"] = float(payload)
    elif action == "dyn_select":
        ds["selected"] = payload
    elif action == "dyn_cmp_a":
        ds["compare_a"] = payload
    elif action == "dyn_cmp_b":
        ds["compare_b"] = payload
    elif action == "dyn_compare":
        ds["selected"] = None
    elif action == "dyn_export":
        if not ds["runs"]:
            app._status("Nothing to export", "warn")
        else:
            out = export_runs(app)
            app._status(f"Exported -> {out}", "ok")
            app.reveal_in_folder(out)
    else:
        return False
    return True
