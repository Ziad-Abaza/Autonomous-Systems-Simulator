"""
Track thumbnail renderer for the library.

Draws a road_definition dict onto a small pygame.Surface — no GL, no
display required, deterministic for a given input. Used for library
cards and the recents strip. Geometry is the control-point polyline
(same source of truth the editor edits), drawn as filled road width
with a centerline, spawn marker and direction chevron.
"""
from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pygame


def _bounds(road: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cps = road.get("control_points") or []
    if not cps:
        return (-10, -10, 10, 10)
    xs = [c.get("x", 0.0) for c in cps]
    ys = [c.get("y", 0.0) for c in cps]
    pad = max(c.get("width", 12.0) for c in cps) * 0.5 + 4.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _left_right_edges(road: Dict[str, Any]) -> Tuple[List, List]:
    """Approximate road edges from control-point polyline."""
    cps = road.get("control_points") or []
    n = len(cps)
    left, right = [], []
    for i, c in enumerate(cps):
        a = cps[i - 1] if i > 0 else cps[(n - 1) if road.get("is_closed") else 0]
        b = cps[(i + 1) % n] if (i < n - 1 or road.get("is_closed")) else c
        dx, dy = b.get("x", 0) - a.get("x", 0), b.get("y", 0) - a.get("y", 0)
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        hw = c.get("width", 12.0) * 0.5
        left.append((c.get("x", 0) + nx * hw, c.get("y", 0) + ny * hw))
        right.append((c.get("x", 0) - nx * hw, c.get("y", 0) - ny * hw))
    return left, right


def render_track_thumbnail(
    road: Dict[str, Any],
    size: Tuple[int, int] = (192, 112),
    *,
    bg=(24, 29, 37),
    road_col=(52, 58, 68),
    edge_col=(110, 120, 132),
    center_col=(0, 190, 225),
    spawn_col=(40, 210, 120),
    pad_frac: float = 0.08,
) -> pygame.Surface:
    """Render a thumbnail Surface for a road_definition dict."""
    w, h = size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(bg)

    cps = road.get("control_points") or []
    if len(cps) < 2:
        return surf

    minx, miny, maxx, maxy = _bounds(road)
    span_x, span_y = max(maxx - minx, 1e-3), max(maxy - miny, 1e-3)
    pad = pad_frac
    scale = min(w * (1 - 2 * pad) / span_x, h * (1 - 2 * pad) / span_y)

    def to_px(x, y):
        px = w * pad + (x - minx) * scale + (w * (1 - 2 * pad) - span_x * scale) / 2
        py = h * pad + (maxy - y) * scale + (h * (1 - 2 * pad) - span_y * scale) / 2
        return (int(px), int(py))

    left, right = _left_right_edges(road)
    poly = [to_px(*p) for p in left] + [to_px(*p) for p in reversed(right)]
    if len(poly) >= 3:
        pygame.draw.polygon(surf, road_col, poly)
        pygame.draw.polygon(surf, edge_col, poly, 1)

    center = [to_px(c.get("x", 0), c.get("y", 0)) for c in cps]
    if len(center) > 1:
        pygame.draw.lines(surf, center_col, bool(road.get("is_closed")), center,
                          max(1, int(2 * scale / 4)) if scale > 2 else 1)
        # direction chevron at ~40% of the polyline
        i = max(1, int(len(center) * 0.4))
        x0, y0 = center[i - 1]
        x1, y1 = center[i]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        tip = (int(x0 + ux * 7), int(y0 + uy * 7))
        l1 = (int(x0 - uy * 4), int(y0 + ux * 4))
        l2 = (int(x0 + uy * 4), int(y0 - ux * 4))
        pygame.draw.polygon(surf, center_col, [tip, l1, l2])

    sp = road.get("spawn_point") or {}
    sx, sy = to_px(sp.get("x", 0), sp.get("y", 0))
    pygame.draw.circle(surf, spawn_col, (sx, sy), 4)
    yaw = float(sp.get("yaw", 0.0))
    pygame.draw.line(surf, spawn_col, (sx, sy),
                     (int(sx + math.cos(yaw) * 10),
                      int(sy - math.sin(yaw) * 10)), 2)
    return surf


def thumbnail_for_file(path: str, size: Tuple[int, int] = (192, 112),
                       cache_path: Optional[str] = None) -> Optional[pygame.Surface]:
    """Load (or render+cache) a thumbnail for a *.sim.json file."""
    import json
    try:
        if cache_path and os.path.exists(cache_path):
            if os.path.getmtime(cache_path) >= os.path.getmtime(path):
                return pygame.image.load(cache_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        surf = render_track_thumbnail(data.get("road_definition") or {}, size)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pygame.image.save(surf, cache_path)
        return surf
    except Exception:
        return None
