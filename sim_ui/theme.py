"""
Design tokens for the Simulation Studio UI.

Single source of truth for color, spacing, and typography scale.
Semantic tokens only — components never hardcode RGB literals.

Two intentional themes: "dark" (default, tuned for long sessions and
viewport legibility) and "light". Both are designed, not inverted:
selection, warning, error and viewport-adjacent surfaces are chosen
per theme for contrast.
"""
from __future__ import annotations
from typing import Dict, Tuple

Color = Tuple[int, int, int]


_DARK: Dict[str, Color] = {
    # Surfaces — deep to raised
    "bg_deep":        (13, 16, 21),     # window chrome / outermost
    "bg":             (18, 22, 28),     # app background
    "canvas":         (16, 20, 25),     # editor canvas backdrop
    "panel":          (24, 29, 37),     # cards, panels
    "panel_alt":      (30, 36, 46),     # raised sections in panels
    "hover":          (38, 46, 58),     # hover fill
    "active":         (46, 56, 70),     # pressed / active fill
    "scrim":          (8, 10, 13),      # modal backdrop (alpha applied)

    # Lines
    "border":         (44, 52, 65),
    "border_soft":    (34, 40, 50),
    "separator":      (38, 44, 55),

    # Text
    "text":           (228, 233, 240),
    "text_dim":       (158, 168, 181),
    "text_faint":     (110, 120, 134),
    "text_on_accent": (255, 255, 255),

    # Accents & semantics
    "accent":         (58, 145, 220),   # primary actions / active nav
    "accent_soft":    (36, 84, 130),    # selected rows, subtle accents
    "accent_line":    (66, 160, 235),   # accent borders/focus ring
    "ok":             (70, 180, 105),
    "warn":           (214, 162, 60),
    "error":          (224, 88, 76),
    "info":           (96, 160, 220),
    "rec":            (216, 66, 58),
    "selection":      (255, 214, 84),   # object selection ring (viewport)

    # Editor-viewport visualization tokens (view-layer only, never
    # part of agent observations — see OBSERVATION_SECURITY)
    "vx_centerline":  (0, 214, 255),
    "vx_boundary":    (150, 160, 175),
    "vx_spawn":       (0, 230, 130),
    "vx_checkpoint":  (70, 190, 130),
    "vx_curv_mid":    (255, 200, 60),
    "vx_curv_high":   (255, 80, 80),
    "vx_grid":        (32, 38, 48),
    "vx_refline":     (70, 110, 170),
}

_LIGHT: Dict[str, Color] = {
    "bg_deep":        (196, 202, 211),
    "bg":             (226, 230, 236),
    "canvas":         (240, 243, 247),
    "panel":          (246, 248, 251),
    "panel_alt":      (236, 240, 245),
    "hover":          (222, 228, 236),
    "active":         (208, 216, 228),
    "scrim":          (60, 68, 80),

    "border":         (176, 184, 196),
    "border_soft":    (200, 206, 216),
    "separator":      (196, 202, 212),

    "text":           (28, 34, 44),
    "text_dim":       (78, 88, 102),
    "text_faint":     (120, 128, 140),
    "text_on_accent": (255, 255, 255),

    "accent":         (24, 118, 200),
    "accent_soft":    (190, 216, 244),
    "accent_line":    (24, 118, 200),
    "ok":             (30, 140, 70),
    "warn":           (170, 120, 22),
    "error":          (190, 56, 46),
    "info":           (36, 110, 180),
    "rec":            (196, 48, 40),
    "selection":      (190, 140, 20),

    "vx_centerline":  (0, 130, 190),
    "vx_boundary":    (96, 106, 120),
    "vx_spawn":       (0, 140, 80),
    "vx_checkpoint":  (30, 130, 85),
    "vx_curv_mid":    (190, 140, 20),
    "vx_curv_high":   (200, 50, 40),
    "vx_grid":        (216, 222, 230),
    "vx_refline":     (110, 140, 185),
}


class _Tokens:
    """Attribute access to the active theme's colors."""
    def __init__(self, palette: Dict[str, Color]):
        self._p = dict(palette)

    def __getattr__(self, name: str) -> Color:
        try:
            return self._p[name]
        except KeyError:
            raise AttributeError(f"unknown theme token: {name}")

    def get(self, name: str, default: Color = (255, 0, 255)) -> Color:
        return self._p.get(name, default)


THEME_NAME = "dark"
C = _Tokens(_DARK)


def set_theme(name: str) -> None:
    global THEME_NAME
    THEME_NAME = name
    C._p = dict(_LIGHT if name == "light" else _DARK)


# ---- Typography scale (px; logical sizes — see ui_scale) ----
FONT_CAPTION = 12
FONT_SMALL = 13
FONT_BODY = 14
FONT_TITLE = 16
FONT_H1 = 20
FONT_MONO = 13

# ---- Spacing / metrics ----
PAD = 12
PAD_SM = 8
GAP = 8
RADIUS = 6
RADIUS_SM = 4
BORDER_W = 1
TOP_BAR_H = 40
STATUS_BAR_H = 28
TOOLBAR_H = 36
RAIL_LEFT_W = 240
RAIL_RIGHT_W = 340
SCROLL_W = 10
MIN_TOUCH = 24  # minimum interactive target (px)
