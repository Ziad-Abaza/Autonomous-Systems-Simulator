"""
Design tokens for the Simulation Studio UI.

Single source of truth for color, spacing, and typography scale.
Semantic tokens only — components never hardcode RGB literals.

Named palettes in two modes — dark (default, tuned for long sessions
and viewport legibility) and light. All palettes are designed, not
inverted: selection, warning, error and viewport-adjacent surfaces are
chosen per palette for contrast.

Palettes are registered in PALETTES and selected by id (e.g. "dark",
"dark_ocean", "light_solar"); the active id persists as the single
"theme" value in studio_settings.json.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

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


def _variant(base: Dict[str, Color],
             **overrides: Color) -> Dict[str, Color]:
    """A complete palette derived from a base — tokens not overridden are
    inherited verbatim, so every variant stays a full token set."""
    p = dict(base)
    p.update(overrides)
    return p


# Ocean — teal accent over cooler, blue-shifted surfaces
_DARK_OCEAN = _variant(
    _DARK,
    bg_deep=(9, 16, 21), bg=(13, 20, 26), canvas=(11, 18, 23),
    panel=(18, 27, 34), panel_alt=(24, 34, 42),
    hover=(31, 43, 52), active=(39, 53, 64),
    accent=(32, 178, 190), accent_soft=(23, 95, 108),
    accent_line=(46, 200, 212), info=(58, 168, 200),
    selection=(110, 230, 210),
    vx_centerline=(0, 225, 225), vx_spawn=(40, 220, 170),
    vx_refline=(60, 130, 180), vx_grid=(26, 36, 46),
)

# Ember — amber accent over warm, brown-shifted surfaces
_DARK_EMBER = _variant(
    _DARK,
    bg_deep=(22, 17, 13), bg=(28, 22, 17), canvas=(25, 20, 16),
    panel=(35, 28, 22), panel_alt=(43, 34, 27),
    hover=(54, 43, 33), active=(64, 51, 40),
    border=(62, 50, 39), border_soft=(50, 41, 33),
    separator=(56, 45, 36),
    text=(240, 232, 222), text_dim=(182, 168, 152), text_faint=(132, 120, 106),
    accent=(224, 138, 56), accent_soft=(140, 86, 36),
    accent_line=(238, 160, 74), info=(226, 152, 82),
    ok=(88, 180, 96), warn=(216, 168, 58), error=(228, 92, 72),
    vx_centerline=(255, 192, 64), vx_spawn=(96, 224, 128),
    vx_checkpoint=(120, 196, 96), vx_refline=(160, 108, 60),
    vx_grid=(46, 38, 31),
)

# Solar — warm paper / solarized-inspired light surfaces
_LIGHT_SOLAR = _variant(
    _LIGHT,
    bg_deep=(199, 191, 175), bg=(231, 224, 208), canvas=(243, 237, 224),
    panel=(250, 246, 235), panel_alt=(238, 231, 215),
    hover=(227, 218, 199), active=(213, 202, 181),
    scrim=(72, 62, 48),
    border=(187, 176, 156), border_soft=(207, 197, 177),
    separator=(201, 191, 171),
    text=(54, 45, 36), text_dim=(97, 85, 68), text_faint=(131, 119, 100),
    accent=(203, 108, 32), accent_soft=(242, 209, 166),
    accent_line=(203, 108, 32), info=(38, 120, 178),
    ok=(62, 140, 62), warn=(178, 128, 22), error=(198, 60, 40),
    rec=(188, 50, 42), selection=(184, 124, 16),
    vx_centerline=(0, 138, 168), vx_spawn=(20, 138, 70),
    vx_checkpoint=(52, 126, 72), vx_curv_mid=(186, 134, 16),
    vx_curv_high=(196, 52, 38), vx_grid=(222, 214, 198),
    vx_refline=(150, 122, 70),
)


PALETTES: Dict[str, Dict[str, Any]] = {
    "dark":        {"name": "Midnight", "mode": "dark",  "colors": _DARK},
    "dark_ocean":  {"name": "Ocean",    "mode": "dark",  "colors": _DARK_OCEAN},
    "dark_ember":  {"name": "Ember",    "mode": "dark",  "colors": _DARK_EMBER},
    "light":       {"name": "Paper",    "mode": "light", "colors": _LIGHT},
    "light_solar": {"name": "Solar",    "mode": "light", "colors": _LIGHT_SOLAR},
}

DEFAULT_PALETTE_ID = "dark"


def list_palettes(mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Registered palettes in display order; each entry is
    {"id", "name", "mode", "colors"}."""
    return [
        {"id": pid, "name": m["name"], "mode": m["mode"], "colors": m["colors"]}
        for pid, m in PALETTES.items()
        if mode is None or m["mode"] == mode
    ]


def palette_mode(theme_id: str) -> str:
    """'dark' or 'light' for a palette id (unknown ids → 'dark')."""
    meta = PALETTES.get(theme_id)
    return meta["mode"] if meta else "dark"


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

# Fixed dark-palette tokens — for surfaces that stay dark regardless of
# the active theme (viewport HUD overlay cards). Using theme tokens on
# those panels produces dark-on-dark invisible text in light themes.
DARK_C = _Tokens(_DARK)


def set_theme(name: str) -> None:
    """Activates a registered palette id; unknown ids fall back to dark."""
    global THEME_NAME
    meta = PALETTES.get(name)
    THEME_NAME = name if meta else DEFAULT_PALETTE_ID
    C._p = dict((meta or PALETTES[DEFAULT_PALETTE_ID])["colors"])


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
