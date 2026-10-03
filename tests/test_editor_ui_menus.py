"""Headless regression tests for the EDIT workspace toolbar dropdowns.

The Place/View menus are drawn below the toolbar, over the outline/canvas
region. They must paint AFTER those panels or they are invisible — this is
a draw-order contract, not a hit-test one (the z-ordered hit registry
already wins either way).
"""
import os
import pygame
import pytest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from sim_ui import theme as T
from sim_ui.widgets import UIContext, Fonts
from sim_ui.editor import VisualTrackEditor
from sim_ui.editor_ui import EditorUI
from sim_core.track.road_definition import RoadDefinition


class _FakeInspector:
    props_area = None
    TAB_TOOLTIPS = {}
    TAB_LABELS = {}

    def draw(self, *a, **k):
        return []


@pytest.fixture()
def rig():
    T.set_theme("dark")
    surf = pygame.Surface((1280, 800), pygame.SRCALPHA)
    ctx = UIContext(surf, Fonts())
    f = ctx.fonts.body
    app = SimpleNamespace(
        track_editor=VisualTrackEditor(
            RoadDefinition.create_default_oval()),
        env=SimpleNamespace(track=None, checkpoint_manager=None,
                            entities=[]),
        ui_ctx=ctx,
        ui_renderer=SimpleNamespace(font_title=f, font_bold=f,
                                    font_small=f, font_mono=f),
        inspector=_FakeInspector(),
        dirty=False,
        project=None,
    )
    ui = EditorUI(app)
    body = pygame.Rect(0, 0, 1280, 800)
    return ctx, surf, app, ui, body


def _frame(ctx, ui, body):
    ctx.begin_frame()
    ui.draw(ctx, body)


def _menu_item(ctx, action):
    for r in ctx.regions:
        if r["kind"] == "menu_item" and r["action"] == action:
            return r
    return None


def test_place_menu_paints_over_canvas(rig):
    ctx, surf, app, ui, body = rig
    _frame(ctx, ui, body)
    r = next(r for r in ctx.regions if r["action"] == "ed_place_menu")
    ctx.mouse_down(r["rect"].center, 1)
    ui.on_action("ed_place_menu", None)
    assert ctx.open_menu_id == "ed_place"

    _frame(ctx, ui, body)
    item = _menu_item(ctx, "ed_place")
    assert item is not None, "place menu items not registered"
    px = surf.get_at(item["rect"].center)[:3]
    assert px == T.C.panel_alt[:3] or px == T.C.hover[:3], (
        f"menu item painted as {px} — covered by canvas ({T.C.canvas[:3]})?")


def test_view_menu_paints_over_canvas(rig):
    ctx, surf, app, ui, body = rig
    _frame(ctx, ui, body)
    r = next(r for r in ctx.regions if r["action"] == "ed_view_menu")
    ctx.mouse_down(r["rect"].center, 1)
    ui.on_action("ed_view_menu", None)
    assert ctx.open_menu_id == "ed_view"

    _frame(ctx, ui, body)
    item = _menu_item(ctx, "view_curv")
    assert item is not None, "view menu items not registered"
    px = surf.get_at(item["rect"].center)[:3]
    assert px == T.C.panel_alt[:3] or px == T.C.hover[:3]


def test_place_selection_arms_tool_and_closes_menu(rig):
    ctx, surf, app, ui, body = rig
    ctx.open_menu_id = "ed_place"
    _frame(ctx, ui, body)
    item = _menu_item(ctx, "ed_place")
    assert item["payload"] in ("obstacle", "barrier", "cone",
                               "traffic_sign", "traffic_light", "spawn")
    ui.on_action("ed_place", "cone")
    assert app.track_editor.active_tool == "cone"
    assert ctx.open_menu_id is None


def test_view_toggle_flips_editor_flag(rig):
    ctx, surf, app, ui, body = rig
    ed = app.track_editor
    assert ed.show_curvature is True
    ui.on_action("view_curv", None)
    assert ed.show_curvature is False
    ui.on_action("view_tang", None)
    assert ed.show_tangents is False
    ui.on_action("view_width", None)
    assert ed.show_width_handles is False
    ui.on_action("view_grid", None)
    assert ed.show_grid is False
