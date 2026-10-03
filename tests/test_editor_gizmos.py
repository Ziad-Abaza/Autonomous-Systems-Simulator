"""Headless tests for editor manipulation gizmos (PH2-ED-002).

Width / elevation / banking handles on the selected control point and
the rotation handle on the selected entity must hit-test on mouse-down,
mutate the model while dragging, and commit history on mouse-up.
"""
import math
import os
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from sim_ui.editor import VisualTrackEditor
from sim_core.track.road_definition import RoadDefinition
from sim_core.math_utils import Vec3
from sim_core.world.entity import create_entity


@pytest.fixture()
def ed():
    rd = RoadDefinition.create_default_oval()
    ed = VisualTrackEditor(rd, [])
    ed.canvas_rect = pygame.Rect(0, 0, 1280, 800)
    ed.zoom = 4.0
    ed.view_offset_x = 0.0
    ed.view_offset_y = 0.0
    ed.select_control_point(1)
    return ed


def _history_len(ed):
    return len(getattr(ed, "history", []))


def test_width_handle_drag(ed):
    cp = ed.road_def.control_points[1]
    hx, hy = ed._cp_handle_pos("width")
    assert ed.handle_mouse_down((hx, hy), 1)
    assert ed.is_dragging_width
    # +40 px at zoom 4 → +10 m per side → +20 m width
    ed.handle_mouse_move((hx + 40, hy))
    assert cp.width == pytest.approx(32.0, abs=0.6)
    ed.handle_mouse_up()
    assert not ed.is_dragging_width


def test_width_clamped(ed):
    cp = ed.road_def.control_points[1]
    hx, hy = ed._cp_handle_pos("width")
    ed.handle_mouse_down((hx, hy), 1)
    ed.handle_mouse_move((hx + 500, hy))
    assert cp.width == pytest.approx(40.0, abs=0.2)
    ed.handle_mouse_up()


def test_elevation_handle_drag(ed):
    cp = ed.road_def.control_points[1]
    hx, hy = ed._cp_handle_pos("elev")
    assert ed.handle_mouse_down((hx, hy), 1)
    # drag UP 20 px at zoom 4 → +5.0 m
    ed.handle_mouse_move((hx, hy - 20))
    assert cp.z == pytest.approx(5.0, abs=0.05)
    ed.handle_mouse_up()


def test_banking_handle_drag(ed):
    cp = ed.road_def.control_points[1]
    hx, hy = ed._cp_handle_pos("bank")
    assert ed.handle_mouse_down((hx, hy), 1)
    # drag UP 10 px → +5.0 deg, clamp ±30
    ed.handle_mouse_move((hx, hy - 10))
    assert cp.banking == pytest.approx(5.0, abs=0.05)
    ed.handle_mouse_move((hx, hy - 100))
    assert cp.banking == pytest.approx(30.0, abs=0.05)
    ed.handle_mouse_up()


def test_entity_rotate_handle(ed):
    ed.select_entity(None)
    ent = create_entity("cone", pos=Vec3(10, 10, 0), yaw=0.0)
    ed.entities.append(ent)
    ed.select_entity(ent.entity_id)
    rhx, rhy = ed._entity_rotate_handle_pos()
    assert rhx is not None
    assert ed.handle_mouse_down((rhx, rhy), 1)
    esx, esy = ed.world_to_screen(10, 10)
    # point the handle 90 deg up → entity yaw ~ +90
    ed.handle_mouse_move((esx, esy - 20))
    assert math.degrees(ent.yaw) == pytest.approx(90.0, abs=1.0)
    ed.handle_mouse_up()


def test_handles_take_priority_over_cp_drag(ed):
    """Clicking a handle must NOT start a control-point drag."""
    hx, hy = ed._cp_handle_pos("width")
    assert ed.handle_mouse_down((hx, hy), 1)
    assert ed.is_dragging_width and not ed.is_dragging_point
    ed.handle_mouse_up()
