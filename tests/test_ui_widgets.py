"""Headless tests for the Phase-7 UI widget layer.

All tests run without a GL context or display — pygame surfaces and
fonts work headless.
"""
import os
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()

from sim_ui import theme as T
from sim_ui.widgets import UIContext, Fonts, scroll_begin, scroll_end


@pytest.fixture()
def ctx():
    T.set_theme("dark")
    surf = pygame.Surface((800, 600), pygame.SRCALPHA)
    return UIContext(surf, Fonts())


def _frame(ctx, draw_fn):
    ctx.begin_frame()
    draw_fn()
    ctx.begin_frame()  # promote to dispatch registry


class TestDispatch:
    def test_topmost_region_wins(self, ctx):
        def draw():
            r1 = pygame.Rect(0, 0, 100, 100)
            r2 = pygame.Rect(50, 50, 100, 100)
            ctx.hit(r1, "under")
            ctx.hit(r2, "over")
        _frame(ctx, draw)
        r = ctx.dispatch((75, 75))
        assert r and r["action"] == "over"
        r = ctx.dispatch((25, 25))
        assert r and r["action"] == "under"

    def test_z_order_beats_recency(self, ctx):
        def draw():
            ctx.hit(pygame.Rect(0, 0, 100, 100), "base")
            ctx.z = UIContext.Z_MODAL
            ctx.hit(pygame.Rect(10, 10, 50, 50), "modal")
        _frame(ctx, draw)
        assert ctx.dispatch((20, 20))["action"] == "modal"

    def test_block_region_consumes(self, ctx):
        def draw():
            ctx.hit(pygame.Rect(0, 0, 50, 50), "below")
            ctx.z = UIContext.Z_MODAL
            ctx.block(pygame.Rect(0, 0, 200, 200))
        _frame(ctx, draw)
        r = ctx.mouse_down((25, 25), 1)
        assert r is not None and r["action"] is None  # swallowed

    def test_disabled_button_not_hittable(self, ctx):
        def draw():
            r = pygame.Rect(0, 0, 50, 30)
            ctx.hit(r, "act", enabled=False, kind="button")
        _frame(ctx, draw)
        assert ctx.dispatch((10, 10)) is None


class TestScroll:
    def test_wheel_scrolls_region_and_clamps(self, ctx):
        rect = pygame.Rect(0, 0, 200, 100)
        ctx.scroll_bounds["s1"] = (500, 100)
        def draw():
            ctx.hit(rect, None, kind="scroll")
            ctx.regions[-1]["id"] = "s1"
        _frame(ctx, draw)
        assert ctx.mouse_wheel((50, 50), -3) is True
        assert ctx.scroll_offsets["s1"] == 108
        for _ in range(20):
            ctx.mouse_wheel((50, 50), -3)
        assert ctx.scroll_offsets["s1"] == 400  # content 500 - view 100
        for _ in range(30):
            ctx.mouse_wheel((50, 50), 3)
        assert ctx.scroll_offsets["s1"] == 0

    def test_wheel_outside_region_ignored(self, ctx):
        ctx.scroll_bounds["s1"] = (500, 100)
        assert ctx.mouse_wheel((700, 500), -3) is False


class TestTextInput:
    def _focus(self, ctx, iid="inp"):
        st = ctx.inputs.setdefault(iid, {"text": "", "caret": 0})
        ctx.focused_input = iid
        return st

    def _key(self, k, mod=0):
        e = pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod)
        return e

    def test_text_entry_and_editing(self, ctx):
        st = self._focus(ctx)
        for ch in "abc":
            ev = pygame.event.Event(pygame.TEXTINPUT, text=ch)
            assert ctx.text_input_event(ev) is True
        assert st["text"] == "abc" and st["caret"] == 3
        ctx.key_down(self._key(pygame.K_LEFT))
        ctx.key_down(self._key(pygame.K_BACKSPACE))
        assert st["text"] == "ac"
        ctx.key_down(self._key(pygame.K_END))
        ev = pygame.event.Event(pygame.TEXTINPUT, text="z")
        ctx.text_input_event(ev)
        assert st["text"] == "acz"

    def test_delete_home_end(self, ctx):
        st = self._focus(ctx)
        st["text"], st["caret"] = "hello", 0
        ctx.key_down(self._key(pygame.K_DELETE))
        assert st["text"] == "ello"
        ctx.key_down(self._key(pygame.K_END))
        assert st["caret"] == 4
        ctx.key_down(self._key(pygame.K_HOME))
        assert st["caret"] == 0

    def test_enter_and_escape_blur(self, ctx):
        self._focus(ctx)
        ctx.key_down(self._key(pygame.K_RETURN))
        assert ctx.focused_input is None
        self._focus(ctx)
        ctx.key_down(self._key(pygame.K_ESCAPE))
        assert ctx.focused_input is None

    def test_keys_unconsumed_without_focus(self, ctx):
        assert ctx.key_down(self._key(pygame.K_BACKSPACE)) is False


class TestTheme:
    def test_dark_light_tokens_differ(self):
        T.set_theme("dark")
        dark_bg = T.C.panel
        T.set_theme("light")
        assert T.C.panel != dark_bg
        T.set_theme("dark")
        assert T.C.panel == dark_bg

    def test_all_tokens_present_both_themes(self):
        for name in ("bg", "panel", "text", "accent", "error",
                     "vx_centerline", "vx_grid"):
            T.set_theme("dark"); d = T.C.get(name)
            T.set_theme("light"); l = T.C.get(name)
            assert d != (255, 0, 255) and l != (255, 0, 255)
        T.set_theme("dark")
