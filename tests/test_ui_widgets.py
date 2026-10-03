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
from sim_ui.widgets import (UIContext, Fonts, scroll_begin, scroll_end,
                            menu_draw)


@pytest.fixture()
def ctx():
    T.set_theme("dark")
    surf = pygame.Surface((800, 600), pygame.SRCALPHA)
    return UIContext(surf, Fonts())


def _frame(ctx, draw_fn):
    ctx.begin_frame()
    draw_fn()
    # dispatch reads the current frame's regions — no promotion needed


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


class TestMenuDraw:
    def test_menu_clamped_to_surface(self, ctx):
        # anchor near the right edge of the 800-wide surface
        anchor = pygame.Rect(740, 10, 30, 20)
        ctx.open_menu_id = "m1"
        def draw():
            menu_draw(ctx, "m1", anchor, [("A", "a", None)], width=170)
        _frame(ctx, draw)
        item = next(r for r in ctx.regions if r["kind"] == "menu_item")
        assert item["rect"].right <= 800

    def test_menu_flips_up_when_no_room_below(self, ctx):
        anchor = pygame.Rect(10, 575, 30, 20)  # near bottom of 600-high
        ctx.open_menu_id = "m1"
        def draw():
            menu_draw(ctx, "m1", anchor,
                      [("A", "a", None), ("B", "b", None)], width=100)
        _frame(ctx, draw)
        item = next(r for r in ctx.regions if r["kind"] == "menu_item")
        assert item["rect"].bottom <= 600
        assert item["rect"].top < anchor.top

    def test_menu_closed_draws_nothing(self, ctx):
        def draw():
            menu_draw(ctx, "m1", pygame.Rect(10, 10, 30, 20),
                      [("A", "a", None)])
        _frame(ctx, draw)
        assert not any(r["kind"] == "menu_item" for r in ctx.regions)


class TestCardMenu:
    """The track-card '...' dropdown must paint after the scroll region —
    deferred via _open_card_menu — or it is clipped/painted over."""

    def _home(self, ctx, n=6):
        from types import SimpleNamespace
        from sim_ui.screens.home_screen import HomeScreen
        from sim_project.library import TrackAsset

        class _Lib:
            def __init__(self, assets):
                self._a = assets
            def scan(self):
                return self._a
            def thumb_path_for(self, p):
                return None

        assets = [TrackAsset(path=f"/t/{i}.sim.json",
                             file_name=f"{i}.sim.json", name=f"T{i}",
                             broken=True)  # skip thumbnail IO
                  for i in range(n)]
        app = SimpleNamespace(
            library=_Lib(assets),
            settings=SimpleNamespace(data_root="/tmp"),
            ui_ctx=ctx, active_dialog=None)
        return HomeScreen(app)

    def test_card_menu_paints_above_later_content(self, ctx):
        hs = self._home(ctx)
        ctx.begin_frame(); hs.draw(ctx)
        r = next(r for r in ctx.regions if r["action"] == "card_menu")
        ctx.mouse_down(r["rect"].center, 1)
        hs.on_action("card_menu", r["payload"])
        ctx.begin_frame(); hs.draw(ctx)
        item = next(r for r in ctx.regions
                    if r["kind"] == "menu_item" and r["action"] == "m_del")
        px = ctx.surface.get_at(item["rect"].center)[:3]
        assert px in (T.C.panel_alt[:3], T.C.hover[:3]), (
            f"menu item painted as {px} — clipped by scroll region?")

    def test_card_action_closes_menu(self, ctx):
        hs = self._home(ctx)
        ctx.open_menu_id = "card:/t/0.sim.json"
        hs.on_action("m_del", "/t/0.sim.json")
        assert ctx.open_menu_id is None
        assert hs.app.active_dialog == "confirm_delete"


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

    def test_palette_registry(self):
        pals = T.list_palettes()
        assert len(pals) >= 4
        assert len(T.list_palettes("dark")) >= 2
        assert len(T.list_palettes("light")) >= 2
        base_tokens = set(T.PALETTES["dark"]["colors"].keys())
        for p in pals:
            # every palette is a complete token set — variants only
            # override values, never drop keys
            assert set(p["colors"].keys()) == base_tokens
            assert p["id"] in T.PALETTES

    def test_set_theme_by_palette_id(self):
        base_accent = T.PALETTES["dark"]["colors"]["accent"]
        T.set_theme("dark_ocean")
        assert T.THEME_NAME == "dark_ocean"
        assert T.palette_mode(T.THEME_NAME) == "dark"
        assert T.C.accent != base_accent
        T.set_theme("light_solar")
        assert T.palette_mode(T.THEME_NAME) == "light"
        T.set_theme("dark")
        assert T.C.accent == base_accent

    def test_unknown_palette_falls_back_to_dark(self):
        T.set_theme("bogus_id")
        assert T.THEME_NAME == "dark"
        assert T.C.accent == T.PALETTES["dark"]["colors"]["accent"]

    def test_settings_theme_roundtrip(self, tmp_path):
        from sim_project.settings import StudioSettings
        s = StudioSettings(str(tmp_path / "s.json"))
        s.theme = "dark_ember"
        s.save()
        s2 = StudioSettings(str(tmp_path / "s.json"))
        assert s2.theme == "dark_ember"
        s2.theme = ""
        assert s2._data["theme"] == "dark"
