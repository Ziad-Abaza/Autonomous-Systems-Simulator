"""
Minimal retained-interaction UI layer for the Simulation Studio.

The app is immediate-mode pygame: every frame we draw everything onto an
alpha surface. This module adds the missing piece — a **hit registry that
is filled during the draw pass** and used to dispatch the *next* frame's
mouse events. Layout lives in exactly one place (the draw code), which
fixes the old "re-run draw inside the click handler" pattern.

Usage model per frame:

    ctx.begin_frame()
    screen.draw(ctx)            # draw + register regions
    # events dispatch against the registry of the previous frame:
    ctx.mouse_down(pos, btn)    # -> action string | consumed bool

Screens implement `on_action(action: str, payload)`.

Components (all pure functions over UIContext):
    panel, section_label, label, button, icon_button, nav_item,
    card, list_row, toggle, slider-ish stepper, divider,
    scroll_begin/scroll_end (clipped scroll regions),
    text_input (focused editing),
    menu_open/menu_draw (dropdowns),
    modal_begin/modal_end (blocking overlay),
    tooltip (deferred hover text).
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

import pygame

from sim_ui import theme as T


# ---------------------------------------------------------------- utils

def _with_alpha(color, a):
    return (color[0], color[1], color[2], a)


class Fonts:
    """Font set honoring a global UI scale."""

    def __init__(self, scale: float = 1.0):
        pygame.font.init()
        s = scale
        self.caption = pygame.font.SysFont("Segoe UI", max(10, round(T.FONT_CAPTION * s)))
        self.small = pygame.font.SysFont("Segoe UI", max(10, round(T.FONT_SMALL * s)))
        self.body = pygame.font.SysFont("Segoe UI", max(11, round(T.FONT_BODY * s)))
        self.bold = pygame.font.SysFont("Segoe UI", max(11, round(T.FONT_BODY * s)), bold=True)
        self.title = pygame.font.SysFont("Segoe UI", max(12, round(T.FONT_TITLE * s)), bold=True)
        self.h1 = pygame.font.SysFont("Segoe UI", max(14, round(T.FONT_H1 * s)), bold=True)
        self.mono = pygame.font.SysFont("Consolas", max(10, round(T.FONT_MONO * s)))


# ---------------------------------------------------------------- context

class UIContext:
    """Per-frame draw+hit context shared by all screens."""

    Z_BASE = 0
    Z_MENU = 100
    Z_MODAL = 200

    def __init__(self, surface: pygame.Surface, fonts: Fonts):
        self.surface = surface
        self.fonts = fonts
        self.mouse_pos = (0, 0)

        # hit registry for the frame being built (draw pass)
        self.regions: List[Dict[str, Any]] = []
        # registry snapshot used for dispatch (previous completed frame)
        self._dispatch_regions: List[Dict[str, Any]] = []  # unused (legacy)

        self.z = self.Z_BASE
        self._clip_stack: List[Optional[pygame.Rect]] = []

        # persistent widget state
        self.scroll_offsets: Dict[str, int] = {}
        self.scroll_bounds: Dict[str, Tuple[int, int]] = {}  # id -> (content_h, view_h)
        self.inputs: Dict[str, Dict[str, Any]] = {}          # id -> {text, caret}
        self.focused_input: Optional[str] = None
        self.open_menu_id: Optional[str] = None
        self.hover_start: float = 0.0
        self.hover_key: Optional[str] = None
        self.modal_open = False
        self.status_msg: Optional[Tuple[str, float, str]] = None  # text, expiry, tone

    # ---- frame lifecycle ----

    def begin_frame(self) -> None:
        # Regions are rebuilt every frame; dispatch always uses the last
        # *completed* frame's list (events are processed between renders,
        # never mid-draw), so a separate snapshot would lag one frame
        # behind what is actually on screen.
        self.regions = []
        self.z = self.Z_BASE
        self.modal_open = False

    # ---- hit registry ----

    def hit(self, rect: pygame.Rect, action: Optional[str],
            payload: Any = None, *, kind: str = "button",
            enabled: bool = True, tooltip: Optional[str] = None) -> None:
        if not enabled and kind == "button":
            return
        self.regions.append({
            "rect": rect, "action": action, "payload": payload,
            "kind": kind, "z": self.z, "enabled": enabled,
            "tooltip": tooltip,
        })

    def block(self, rect: pygame.Rect) -> None:
        """Register a swallow-everything region (panel background)."""
        self.hit(rect, None, kind="block")

    def dispatch(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Topmost region containing pos, honoring z-order then recency."""
        best = None
        for r in self.regions:
            if r["rect"].collidepoint(pos):
                if best is None or (r["z"], ) >= (best["z"], ) or r["z"] == best["z"]:
                    best = r
        return best

    def hovered(self, rect: pygame.Rect) -> bool:
        return rect.collidepoint(self.mouse_pos)

    def mouse_down(self, pos: Tuple[int, int], button: int) -> Optional[Dict[str, Any]]:
        """Returns the region that consumed the click, or None."""
        # text input focus management: click outside blurs
        r = self.dispatch(pos)
        if self.focused_input:
            if not r or r.get("kind") != "input" or r.get("id") != self.focused_input:
                self.focused_input = None
        if r is None:
            # click on empty space: close menu
            self.open_menu_id = None
            return None
        if r["kind"] == "input":
            self.focused_input = r["id"]
        if r["kind"] == "menu_item" or r["kind"] == "block":
            self.open_menu_id = None
        if r["kind"] in ("button", "input", "menu_item"):
            self.open_menu_id = r.get("menu_id") if r["kind"] == "menu_item" else None
        return r

    def mouse_wheel(self, pos: Tuple[int, int], y: int) -> bool:
        """Scroll the topmost scroll region under the cursor."""
        for r in reversed(self.regions):
            if r["kind"] == "scroll" and r["rect"].collidepoint(pos):
                sid = r["id"]
                content_h, view_h = self.scroll_bounds.get(sid, (0, 0))
                max_off = max(0, content_h - view_h)
                cur = self.scroll_offsets.get(sid, 0)
                self.scroll_offsets[sid] = max(0, min(max_off, cur - y * 36))
                return True
        return False

    def key_down(self, event: pygame.event.Event) -> bool:
        """Route editing keys to the focused text input. True = consumed."""
        if not self.focused_input:
            return False
        st = self.inputs.get(self.focused_input)
        if st is None:
            self.focused_input = None
            return False
        if event.key == pygame.K_ESCAPE:
            self.focused_input = None
            return True
        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
            self.focused_input = None
            return True
        if event.key == pygame.K_BACKSPACE:
            if st["caret"] > 0:
                st["text"] = st["text"][: st["caret"] - 1] + st["text"][st["caret"]:]
                st["caret"] -= 1
            return True
        if event.key == pygame.K_DELETE:
            st["text"] = st["text"][: st["caret"]] + st["text"][st["caret"] + 1:]
            return True
        if event.key == pygame.K_LEFT:
            st["caret"] = max(0, st["caret"] - 1)
            return True
        if event.key == pygame.K_RIGHT:
            st["caret"] = min(len(st["text"]), st["caret"] + 1)
            return True
        if event.key == pygame.K_HOME:
            st["caret"] = 0
            return True
        if event.key == pygame.K_END:
            st["caret"] = len(st["text"])
            return True
        if event.key == pygame.K_a and (event.mod & pygame.KMOD_CTRL):
            st["caret"] = len(st["text"])
            st["sel_anchor"] = 0
            return True
        return False

    def text_input_event(self, event: pygame.event.Event) -> bool:
        if not self.focused_input:
            return False
        st = self.inputs.get(self.focused_input)
        if st is None or not getattr(event, "text", ""):
            return False
        ch = event.text
        if ch.isprintable():
            st["text"] = st["text"][: st["caret"]] + ch + st["text"][st["caret"]:]
            st["caret"] += len(ch)
        return True

    def clip_push(self, rect: pygame.Rect) -> None:
        prev = self.surface.get_clip()
        self._clip_stack.append(prev)
        if prev:
            rect = rect.clip(prev)
        self.surface.set_clip(rect)

    def clip_pop(self) -> None:
        prev = self._clip_stack.pop() if self._clip_stack else None
        self.surface.set_clip(prev if prev else pygame.Rect(0, 0, *self.surface.get_size()))

    def status(self, text: str, tone: str = "info", seconds: float = 3.0) -> None:
        self.status_msg = (text, time.time() + seconds, tone)


# ---------------------------------------------------------------- drawing

def _draw_text(ctx: UIContext, font, text: str, color,
               x: int, y: int, max_w: int = 0) -> pygame.Rect:
    """Draw text with optional elision to max width."""
    if max_w and font.size(text)[0] > max_w:
        while text and font.size(text + "…")[0] > max_w:
            text = text[:-1]
        text = text + "…" if text else ""
    img = font.render(text, True, color)
    ctx.surface.blit(img, (x, y))
    return img.get_rect(topleft=(x, y))


def label(ctx: UIContext, text: str, x: int, y: int, *,
          font=None, color=None, max_w: int = 0) -> pygame.Rect:
    return _draw_text(ctx, font or ctx.fonts.body, text,
                      color or T.C.text, x, y, max_w)


def section_label(ctx: UIContext, text: str, x: int, y: int,
                  max_w: int = 0) -> int:
    r = _draw_text(ctx, ctx.fonts.caption, text.upper(), T.C.text_faint,
                   x, y, max_w)
    return r.bottom + 4


def panel(ctx: UIContext, rect: pygame.Rect, *, fill=None,
          border: bool = True, radius: int = T.RADIUS) -> None:
    pygame.draw.rect(ctx.surface, fill or T.C.panel, rect, border_radius=radius)
    if border:
        pygame.draw.rect(ctx.surface, T.C.border, rect, T.BORDER_W,
                         border_radius=radius)
    ctx.block(rect)


def divider(ctx: UIContext, x: int, y: int, w: int) -> int:
    pygame.draw.line(ctx.surface, T.C.separator, (x, y), (x + w, y), 1)
    return y + 1


def button(ctx: UIContext, rect: pygame.Rect, label_text: str,
           action: Optional[str], payload: Any = None, *,
           style: str = "default", enabled: bool = True,
           tooltip: Optional[str] = None, font=None,
           align: str = "center") -> None:
    hover = enabled and ctx.hovered(rect)
    if not enabled:
        bg, fg = T.C.panel_alt, T.C.text_faint
    elif style == "primary":
        bg, fg = (T.C.accent_line if hover else T.C.accent), T.C.text_on_accent
    elif style == "danger":
        bg, fg = (tuple(min(255, c + 18) for c in T.C.error) if hover else T.C.error), T.C.text_on_accent
    elif style == "ghost":
        bg, fg = (T.C.hover if hover else (0, 0, 0)), T.C.text
    else:
        bg, fg = (T.C.hover if hover else T.C.panel_alt), T.C.text
    if bg[3] if len(bg) > 3 else True:
        pygame.draw.rect(ctx.surface, bg, rect, border_radius=T.RADIUS_SM)
    if style != "ghost" or hover:
        pygame.draw.rect(ctx.surface, T.C.border_soft, rect, T.BORDER_W,
                         border_radius=T.RADIUS_SM)
    f = font or ctx.fonts.body
    tw = f.size(label_text)[0]
    tx = rect.centerx - tw // 2 if align == "center" else rect.x + T.PAD_SM
    _draw_text(ctx, f, label_text, fg, tx,
               rect.centery - f.get_height() // 2, max_w=rect.w - 8)
    if action:
        ctx.hit(rect, action, payload, enabled=enabled, tooltip=tooltip)


def icon_button(ctx: UIContext, rect: pygame.Rect, glyph: str,
                action: str, payload: Any = None, *,
                enabled: bool = True, active: bool = False,
                tooltip: Optional[str] = None) -> None:
    hover = enabled and ctx.hovered(rect)
    bg = T.C.accent_soft if active else (T.C.hover if hover else T.C.panel_alt)
    pygame.draw.rect(ctx.surface, bg, rect, border_radius=T.RADIUS_SM)
    pygame.draw.rect(ctx.surface, T.C.accent_line if active else T.C.border_soft,
                     rect, T.BORDER_W, border_radius=T.RADIUS_SM)
    f = ctx.fonts.body
    fg = T.C.text if enabled else T.C.text_faint
    ctx.surface.blit(f.render(glyph, True, fg),
                     (rect.centerx - f.size(glyph)[0] // 2,
                      rect.centery - f.get_height() // 2))
    ctx.hit(rect, action, payload, enabled=enabled, tooltip=tooltip)


def nav_item(ctx: UIContext, rect: pygame.Rect, text: str,
             action: str, payload: Any = None, *,
             active: bool = False, badge: Optional[str] = None,
             badge_tone: str = "info", tooltip: Optional[str] = None) -> None:
    hover = ctx.hovered(rect)
    bg = T.C.accent_soft if active else (T.C.hover if hover else T.C.panel)
    pygame.draw.rect(ctx.surface, bg, rect, border_radius=T.RADIUS_SM)
    if active:
        pygame.draw.rect(ctx.surface, T.C.accent_line,
                         (rect.x, rect.y + 4, 3, rect.h - 8))
    f = ctx.fonts.body
    _draw_text(ctx, f, text, T.C.text if active or hover else T.C.text_dim,
               rect.x + 12, rect.centery - f.get_height() // 2,
               max_w=rect.w - 20 - (18 if badge else 0))
    if badge:
        tone = {"info": T.C.info, "warn": T.C.warn, "error": T.C.error,
                "ok": T.C.ok}.get(badge_tone, T.C.info)
        bx = rect.right - 14
        pygame.draw.circle(ctx.surface, tone, (bx, rect.centery), 6)
        bf = ctx.fonts.caption
        ctx.surface.blit(bf.render(badge, True, T.C.bg_deep),
                         (bx - bf.size(badge)[0] // 2,
                          rect.centery - bf.get_height() // 2))
    ctx.hit(rect, action, payload, tooltip=tooltip)


def list_row(ctx: UIContext, rect: pygame.Rect, primary: str,
             action: str, payload: Any = None, *,
             secondary: str = "", right: str = "",
             selected: bool = False, enabled: bool = True,
             tooltip: Optional[str] = None) -> None:
    hover = enabled and ctx.hovered(rect)
    bg = T.C.accent_soft if selected else (T.C.hover if hover else T.C.panel)
    pygame.draw.rect(ctx.surface, bg, rect, border_radius=T.RADIUS_SM)
    _draw_text(ctx, ctx.fonts.body, primary,
               T.C.text if enabled else T.C.text_faint,
               rect.x + T.PAD_SM, rect.y + 4, max_w=rect.w - 90)
    if secondary:
        _draw_text(ctx, ctx.fonts.caption, secondary, T.C.text_dim,
                   rect.x + T.PAD_SM, rect.bottom - 16, max_w=rect.w - 90)
    if right:
        _draw_text(ctx, ctx.fonts.caption, right, T.C.text_faint,
                   rect.right - T.PAD_SM - ctx.fonts.caption.size(right)[0],
                   rect.centery - 7)
    ctx.hit(rect, action, payload, enabled=enabled, tooltip=tooltip)


def toggle(ctx: UIContext, rect: pygame.Rect, text: str, value: bool,
           action: str, payload: Any = None) -> None:
    box = pygame.Rect(rect.x, rect.centery - 7, 28, 14)
    pygame.draw.rect(ctx.surface, T.C.accent if value else T.C.panel_alt,
                     box, border_radius=7)
    pygame.draw.rect(ctx.surface, T.C.border, box, 1, border_radius=7)
    knob_x = box.right - 8 if value else box.x + 8
    pygame.draw.circle(ctx.surface, T.C.text, (knob_x, box.centery), 5)
    _draw_text(ctx, ctx.fonts.body, text, T.C.text, box.right + 8,
               rect.centery - ctx.fonts.body.get_height() // 2,
               max_w=rect.w - 40)
    ctx.hit(pygame.Rect(rect.x, rect.y, rect.w, rect.h), action, payload)


def scroll_begin(ctx: UIContext, sid: str, rect: pygame.Rect,
                 content_h: int) -> int:
    """Begin a clipped scroll region; returns the y-offset to apply."""
    ctx.scroll_bounds[sid] = (content_h, rect.h)
    max_off = max(0, content_h - rect.h)
    off = max(0, min(max_off, ctx.scroll_offsets.get(sid, 0)))
    ctx.scroll_offsets[sid] = off
    ctx.clip_push(rect)
    ctx.hit(rect, None, kind="scroll")
    ctx.regions[-1]["id"] = sid
    return off


def scroll_end(ctx: UIContext, sid: str, rect: pygame.Rect) -> None:
    ctx.clip_pop()
    content_h, view_h = ctx.scroll_bounds.get(sid, (0, 0))
    if content_h > view_h:  # scrollbar
        frac = view_h / content_h
        bar_h = max(24, int(rect.h * frac))
        off = ctx.scroll_offsets.get(sid, 0)
        bar_y = rect.y + int((rect.h - bar_h) * off / max(1, content_h - view_h))
        pygame.draw.rect(ctx.surface, T.C.border,
                         (rect.right - T.SCROLL_W + 4, rect.y, 4, rect.h),
                         border_radius=2)
        pygame.draw.rect(ctx.surface, T.C.text_faint,
                         (rect.right - T.SCROLL_W + 4, bar_y, 4, bar_h),
                         border_radius=2)


def text_input(ctx: UIContext, iid: str, rect: pygame.Rect,
               placeholder: str = "", *, label_text: str = "",
               max_chars: int = 200) -> str:
    """Focused single-line editor; returns current text."""
    st = ctx.inputs.setdefault(iid, {"text": "", "caret": 0})
    if label_text:
        _draw_text(ctx, ctx.fonts.caption, label_text, T.C.text_dim,
                   rect.x, rect.y - 14)
    focused = ctx.focused_input == iid
    pygame.draw.rect(ctx.surface, T.C.bg_deep if focused else T.C.panel_alt,
                     rect, border_radius=T.RADIUS_SM)
    pygame.draw.rect(ctx.surface,
                     T.C.accent_line if focused else T.C.border,
                     rect, 1, border_radius=T.RADIUS_SM)
    txt = st["text"]
    if not txt and not focused and placeholder:
        _draw_text(ctx, ctx.fonts.body, placeholder, T.C.text_faint,
                   rect.x + T.PAD_SM, rect.centery - 7, max_w=rect.w - 16)
    else:
        _draw_text(ctx, ctx.fonts.body, txt, T.C.text,
                   rect.x + T.PAD_SM, rect.centery - 7, max_w=rect.w - 16)
    if focused:
        cx = rect.x + T.PAD_SM + ctx.fonts.body.size(txt[: st["caret"]])[0]
        if int(time.time() * 2.2) % 2 == 0:
            pygame.draw.line(ctx.surface, T.C.text,
                             (cx, rect.y + 5), (cx, rect.bottom - 5), 1)
    ctx.hit(rect, None, kind="input")
    ctx.regions[-1]["id"] = iid
    st["text"] = st["text"][:max_chars]
    return st["text"]


def menu_open(ctx: UIContext, menu_id: str) -> None:
    ctx.open_menu_id = menu_id


def menu_draw(ctx: UIContext, menu_id: str, anchor: pygame.Rect,
              items: List[Tuple[str, str, Any]], width: int = 170) -> None:
    """Dropdown list; items = (label, action, payload)."""
    if ctx.open_menu_id != menu_id:
        return
    old_z, ctx.z = ctx.z, ctx.Z_MENU
    h = len(items) * 26 + 8
    rect = pygame.Rect(anchor.x, anchor.bottom + 2, width, h)
    pygame.draw.rect(ctx.surface, T.C.panel_alt, rect, border_radius=T.RADIUS_SM)
    pygame.draw.rect(ctx.surface, T.C.border, rect, 1, border_radius=T.RADIUS_SM)
    ctx.hit(rect, None, kind="block")
    y = rect.y + 4
    for lbl, action, payload in items:
        r = pygame.Rect(rect.x + 4, y, rect.w - 8, 24)
        if ctx.hovered(r):
            pygame.draw.rect(ctx.surface, T.C.hover, r, border_radius=3)
        _draw_text(ctx, ctx.fonts.body, lbl, T.C.text, r.x + 8, r.y + 5,
                   max_w=r.w - 12)
        ctx.hit(r, action, payload, kind="menu_item")
        ctx.regions[-1]["menu_id"] = menu_id
        y += 26
    ctx.z = old_z


def tooltip(ctx: UIContext) -> None:
    """Call at end of frame: draws tooltip for hovered region."""
    for r in reversed(ctx.regions):
        tip = r.get("tooltip")
        if tip and r["rect"].collidepoint(ctx.mouse_pos):
            key = f"{r['rect'].x},{r['rect'].y}:{tip}"
            if ctx.hover_key != key:
                ctx.hover_key, ctx.hover_start = key, time.time()
            if time.time() - ctx.hover_start < 0.35:
                return
            f = ctx.fonts.caption
            w, h = f.size(tip)
            bx = min(ctx.mouse_pos[0] + 14, ctx.surface.get_width() - w - 20)
            by = min(ctx.mouse_pos[1] + 18, ctx.surface.get_height() - h - 16)
            box = pygame.Rect(bx - 6, by - 3, w + 12, h + 6)
            pygame.draw.rect(ctx.surface, T.C.panel_alt, box, border_radius=4)
            pygame.draw.rect(ctx.surface, T.C.border, box, 1, border_radius=4)
            ctx.surface.blit(f.render(tip, True, T.C.text), (bx, by))
            return


def status_toast(ctx: UIContext, width: int, height: int) -> None:
    """Draw transient status message bottom-center."""
    if not ctx.status_msg:
        return
    text, expiry, tone = ctx.status_msg
    if time.time() > expiry:
        ctx.status_msg = None
        return
    f = ctx.fonts.body
    w = f.size(text)[0] + 28
    box = pygame.Rect((width - w) // 2, height - 64, w, 30)
    col = {"info": T.C.info, "ok": T.C.ok, "warn": T.C.warn,
           "error": T.C.error}.get(tone, T.C.info)
    pygame.draw.rect(ctx.surface, T.C.panel_alt, box, border_radius=6)
    pygame.draw.rect(ctx.surface, col, box, 1, border_radius=6)
    ctx.surface.blit(f.render(text, True, T.C.text),
                     (box.centerx - f.size(text)[0] // 2, box.y + 7))
