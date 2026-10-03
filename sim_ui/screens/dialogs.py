"""
Modal dialogs for the Simulation Studio.

Dialogs are plain draw-functions over UIContext rendered at Z_MODAL
above a full-screen scrim that swallows clicks. They communicate via
ctx actions:

    dlg_confirm:<id>   — user accepted (payload: dialog result dict)
    dlg_cancel:<id>    — user cancelled/dismissed

Active dialog lives on the app/screen as `active_dialog`; ESC handling
is the caller's job.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, panel, button, label, text_input, section_label,
    _draw_text,
)


def _scrim(ctx: UIContext, w: int, h: int) -> None:
    ctx.z = UIContext.Z_MODAL
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((T.C.scrim[0], T.C.scrim[1], T.C.scrim[2], 170))
    ctx.surface.blit(s, (0, 0))
    ctx.block(pygame.Rect(0, 0, w, h))  # swallow everything below modal


def _dialog_box(ctx: UIContext, w: int, h: int, dw: int, dh: int,
                title: str) -> pygame.Rect:
    rect = pygame.Rect((w - dw) // 2, (h - dh) // 2, dw, dh)
    pygame.draw.rect(ctx.surface, T.C.panel, rect, border_radius=T.RADIUS)
    pygame.draw.rect(ctx.surface, T.C.border, rect, 1, border_radius=T.RADIUS)
    ctx.block(rect)
    _draw_text(ctx, ctx.fonts.title, title, T.C.text,
               rect.x + T.PAD + 4, rect.y + T.PAD, max_w=rect.w - 32)
    return rect


def draw_confirm(ctx: UIContext, w: int, h: int, dlg_id: str,
                 title: str, message: str,
                 confirm_label: str = "Confirm",
                 danger: bool = False, detail: str = "") -> None:
    """Generic confirmation dialog."""
    _scrim(ctx, w, h)
    rect = _dialog_box(ctx, w, h, 440, 168 if detail else 148, title)
    _draw_text(ctx, ctx.fonts.body, message, T.C.text_dim,
               rect.x + T.PAD + 4, rect.y + 44, max_w=rect.w - 32)
    if detail:
        _draw_text(ctx, ctx.fonts.caption, detail, T.C.text_faint,
                   rect.x + T.PAD + 4, rect.y + 66, max_w=rect.w - 32)
    bw, bh, by = 120, 30, rect.bottom - 44
    button(ctx, pygame.Rect(rect.right - bw - 14, by, bw, bh),
           confirm_label, f"dlg_confirm:{dlg_id}",
           style="danger" if danger else "primary")
    button(ctx, pygame.Rect(rect.right - 2 * bw - 22, by, bw, bh),
           "Cancel", f"dlg_cancel:{dlg_id}")


def draw_new_track(ctx: UIContext, w: int, h: int,
                   templates: List[Dict[str, str]]) -> None:
    """New-track dialog: name + template pick. Returns via dlg_confirm:new."""
    _scrim(ctx, w, h)
    rect = _dialog_box(ctx, w, h, 560, 360, "Create New Track")

    y = rect.y + 48
    _draw_text(ctx, ctx.fonts.caption, "NAME", T.C.text_dim, rect.x + 16, y)
    y += 16
    text_input(ctx, "nt_name", pygame.Rect(rect.x + 16, y, rect.w - 32, 30),
               placeholder="My Track")
    y += 44

    _draw_text(ctx, ctx.fonts.caption, "START FROM", T.C.text_dim, rect.x + 16, y)
    y += 16
    sel = getattr(draw_new_track, "selected", "empty")
    row_h = 34
    for i, t in enumerate(templates):
        r = pygame.Rect(rect.x + 16, y + i * row_h, rect.w - 32, row_h - 4)
        active = (sel == t["id"])
        bg = T.C.accent_soft if active else (T.C.hover if ctx.hovered(r) else T.C.panel_alt)
        pygame.draw.rect(ctx.surface, bg, r, border_radius=T.RADIUS_SM)
        _draw_text(ctx, ctx.fonts.body, t["name"], T.C.text,
                   r.x + 10, r.y + 5, max_w=r.w - 20)
        _draw_text(ctx, ctx.fonts.caption, t["description"], T.C.text_dim,
                   r.x + 10, r.y + 17, max_w=r.w - 20)
        ctx.hit(r, "nt_template", t["id"])
    y += len(templates) * row_h + 8

    bw = 130
    by = rect.bottom - 44
    name = ctx.inputs.get("nt_name", {}).get("text", "").strip()
    button(ctx, pygame.Rect(rect.right - bw - 14, by, bw, 30),
           "Create Track", "dlg_confirm:new",
           payload={"name": name, "template_id": sel},
           style="primary", enabled=bool(name))
    button(ctx, pygame.Rect(rect.right - 2 * bw - 22, by, bw, 30),
           "Cancel", "dlg_cancel:new")


def draw_recording(ctx: UIContext, w: int, h: int,
                   dest_dir: str, default_name: str) -> None:
    """Recording setup dialog: destination + name + start."""
    _scrim(ctx, w, h)
    rect = _dialog_box(ctx, w, h, 540, 260, "Record Episode")

    y = rect.y + 48
    _draw_text(ctx, ctx.fonts.caption, "SAVE LOCATION", T.C.text_dim,
               rect.x + 16, y)
    y += 16
    _draw_text(ctx, ctx.fonts.small, dest_dir, T.C.text_faint,
               rect.x + 16, y + 6, max_w=rect.w - 140)
    button(ctx, pygame.Rect(rect.right - 116, y, 100, 26),
           "Choose…", "rec_browse")
    y += 40

    _draw_text(ctx, ctx.fonts.caption, "RECORDING NAME", T.C.text_dim,
               rect.x + 16, y)
    y += 16
    text_input(ctx, "rec_name", pygame.Rect(rect.x + 16, y, rect.w - 32, 30),
               placeholder=default_name)
    y += 46

    _draw_text(ctx, ctx.fonts.caption,
               "Frames, actions, reward breakdown and schemas are stored "
               "per episode with a manifest.", T.C.text_faint,
               rect.x + 16, y, max_w=rect.w - 32)
    bw = 130
    by = rect.bottom - 44
    name = ctx.inputs.get("rec_name", {}).get("text", "").strip() or default_name
    button(ctx, pygame.Rect(rect.right - bw - 14, by, bw, 30),
           "Start Recording", "dlg_confirm:record",
           payload={"name": name, "dest": dest_dir}, style="primary")
    button(ctx, pygame.Rect(rect.right - 2 * bw - 22, by, bw, 30),
           "Cancel", "dlg_cancel:record")


def draw_recording_summary(ctx: UIContext, w: int, h: int,
                           summary: Dict[str, Any]) -> None:
    """Post-stop recording summary."""
    _scrim(ctx, w, h)
    rect = _dialog_box(ctx, w, h, 460, 240, "Recording Saved")
    rows = [
        ("Steps", str(summary.get("steps", 0))),
        ("Duration", f"{summary.get('duration_s', 0):.1f} s"),
        ("Return", f"{summary.get('total_return', 0.0):+.2f}"),
        ("Termination", str(summary.get("termination", "—"))),
        ("File", str(summary.get("path", ""))),
    ]
    y = rect.y + 46
    for k, v in rows:
        _draw_text(ctx, ctx.fonts.small, k, T.C.text_dim, rect.x + 16, y)
        _draw_text(ctx, ctx.fonts.small, v, T.C.text,
                   rect.x + 130, y, max_w=rect.w - 150)
        y += 22
    bw = 130
    by = rect.bottom - 44
    button(ctx, pygame.Rect(rect.right - bw - 14, by, bw, 30),
           "Open Folder", "dlg_confirm:rec_open_folder",
           payload=summary.get("path"))
    button(ctx, pygame.Rect(rect.right - 2 * bw - 22, by, bw, 30),
           "Close", "dlg_cancel:rec_summary")
