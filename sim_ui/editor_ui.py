"""
EDIT workspace — toolbar + outline + canvas + inspector + status bar.

Layout (inside the workspace body rect):

    toolbar (36px)   [Select][Draw] [Place ▸] | [Snap][Grid] |
                     [Fit All][Fit Sel] [View ▸]           zoom controls
    left rail (230)  OUTLINE: control points / entities / spawn / gates
    canvas (flex)    2D spline editor (VisualTrackEditor)
    right rail       EnvironmentInspector (docked)
    status bar       mode · tool · cursor world coords · zoom · save state

All interactions register through UIContext; editor canvas mouse events
are routed by the workspace shell.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, button, icon_button, label, nav_item, list_row,
    section_label, scroll_begin, scroll_end, menu_draw, toggle,
    _draw_text,
)


TOOLS = [("select", "Select"), ("draw", "Draw Pt")]
PLACE_ITEMS = [
    ("Obstacle", "obstacle"), ("Barrier", "barrier"), ("Cone", "cone"),
    ("Sign", "traffic_sign"), ("Light", "traffic_light"),
    ("Spawn Point", "spawn"),
]
VIEW_ITEMS = [
    ("Grid", "view_grid"), ("Curvature", "view_curv"),
    ("Direction", "view_tang"), ("Width", "view_width"),
]


class EditorUI:
    RAIL_W = 230
    TB_H = 36
    SB_H = 26
    INSP_W = 350

    def __init__(self, app):
        self.app = app
        self.show_outline = True
        self.show_inspector = True
        self.canvas_rect = pygame.Rect(0, 0, 100, 100)
        self._place_anchor = pygame.Rect(0, 0, 0, 0)
        self._view_anchor = pygame.Rect(0, 0, 0, 0)

    # ------------------------------------------------------------ layout

    def layout(self, body: pygame.Rect) -> Dict[str, pygame.Rect]:
        y = body.y
        toolbar = pygame.Rect(body.x, y, body.w, self.TB_H)
        y += self.TB_H
        sb = pygame.Rect(body.x, body.bottom - self.SB_H,
                         body.w, self.SB_H)
        content = pygame.Rect(body.x, y, body.w, sb.top - y)
        x = content.x
        if self.show_outline:
            outline = pygame.Rect(x, y, self.RAIL_W, content.h)
            x += self.RAIL_W
        else:
            outline = pygame.Rect(x, y, 0, content.h)
        if self.show_inspector:
            insp = pygame.Rect(content.right - self.INSP_W, y,
                               self.INSP_W, content.h)
        else:
            insp = pygame.Rect(content.right, y, 0, content.h)
        canvas = pygame.Rect(x, y, max(40, insp.x - x), content.h)
        return {"toolbar": toolbar, "outline": outline, "canvas": canvas,
                "inspector": insp, "status": sb, "content": content}

    # -------------------------------------------------------------- draw

    def draw(self, ctx: UIContext, body: pygame.Rect) -> None:
        ed = self.app.track_editor
        lay = self.layout(body)
        self.canvas_rect = lay["canvas"]
        ed.canvas_rect = lay["canvas"]

        self._draw_toolbar(ctx, lay["toolbar"])
        self._draw_outline(ctx, lay["outline"])

        # canvas
        cv = lay["canvas"]
        pygame.draw.rect(ctx.surface, T.C.canvas, cv)
        ctx.hit(cv, None, kind="canvas")  # forwarded to the 2D editor
        if getattr(ed, "_needs_frame", False):
            ed._needs_frame = False
            ed.frame_all()
        gates = (self.app.env.track.checkpoints
                 if self.app.env.track else None)
        ed.draw_editor(ctx.surface, cv, ctx.fonts.small,
                       self.app.env.track.spline if self.app.env.track else None,
                       gates)

        self._draw_inspector(ctx, lay["inspector"])
        self._draw_status(ctx, lay["status"])
        # Dropdowns must paint LAST — they hang below the toolbar into the
        # outline/canvas region, which would otherwise cover them.
        self._draw_menus(ctx)

    def _draw_menus(self, ctx: UIContext) -> None:
        ed = self.app.track_editor
        menu_draw(ctx, "ed_place", self._place_anchor,
                  [(lbl, "ed_place", tid) for lbl, tid in PLACE_ITEMS],
                  width=150)
        view_state = {"view_grid": ed.show_grid,
                      "view_curv": ed.show_curvature,
                      "view_tang": ed.show_tangents,
                      "view_width": ed.show_width_handles}
        items = [(("✓ " if view_state[a] else "   ") + lbl, a, None)
                 for lbl, a in VIEW_ITEMS]
        menu_draw(ctx, "ed_view", self._view_anchor, items, width=150)

    # ------------------------------------------------------------ toolbar

    def _draw_toolbar(self, ctx: UIContext, r: pygame.Rect) -> None:
        ed = self.app.track_editor
        pygame.draw.rect(ctx.surface, T.C.bg_deep, r)
        pygame.draw.line(ctx.surface, T.C.border,
                         (r.x, r.bottom - 1), (r.right, r.bottom - 1), 1)
        ctx.block(r)
        x = r.x + 6
        h = r.h - 8

        # tool cluster
        for tool, lbl in TOOLS:
            w = 58
            b = pygame.Rect(x, r.y + 4, w, h)
            button(ctx, b, lbl, "ed_tool", tool,
                   style="primary" if ed.tool == tool else "default",
                   tooltip="Select tool" if tool == "select"
                   else "Draw control points (click to add/insert)")
            x += w + 4
        x += 6
        # place menu
        b = pygame.Rect(x, r.y + 4, 74, h)
        button(ctx, b, "Place", "ed_place_menu",
               tooltip="Place entity on canvas")
        pygame.draw.polygon(ctx.surface, T.C.text_dim,
                            [(b.right - 16, b.centery - 3),
                             (b.right - 8, b.centery - 3),
                             (b.right - 12, b.centery + 3)])
        self._place_anchor = b
        x += 78
        x += 8
        # snap + grid
        b = pygame.Rect(x, r.y + 4, 54, h)
        button(ctx, b, "Snap", "ed_snap", style="primary" if ed.snap_enabled else "default",
               tooltip="Snap edits to 1 m grid (S)")
        x += 58
        b = pygame.Rect(x, r.y + 4, 52, h)
        button(ctx, b, "Grid", "ed_grid", style="primary" if ed.show_grid else "default",
               tooltip="Toggle grid (G)")
        x += 56
        x += 8
        # framing
        b = pygame.Rect(x, r.y + 4, 62, h)
        button(ctx, b, "Fit All", "ed_frame_all", tooltip="Frame entire track (A)")
        x += 66
        b = pygame.Rect(x, r.y + 4, 62, h)
        button(ctx, b, "Fit Sel", "ed_frame_sel", tooltip="Frame selection (F)")
        x += 70
        # loop toggle — explicit track topology (open route ↔ closed circuit)
        can_close = len(ed.road_def.control_points) >= 3
        loop_lbl = "Closed" if ed.road_def.is_closed else "Open"
        b = pygame.Rect(x, r.y + 4, 64, h)
        button(ctx, b, loop_lbl, "ed_loop",
               style="primary" if ed.road_def.is_closed else "default",
               enabled=can_close,
               tooltip="Closed circuit (laps)" if ed.road_def.is_closed
               else "Open route — click to close the loop")
        x += 68
        x += 6
        # view menu
        b = pygame.Rect(x, r.y + 4, 62, h)
        button(ctx, b, "View", "ed_view_menu", tooltip="Visualization toggles")
        pygame.draw.polygon(ctx.surface, T.C.text_dim,
                            [(b.right - 16, b.centery - 3),
                             (b.right - 8, b.centery - 3),
                             (b.right - 12, b.centery + 3)])
        self._view_anchor = b
        x += 70

        # right side: zoom
        zr = f"{int(round(ed.zoom * 25))}%"
        b = pygame.Rect(r.right - 116, r.y + 4, 26, h)
        icon_button(ctx, b, "−", "ed_zoom_out", tooltip="Zoom out")
        _draw_text(ctx, ctx.fonts.small, zr, T.C.text_dim,
                   r.right - 84, r.y + 12)
        b = pygame.Rect(r.right - 34, r.y + 4, 26, h)
        icon_button(ctx, b, "+", "ed_zoom_in", tooltip="Zoom in")

    # ------------------------------------------------------------ outline

    def _draw_outline(self, ctx: UIContext, r: pygame.Rect) -> None:
        if r.w <= 0:
            return
        ed = self.app.track_editor
        pygame.draw.rect(ctx.surface, T.C.panel, r)
        pygame.draw.line(ctx.surface, T.C.border,
                         (r.right - 1, r.y), (r.right - 1, r.bottom), 1)
        ctx.block(r)
        _draw_text(ctx, ctx.fonts.caption, "OUTLINE", T.C.text_faint,
                   r.x + 10, r.y + 8)

        entries = []  # (kind, label, secondary, payload, selected)
        entries.append(("sec", f"CONTROL POINTS "
                        f"({len(ed.road_def.control_points)})", "", None, False))
        for i, cp in enumerate(ed.road_def.control_points):
            sel = (ed.selected_point_idx == i)
            entries.append(("cp", f"P{i}", f"({cp.x:.0f}, {cp.y:.0f})",
                            i, sel))
        entries.append(("sec", f"ENTITIES ({len(ed.entities)})", "", None, False))
        for ent in ed.entities:
            sel = (ed.selected_entity_id == ent.entity_id)
            entries.append(("ent", ent.name, ent.entity_type,
                            ent.entity_id, sel))
        entries.append(("sec", "SPAWN", "", None, False))
        sp = ed.road_def.spawn_point
        entries.append(("spawn", "Spawn Point",
                        f"({sp.x:.0f}, {sp.y:.0f})",
                        "spawn", ed.is_spawn_selected))
        entries.append(("sec", f"GATES ({ed.road_def.num_checkpoints})",
                        "", None, False))

        row_h = 26
        content_h = sum(24 if e[0] == "sec" else row_h for e in entries) + 16
        area = pygame.Rect(r.x + 4, r.y + 26, r.w - 8, r.h - 30)
        off = scroll_begin(ctx, "outline", area, content_h)
        y = area.y + 4 - off
        for kind, lbl, sec, payload, sel in entries:
            if kind == "sec":
                _draw_text(ctx, ctx.fonts.caption, lbl, T.C.text_faint,
                           area.x + 4, y + 4, max_w=area.w - 8)
                y += 24
                continue
            if y + row_h < area.top or y > area.bottom:
                y += row_h
                continue
            rr = pygame.Rect(area.x + 2, y, area.w - 4, row_h - 2)
            act = {"cp": "ol_sel_cp", "ent": "ol_sel_ent",
                   "spawn": "ol_sel_spawn"}[kind]
            list_row(ctx, rr, lbl, act, payload, secondary="", selected=sel,
                     right=sec)
            y += row_h
        scroll_end(ctx, "outline", area)

    # ---------------------------------------------------------- inspector

    def _draw_inspector(self, ctx: UIContext, r: pygame.Rect) -> None:
        if r.w <= 0:
            return
        ctx.block(r)  # clicks must not fall through to the canvas
        ur = self.app.ui_renderer
        fonts = {"title": ur.font_title, "bold": ur.font_bold,
                 "small": ur.font_small, "mono": ur.font_mono}
        buttons = self.app.inspector.draw(ctx.surface, r.x, r.y, r.w, r.h,
                                          fonts)
        insp = self.app.inspector
        for rect, action_id in buttons:
            tip = None
            if action_id.startswith("tab_"):
                tab = action_id[4:]
                tip = insp.TAB_TOOLTIPS.get(tab) or \
                    insp.TAB_LABELS.get(tab, tab).title() + " settings"
            ctx.hit(rect, action_id, tooltip=tip)

    # ---------------------------------------------------------- status

    def _draw_status(self, ctx: UIContext, r: pygame.Rect) -> None:
        ed = self.app.track_editor
        app = self.app
        pygame.draw.rect(ctx.surface, T.C.bg_deep, r)
        pygame.draw.line(ctx.surface, T.C.border,
                         (r.x, r.y), (r.right, r.y), 1)
        ctx.block(r)
        y = r.centery - 7
        x = r.x + 8
        _draw_text(ctx, ctx.fonts.small, "EDIT", T.C.accent_line, x, y)
        x += 44
        tool = ed.active_tool or ed.tool
        _draw_text(ctx, ctx.fonts.small, f"Tool: {tool}",
                   T.C.text, x, y)
        x += 110
        wx, wy = ed.screen_to_world(*ctx.mouse_pos)
        inside = self.canvas_rect.collidepoint(ctx.mouse_pos)
        _draw_text(ctx, ctx.fonts.mono,
                   f"({wx:.1f}, {wy:.1f}) m" if inside else "—",
                   T.C.text_dim, x, y)
        x += 130
        _draw_text(ctx, ctx.fonts.small,
                   f"{ed.zoom:.1f} px/m", T.C.text_dim, x, y)
        x += 80
        if ed.snap_enabled:
            _draw_text(ctx, ctx.fonts.small, "SNAP 1m", T.C.warn, x, y)
            x += 70
        dirty_txt = "● modified" if app.dirty else "saved"
        _draw_text(ctx, ctx.fonts.small, dirty_txt,
                   T.C.warn if app.dirty else T.C.ok, x, y)

        hint = ("A fit-all · F fit-sel · Del delete · Ctrl+S save · "
                "Ctrl+Z undo")
        _draw_text(ctx, ctx.fonts.caption, hint, T.C.text_faint,
                   r.right - ctx.fonts.caption.size(hint)[0] - 10,
                   r.centery - 6)

    # ------------------------------------------------------------- input

    def canvas_mouse_down(self, pos: Tuple[int, int], btn: int) -> bool:
        if not self.canvas_rect.collidepoint(pos):
            return False
        return self.app.track_editor.handle_mouse_down(pos, btn)

    def canvas_mouse_move(self, pos: Tuple[int, int]) -> None:
        self.app.track_editor.handle_mouse_move(pos)

    def canvas_mouse_up(self) -> None:
        self.app.track_editor.handle_mouse_up()

    def canvas_wheel(self, pos: Tuple[int, int], y: int) -> bool:
        if not self.canvas_rect.collidepoint(pos):
            return False
        self.app.track_editor.handle_mouse_wheel(y, pos)
        return True

    def on_action(self, action: str, payload: Any) -> bool:
        ed = self.app.track_editor
        app = self.app
        if action == "ed_tool":
            ed.tool = payload
            ed.active_tool = None
        elif action == "ed_place_menu":
            app.ui_ctx.open_menu_id = "ed_place" if app.ui_ctx.open_menu_id != "ed_place" else None
        elif action == "ed_place":
            ed.active_tool = payload
            app.ui_ctx.open_menu_id = None
        elif action == "ed_view_menu":
            app.ui_ctx.open_menu_id = "ed_view" if app.ui_ctx.open_menu_id != "ed_view" else None
        elif action == "ed_snap":
            ed.snap_enabled = not ed.snap_enabled
        elif action == "ed_grid" or action == "view_grid":
            ed.show_grid = not ed.show_grid
        elif action == "view_curv":
            ed.show_curvature = not ed.show_curvature
        elif action == "view_tang":
            ed.show_tangents = not ed.show_tangents
        elif action == "view_width":
            ed.show_width_handles = not ed.show_width_handles
        elif action == "ed_loop":
            ed = self.app.track_editor
            if len(ed.road_def.control_points) >= 3:
                ed.road_def.is_closed = not ed.road_def.is_closed
                self.app._editor_changed()
                self.app._status(
                    "Closed circuit" if ed.road_def.is_closed
                    else "Open route", "ok")
        elif action == "ed_frame_all":
            ed.frame_all()
        elif action == "ed_frame_sel":
            ed.frame_selected()
        elif action == "ed_zoom_in":
            ed.zoom = min(60.0, ed.zoom * 1.25)
        elif action == "ed_zoom_out":
            ed.zoom = max(0.05, ed.zoom / 1.25)
        elif action == "ol_sel_cp":
            ed.select_control_point(payload)
            app.inspector.select_control_point(payload)
        elif action == "ol_sel_ent":
            ed.select_entity(payload)
            app.inspector.select_entity(payload)
        elif action == "ol_sel_spawn":
            ed.select_spawn()
            app.inspector.active_tab = "TRACK"
        else:
            return False
        return True

    def handle_key(self, event: pygame.event.Event) -> bool:
        """Editor-mode shortcuts. Returns True if consumed."""
        ed = self.app.track_editor
        app = self.app
        k = event.key
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        if ctrl and k == pygame.K_s:
            app.save_project()
            return True
        if ctrl and k == pygame.K_z:
            app.undo()
            return True
        if ctrl and k in (pygame.K_y,) or (ctrl and event.mod & pygame.KMOD_SHIFT and k == pygame.K_z):
            app.redo()
            return True
        if k == pygame.K_a and not ctrl:
            ed.frame_all()
            return True
        if k == pygame.K_f and not ctrl:
            ed.frame_selected()
            return True
        if k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            if ed.delete_selected():
                app.mark_dirty()
            return True
        if k == pygame.K_g and not ctrl:
            ed.show_grid = not ed.show_grid
            return True
        if k == pygame.K_s and not ctrl:
            ed.snap_enabled = not ed.snap_enabled
            return True
        if k == pygame.K_v:
            ed.tool = "select"
            return True
        if k == pygame.K_p:
            ed.tool = "draw"
            return True
        if k == pygame.K_ESCAPE:
            if ed.active_tool:
                ed.active_tool = None
            elif ed.tool == "draw":
                ed.tool = "select"
            else:
                ed.deselect_all()
            return True
        return False
