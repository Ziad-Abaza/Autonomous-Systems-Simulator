"""
HOME / Library screen — the studio's front door.

Sections (left nav):
    TRACKS      — user library cards (search/sort/grid)
    RECENT      — recently opened tracks
    TEMPLATES   — bundled starting points
    DATASETS    — recorded datasets browser
    EXPERIMENTS — experiment registry
    SETTINGS    — theme, data root, ui scale

All data comes from services: TrackLibrary, StudioSettings,
ExperimentManager, dataset inspection — never from internals.
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, panel, button, icon_button, label, nav_item, list_row,
    section_label, text_input, scroll_begin, scroll_end, menu_draw,
    divider, _draw_text, toggle,
)
from sim_ui.thumbnails import thumbnail_for_file
from sim_project.library import TrackAsset
from sim_ui.screens import dialogs
from sim_ui.screens.datasets_panel import scan_datasets, format_bytes


def _ago(ts: float) -> str:
    if not ts:
        return "—"
    d = time.time() - ts
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // 86400)}d ago"


SECTIONS = [
    ("TRACKS", "sec_tracks"), ("RECENT", "sec_recent"),
    ("TEMPLATES", "sec_templates"), ("DATASETS", "sec_datasets"),
    ("EXPERIMENTS", "sec_experiments"), ("SETTINGS", "sec_settings"),
]


class HomeScreen:
    NAV_W = 190
    CARD_W, CARD_H = 224, 176
    CARD_GAP = 14

    def __init__(self, app):
        self.app = app
        self.section = "TRACKS"
        self.search = ""
        self.sort_key = "modified"   # modified|name|length
        self._thumbs: Dict[str, Optional[pygame.Surface]] = {}
        self._exp_detail: Optional[str] = None
        self._ds_detail: Optional[str] = None

    # ------------------------------------------------------------ helpers

    def _thumb(self, asset: TrackAsset) -> Optional[pygame.Surface]:
        if asset.path not in self._thumbs:
            lib = self.app.library if not asset.readonly else self.app.presets_lib
            self._thumbs[asset.path] = thumbnail_for_file(
                asset.path, cache_path=lib.thumb_path_for(asset.path)
                if not asset.readonly else None)
        return self._thumbs[asset.path]

    def invalidate_thumbs(self, path: Optional[str] = None) -> None:
        if path:
            self._thumbs.pop(path, None)
        else:
            self._thumbs.clear()

    # --------------------------------------------------------------- draw

    def draw(self, ctx: UIContext) -> None:
        w, h = ctx.surface.get_size()
        ctx.surface.fill(T.C.bg)

        # left nav
        nav = pygame.Rect(0, 0, self.NAV_W, h)
        pygame.draw.rect(ctx.surface, T.C.bg_deep, nav)
        ctx.block(nav)
        _draw_text(ctx, ctx.fonts.title, "SIMULATION", T.C.text, 20, 18)
        _draw_text(ctx, ctx.fonts.title, "STUDIO", T.C.accent_line, 20, 40)
        y = 78
        for label_text, sec in SECTIONS:
            badge = None
            if sec == "TRACKS":
                badge = str(len(self.app.library.scan()))
            r = pygame.Rect(10, y, self.NAV_W - 20, 32)
            nav_item(ctx, r, label_text.title(), "nav",
                     sec.removeprefix("sec_").title(),
                     active=(self.section == sec.removeprefix("sec_").upper()),
                     badge=badge, badge_tone="info")
            y += 36
        # data root footer
        _draw_text(ctx, ctx.fonts.caption, "Data root:", T.C.text_faint,
                   14, h - 46)
        _draw_text(ctx, ctx.fonts.caption,
                   self.app.settings.data_root, T.C.text_dim,
                   14, h - 30, max_w=self.NAV_W - 24)

        body = pygame.Rect(self.NAV_W, 0, w - self.NAV_W, h)
        getattr(self, f"_draw_{self.section.lower()}")(ctx, body)

        if self.app.active_dialog == "new_track":
            from sim_env.templates import EnvironmentTemplateManager
            dialogs.draw_new_track(ctx, w, h,
                                   EnvironmentTemplateManager.list_templates())
        elif self.app.active_dialog == "confirm_delete":
            d = self.app.dialog_payload or {}
            dialogs.draw_confirm(ctx, w, h, "delete_track",
                                 "Delete Track",
                                 f"Permanently delete “{d.get('name','')}”?",
                                 confirm_label="Delete", danger=True,
                                 detail=d.get("path", ""))
        elif self.app.active_dialog == "rename":
            d = self.app.dialog_payload or {}
            dialogs.draw_rename(ctx, w, h, d.get("name", "Track"))

    # ------------------------------------------------------------ tracks

    def _draw_tracks(self, ctx: UIContext, body: pygame.Rect) -> None:
        x0, y0 = body.x + 24, body.y + 20
        _draw_text(ctx, ctx.fonts.h1, "Tracks", T.C.text, x0, y0)
        _draw_text(ctx, ctx.fonts.small, "Your environments", T.C.text_dim,
                   x0, y0 + 30)

        # toolbar: new + search + sort
        button(ctx, pygame.Rect(body.right - 150, y0 - 2, 126, 32),
               "+ New Track", "new_track", style="primary")
        text_input(ctx, "lib_search",
                   pygame.Rect(body.right - 430, y0 - 2, 190, 32),
                   placeholder="Search tracks…")
        sort_lbl = {"modified": "Recent", "name": "Name",
                    "length": "Length"}[self.sort_key]
        button(ctx, pygame.Rect(body.right - 232, y0 - 2, 74, 32),
               sort_lbl, "sort_cycle", tooltip="Cycle sort order")

        self.search = ctx.inputs.get("lib_search", {}).get("text", "")
        assets = self.app.library.scan()
        q = self.search.strip().lower()
        if q:
            assets = [a for a in assets if q in a.name.lower()
                      or q in a.description.lower()]
        if self.sort_key == "name":
            assets.sort(key=lambda a: a.name.lower())
        elif self.sort_key == "length":
            assets.sort(key=lambda a: a.length_m, reverse=True)
        else:
            assets.sort(key=lambda a: max(a.modified, a.last_opened),
                        reverse=True)
        assets.sort(key=lambda a: not a.favorite)

        area = pygame.Rect(body.x + 24, y0 + 56, body.w - 48,
                           body.bottom - y0 - 70)
        cols = max(1, area.w // (self.CARD_W + self.CARD_GAP))
        rows = (len(assets) + cols - 1) // cols
        content_h = rows * (self.CARD_H + self.CARD_GAP) + 8

        off = scroll_begin(ctx, "tracks_grid", area, content_h)
        if not assets:
            _draw_text(ctx, ctx.fonts.body,
                       "No tracks yet — create one from the button above "
                       "or start from a template.", T.C.text_faint,
                       area.x + 8, area.y + 20)
        for i, a in enumerate(assets):
            col, row = i % cols, i // cols
            cx = area.x + col * (self.CARD_W + self.CARD_GAP)
            cy = area.y + row * (self.CARD_H + self.CARD_GAP) - off
            if cy + self.CARD_H < area.top - 4 or cy > area.bottom + 4:
                continue
            self._draw_card(ctx, pygame.Rect(cx, cy, self.CARD_W, self.CARD_H), a)
        scroll_end(ctx, "tracks_grid", area)

    def _draw_card(self, ctx: UIContext, r: pygame.Rect,
                   a: TrackAsset) -> None:
        hover = ctx.hovered(r)
        pygame.draw.rect(ctx.surface, T.C.panel, r, border_radius=T.RADIUS)
        pygame.draw.rect(ctx.surface,
                         T.C.accent_line if hover else T.C.border, r, 1,
                         border_radius=T.RADIUS)
        ctx.block(r)

        th = self._thumb(a) if not a.broken else None
        tr = pygame.Rect(r.x + 1, r.y + 1, r.w - 2, 96)
        if th:
            ctx.surface.blit(pygame.transform.smoothscale(th, (tr.w, tr.h)), tr)
        else:
            pygame.draw.rect(ctx.surface, T.C.bg_deep, tr)
            msg = "Broken file" if a.broken else "No preview"
            _draw_text(ctx, ctx.fonts.caption, msg, T.C.text_faint,
                       tr.centerx - 40, tr.centery - 6)

        _draw_text(ctx, ctx.fonts.bold, a.name, T.C.text,
                   r.x + 10, tr.bottom + 6, max_w=r.w - 60)
        # open-document indicator: the card whose file is loaded shows a
        # marker; unsaved changes show a dirty dot
        app = self.app
        if (getattr(app, "studio_screen", None) == "workspace"
                and getattr(app.project, "file_path", None) == a.path):
            tag = "● unsaved" if app.dirty else "open"
            _draw_text(ctx, ctx.fonts.caption, tag,
                       T.C.warn if app.dirty else T.C.info,
                       r.right - 60, tr.bottom + 10)
        meta = (f"{a.point_count} pts · {a.entity_count} ent · "
                f"{a.length_m:.0f} m · v{a.env_version}")
        _draw_text(ctx, ctx.fonts.caption, meta, T.C.text_dim,
                   r.x + 10, tr.bottom + 26, max_w=r.w - 20)
        if a.description:
            _draw_text(ctx, ctx.fonts.caption, a.description, T.C.text_faint,
                       r.x + 10, tr.bottom + 42, max_w=r.w - 20)

        if a.favorite:
            _draw_text(ctx, ctx.fonts.small, "★", T.C.warn,
                       r.right - 22, r.y + 6)
        _draw_text(ctx, ctx.fonts.caption, _ago(a.last_opened or a.modified),
                   T.C.text_faint, r.x + 10, r.bottom - 20)

        button(ctx, pygame.Rect(r.right - 86, r.bottom - 34, 56, 26),
               "Open", "open_track", a.path, style="primary")
        icon_button(ctx, pygame.Rect(r.right - 26, r.bottom - 34, 22, 26),
                    "...", "card_menu", a.path, tooltip="Track actions")
        menu_draw(ctx, f"card:{a.path}",
                  pygame.Rect(r.right - 26, r.bottom - 34, 22, 26),
                  [("Open", "m_open", a.path),
                   ("Favorite" if not a.favorite else "Unfavorite", "m_fav", a.path),
                   ("Rename…", "m_rename", a.path),
                   ("Duplicate", "m_dup", a.path),
                   ("Reveal in Folder", "m_reveal", a.path),
                   ("Delete…", "m_del", a.path)])

    # ------------------------------------------------------------ recent

    def _draw_recent(self, ctx: UIContext, body: pygame.Rect) -> None:
        x0, y0 = body.x + 24, body.y + 20
        _draw_text(ctx, ctx.fonts.h1, "Recent", T.C.text, x0, y0)
        rec = self.app.library.recents(12)
        if not rec:
            _draw_text(ctx, ctx.fonts.body, "Nothing opened yet.",
                       T.C.text_faint, x0, y0 + 50)
            return
        y = y0 + 46
        for a in rec:
            r = pygame.Rect(x0, y, body.w - 48, 44)
            list_row(ctx, r, a.name, "open_track", a.path,
                     secondary=(a.description or
                                f"{a.point_count} pts · {a.length_m:.0f} m"),
                     right=_ago(a.last_opened))
            y += 48

    # ---------------------------------------------------------- templates

    def _draw_templates(self, ctx: UIContext, body: pygame.Rect) -> None:
        x0, y0 = body.x + 24, body.y + 20
        _draw_text(ctx, ctx.fonts.h1, "Templates", T.C.text, x0, y0)
        _draw_text(ctx, ctx.fonts.small,
                   "Starting points — instantiating a template creates a "
                   "new track in your library.", T.C.text_dim, x0, y0 + 30)
        from sim_env.templates import EnvironmentTemplateManager
        tmpls = EnvironmentTemplateManager.list_templates()
        presets = self.app.presets_lib.scan() if hasattr(
            self.app, "presets_lib") else []
        y = y0 + 58
        for t in tmpls:
            r = pygame.Rect(x0, y, body.w - 48, 52)
            pygame.draw.rect(ctx.surface, T.C.panel, r, border_radius=T.RADIUS)
            pygame.draw.rect(ctx.surface, T.C.border, r, 1,
                             border_radius=T.RADIUS)
            ctx.block(r)
            _draw_text(ctx, ctx.fonts.bold, t["name"], T.C.text,
                       r.x + 12, r.y + 8)
            _draw_text(ctx, ctx.fonts.caption, t["description"], T.C.text_dim,
                       r.x + 12, r.y + 28, max_w=r.w - 150)
            button(ctx, pygame.Rect(r.right - 104, r.y + 12, 92, 28),
                   "Use", "use_template", t["id"], style="primary")
            y += 60
        # bundled preset files
        if presets:
            y = section_label(ctx, "Bundled tracks", x0, y + 10) + 8
            for a in self.app.presets_lib.scan():
                r = pygame.Rect(x0, y, body.w - 48, 40)
                list_row(ctx, r, a.name, "open_preset", a.path,
                         secondary=f"{a.point_count} pts · {a.length_m:.0f} m",
                         right="preset")
                y += 44

    # ----------------------------------------------------------- datasets

    def _draw_datasets(self, ctx: UIContext, body: pygame.Rect) -> None:
        from sim_ui.screens.datasets_panel import draw_datasets_browser
        draw_datasets_browser(ctx, body, self.app)

    # --------------------------------------------------------- experiments

    def _draw_experiments(self, ctx: UIContext, body: pygame.Rect) -> None:
        from sim_ui.screens.experiments_panel import draw_experiments_browser
        draw_experiments_browser(ctx, body, self.app)

    # ------------------------------------------------------------ settings

    def _draw_settings(self, ctx: UIContext, body: pygame.Rect) -> None:
        x0, y0 = body.x + 24, body.y + 20
        _draw_text(ctx, ctx.fonts.h1, "Settings", T.C.text, x0, y0)
        s = self.app.settings
        y = y0 + 50

        _draw_text(ctx, ctx.fonts.bold, "Theme", T.C.text, x0, y)
        toggle(ctx, pygame.Rect(x0 + 200, y - 4, 300, 24),
               "Light theme", s.theme == "light", "set_theme_toggle")
        y += 40

        _draw_text(ctx, ctx.fonts.bold, "Data root", T.C.text, x0, y)
        _draw_text(ctx, ctx.fonts.small, s.data_root, T.C.text_dim,
                   x0, y + 20, max_w=body.w - 220)
        button(ctx, pygame.Rect(body.right - 130, y, 106, 28),
               "Choose…", "set_data_root")
        _draw_text(ctx, ctx.fonts.caption,
                   "Recordings and exported datasets are stored here.",
                   T.C.text_faint, x0, y + 40)
        y += 70

        _draw_text(ctx, ctx.fonts.bold, "UI scale", T.C.text, x0, y)
        for i, v in enumerate((0.9, 1.0, 1.15, 1.3)):
            r = pygame.Rect(x0 + 200 + i * 64, y - 4, 58, 26)
            button(ctx, r, f"{v:.2f}×", "set_ui_scale", v,
                   style="primary" if abs(s.ui_scale - v) < 0.01 else "default")
        y += 46

        _draw_text(ctx, ctx.fonts.caption,
                   "Changes apply immediately and persist in "
                   "studio_settings.json.", T.C.text_faint, x0, y + 8)

    # -------------------------------------------------------------- actions

    def on_action(self, action: str, payload: Any) -> bool:
        app = self.app
        if action == "nav":
            self.section = str(payload).upper()
        elif action == "sort_cycle":
            order = ["modified", "name", "length"]
            self.sort_key = order[(order.index(self.sort_key) + 1) % len(order)]
        elif action == "open_track":
            app.open_project_path(payload)
        elif action == "open_preset":
            app.open_project_path(payload)
        elif action == "new_track":
            app.active_dialog = "new_track"
            ctx_inputs = app.ui_ctx.inputs
            ctx_inputs["nt_name"] = {"text": "Untitled Track", "caret": 13}
        elif action == "nt_template":
            dialogs.draw_new_track.selected = payload
        elif action == "use_template":
            app.new_track_from_template(payload)
        elif action == "card_menu":
            app.ui_ctx.open_menu_id = f"card:{payload}"
        elif action == "m_open":
            app.open_project_path(payload)
        elif action == "m_fav":
            a = next((x for x in app.library.scan() if x.path == payload), None)
            app.library.set_favorite(payload, not (a and a.favorite))
        elif action == "m_dup":
            app.library.duplicate(payload)
            self.invalidate_thumbs()
        elif action == "m_reveal":
            app.reveal_in_folder(payload)
        elif action == "m_rename":
            a = next((x for x in app.library.scan() if x.path == payload),
                     None)
            app.active_dialog = "rename"
            app.dialog_payload = {"path": payload,
                                  "name": a.name if a else ""}
            app.ui_ctx.inputs["rn_name"] = {
                "text": a.name if a else "", "caret": len(a.name) if a else 0}
        elif action == "m_del":
            a = next((x for x in app.library.scan() if x.path == payload), None)
            app.active_dialog = "confirm_delete"
            app.dialog_payload = {"path": payload,
                                  "name": a.name if a else payload}
        elif action == "set_theme_toggle":
            s = app.settings
            s.theme = "light" if s.theme == "dark" else "dark"
            s.save()
            import sim_ui.theme as _t
            _t.set_theme(s.theme)
        elif action == "set_data_root":
            app.choose_data_root()
        elif action == "set_ui_scale":
            app.settings.ui_scale = payload
            app.settings.save()
            app.rebuild_fonts()
        else:
            return False
        return True
