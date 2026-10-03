"""
TRACK workspace — the studio's working surface.

Header: back-to-library · track name + dirty state · tabs · save · mode
badge. Body delegates per tab:

    EDIT      → EditorUI (toolbar + outline + canvas + inspector)
    SIMULATE  → live 3D viewport + telemetry/reward/camera panels
    REPLAY    → recorded-episode player (load, scrub, play)
    DATA      → recording & dataset browser

The workspace shell owns the top chrome and routes ctx actions; tab
content reuses existing HUD/inspector components where they still earn
their place.
"""
from __future__ import annotations
import os
import time
from typing import Any, Optional, Tuple

import pygame

from sim_ui import theme as T
from sim_ui.widgets import (
    UIContext, button, icon_button, label, nav_item, list_row,
    section_label, scroll_begin, scroll_end, _draw_text, divider,
)
from sim_ui.screens import dialogs
from sim_ui.screens.datasets_panel import draw_datasets_browser
from sim_ui.screens.dynamics_panel import (
    draw_dynamics_panel, handle_dynamics_action,
)


TABS = ["EDIT", "SIMULATE", "REPLAY", "DATA", "DYNAMICS"]


class WorkspaceScreen:
    HEADER_H = 46
    STATUS_H = 28

    def __init__(self, app):
        self.app = app
        from sim_ui.editor_ui import EditorUI
        self.editor_ui = EditorUI(app)

    # ----------------------------------------------------------------- draw

    def draw(self, ctx: UIContext) -> None:
        app = self.app
        w, h = ctx.surface.get_size()
        self._draw_header(ctx, w)
        body = pygame.Rect(0, self.HEADER_H, w, h - self.HEADER_H)

        if app.ws_tab == "EDIT":
            self.editor_ui.draw(ctx, body)
        elif app.ws_tab == "SIMULATE":
            self._draw_simulate(ctx, body)
        elif app.ws_tab == "REPLAY":
            self._draw_replay(ctx, body)
        elif app.ws_tab == "DATA":
            draw_datasets_browser(ctx, body, app)
            self._draw_statusbar(ctx, body)
        elif app.ws_tab == "DYNAMICS":
            draw_dynamics_panel(ctx, body, app)
            self._draw_statusbar(ctx, body)

        # modal dialogs
        dlg = app.active_dialog
        if dlg == "confirm_quit":
            dialogs.draw_confirm(ctx, w, h, "quit", "Unsaved Changes",
                                 "You have unsaved changes. Quit anyway?",
                                 confirm_label="Discard & Quit", danger=True)
        elif dlg == "confirm_new":
            dialogs.draw_confirm(ctx, w, h, "discard_new", "Discard Track",
                                 "Discard current track and its unsaved "
                                 "changes?", confirm_label="Discard",
                                 danger=True)
        elif dlg == "record":
            dialogs.draw_recording(ctx, w, h, app.record_dir,
                                   app.suggest_recording_name())
        elif dlg == "rec_summary":
            dialogs.draw_recording_summary(ctx, w, h,
                                           app.recording_summary or {})
        elif dlg == "confirm_delete":
            d = app.dialog_payload or {}
            dialogs.draw_confirm(ctx, w, h, "delete_track", "Delete Track",
                                 f"Permanently delete “{d.get('name','')}”?",
                                 confirm_label="Delete", danger=True,
                                 detail=d.get("path", ""))
        elif dlg == "confirm_del_ds":
            d = app.dialog_payload or {}
            dialogs.draw_confirm(ctx, w, h, "delete_ds", "Delete Dataset",
                                 f"Permanently delete {d.get('name','')}?",
                                 confirm_label="Delete", danger=True,
                                 detail=d.get("path", ""))

    # ----------------------------------------------------------------- header

    def _draw_header(self, ctx: UIContext, w: int) -> None:
        app = self.app
        r = pygame.Rect(0, 0, w, self.HEADER_H)
        pygame.draw.rect(ctx.surface, T.C.bg_deep, r)
        pygame.draw.line(ctx.surface, T.C.border, (0, r.bottom - 1),
                         (w, r.bottom - 1), 1)
        ctx.block(r)
        x = 10
        b = pygame.Rect(x, 8, 84, 30)
        button(ctx, b, "< Library", "ws_home", tooltip="Back to track library")
        x += 94

        # track name + dirty dot + version
        name = app.project.name if app.project else "Untitled"
        _draw_text(ctx, ctx.fonts.bold, name, T.C.text, x, 14, max_w=220)
        nx = x + min(220, ctx.fonts.bold.size(name)[0]) + 8
        if app.dirty:
            pygame.draw.circle(ctx.surface, T.C.warn, (nx, 23), 4)
            _draw_text(ctx, ctx.fonts.caption, "unsaved", T.C.warn,
                       nx + 10, 17)
        elif app.project is not None:
            _draw_text(ctx, ctx.fonts.caption,
                       f"v{app.project.environment_version}", T.C.text_faint,
                       nx, 17)

        # tabs — centered
        tab_w, tab_h = 96, 30
        total = len(TABS) * (tab_w + 6)
        tx = max(x + 240, (w - total) // 2)
        for tab in TABS:
            r = pygame.Rect(tx, 8, tab_w, tab_h)
            button(ctx, r, tab, "ws_tab", tab,
                   style="primary" if app.ws_tab == tab else "default")
            tx += tab_w + 6

        # right side: save + camera buttons (sim/replay)
        bx = w - 96
        b = pygame.Rect(bx, 8, 86, 30)
        button(ctx, b, "Save", "ws_save", style="primary",
               tooltip="Save track (Ctrl+S)")
        if app.ws_tab in ("SIMULATE", "REPLAY"):
            cx = bx - 4
            for mode, lbl in (("chase", "Chase"), ("hood", "Hood"),
                              ("top_down", "Top"), ("orbit", "Orbit")):
                cw = 56
                cx -= cw + 4
                b = pygame.Rect(cx, 8, cw, 30)
                cur = app.renderer.camera.mode
                button(ctx, b, lbl, "ws_cam", mode,
                       style="primary" if cur == mode else "default")

    # ----------------------------------------------------------------- sim

    def _draw_simulate(self, ctx: UIContext, body: pygame.Rect) -> None:
        app = self.app
        hud = app.hud
        fonts = {
            "title": app.ui_renderer.font_title,
            "bold": app.ui_renderer.font_bold,
            "small": app.ui_renderer.font_small,
            "mono": app.ui_renderer.font_mono,
        }
        w, h = ctx.surface.get_size()
        st = app.env.vehicle.state
        hud.draw_telemetry_hud(
            ctx.surface, 15, 55, app.step_info,
            st.speed, app.step_info.get("heading_error", 0.0),
            app.step_info.get("lateral_offset", 0.0),
            app.env.last_action, fonts)
        hud.draw_reward_inspector(
            ctx.surface, w - 315, 55,
            app.step_info.get("reward_breakdown", {}),
            app.env.reward_engine.total_accumulated_reward, fonts)
        # Camera PiP — every enabled camera sensor gets a stacked preview
        cam_y = 345
        for name, s in app.env.sensors.sensors.items():
            if getattr(s, "sensor_type", "") != "camera_rgb":
                continue
            hud.draw_camera_pip(ctx.surface, w - 315, cam_y, s, fonts)
            cam_y += 120
        if hud.show_obs_inspector:
            hud.draw_observation_inspector(
                ctx.surface, (w - 620) // 2, (h - 360) // 2,
                app.obs, app.env.observation_schema, app.step_info, fonts)
        self._draw_statusbar(ctx, body, sim=True)

    def _draw_statusbar(self, ctx: UIContext, body: pygame.Rect,
                        sim: bool = False) -> None:
        app = self.app
        r = pygame.Rect(0, body.bottom - self.STATUS_H,
                        body.w, self.STATUS_H)
        pygame.draw.rect(ctx.surface, T.C.bg_deep, r)
        pygame.draw.line(ctx.surface, T.C.border, (r.x, r.y), (r.right, r.y), 1)
        ctx.block(r)
        x = 10
        if sim:
            if app.recorder.is_recording:
                rec = pygame.Rect(x, r.y + 4, 96, 20)
                button(ctx, rec, "■ Stop", "ws_record_stop", style="danger")
                _draw_text(ctx, ctx.fonts.small,
                           f"REC {len(app.recorder.frames)} steps",
                           T.C.rec, x + 104, r.centery - 7)
            else:
                rec = pygame.Rect(x, r.y + 4, 96, 20)
                button(ctx, rec, "● Record", "ws_record_start",
                       tooltip="Configure + start episode recording")
            x += 130
            conn = ("AI CONNECTED" if app.server.is_client_connected
                    else "AI listening")
            _draw_text(ctx, ctx.fonts.small, conn,
                       T.C.ok if app.server.is_client_connected
                       else T.C.text_dim, x, r.centery - 7)
            x += 120
        _draw_text(ctx, ctx.fonts.small, app.ws_tab, T.C.accent_line,
                   x, r.centery - 7)
        hint = {"SIMULATE": "WASD drive · R reset · C camera · TAB inspect",
                "REPLAY": "Space play · arrows scrub · C camera",
                "DATA": "Datasets & episode recordings",
                "DYNAMICS": "Select maneuver · run · compare · export"}.get(app.ws_tab, "")
        _draw_text(ctx, ctx.fonts.caption, hint, T.C.text_faint,
                   r.right - ctx.fonts.caption.size(hint)[0] - 10,
                   r.centery - 6)
        if app.ws_tab != "DATA":
            dirty = "● modified" if app.dirty else "saved"
            _draw_text(ctx, ctx.fonts.caption, dirty,
                       T.C.warn if app.dirty else T.C.ok,
                       r.right - ctx.fonts.caption.size(hint)[0] - 96,
                       r.centery - 6)

    # ----------------------------------------------------------------- replay

    def _draw_replay(self, ctx: UIContext, body: pygame.Rect) -> None:
        app = self.app
        player = app.replay_player
        if not player.frames:
            self._draw_replay_picker(ctx, body)
        else:
            self._draw_replay_player(ctx, body)

    def _draw_replay_picker(self, ctx: UIContext, body: pygame.Rect) -> None:
        app = self.app
        x0 = body.x + 40
        _draw_text(ctx, ctx.fonts.h1, "Replay", T.C.text, x0, body.y + 24)
        _draw_text(ctx, ctx.fonts.small,
                   "Pick a recorded episode to inspect.", T.C.text_dim,
                   x0, body.y + 54)
        b = pygame.Rect(x0, body.y + 82, 130, 28)
        button(ctx, b, "Browse file…", "rp_browse")

        # list recordings from the recordings dir — cached ~2 s; the
        # manifest JSON is also loaded per-entry per frame below, so the
        # entry list AND subtitles ride on the same bucket.
        rec_dir = app.record_dir
        _bucket = int(time.time() / 2.0)
        if getattr(self, "_rp_bucket", -1) != _bucket \
                or getattr(self, "_rp_dir", None) != rec_dir:
            entries = []
            if os.path.isdir(rec_dir):
                for d in sorted(os.listdir(rec_dir), reverse=True):
                    ep = os.path.join(rec_dir, d, "episode.json")
                    if os.path.exists(ep):
                        entries.append((d, ep))
            # legacy root file
            legacy = os.path.join(os.path.dirname(app.settings.path),
                                  "last_episode.json")
            if os.path.exists(legacy):
                entries.append(("last_episode.json (legacy)", legacy))
            # Pre-read subtitles so per-frame manifest open() goes away.
            import json as _json
            enriched = []
            for name, path in entries:
                sec = ""
                try:
                    man_path = os.path.join(os.path.dirname(path),
                                            "manifest.json")
                    if os.path.exists(man_path):
                        with open(man_path, "r", encoding="utf-8") as f:
                            man = _json.load(f)
                        sec = (f"{man.get('track_name','?')} · "
                               f"{man.get('steps', 0)} steps")
                        name = man.get("name", name)
                except Exception:
                    pass
                enriched.append((name, path, sec))
            self._rp_entries = enriched
            self._rp_bucket = _bucket
            self._rp_dir = rec_dir
        entries = self._rp_entries

        y = body.y + 124
        _draw_text(ctx, ctx.fonts.caption, "RECORDED EPISODES",
                   T.C.text_faint, x0, y)
        y += 20
        if not entries:
            _draw_text(ctx, ctx.fonts.body,
                       "No recordings yet — record one from SIMULATE.",
                       T.C.text_faint, x0, y)
            return
        area = pygame.Rect(x0, y, min(520, body.w - 80),
                           body.bottom - y - 40)
        content_h = len(entries) * 44 + 8
        off = scroll_begin(ctx, "rp_list", area, content_h)
        for i, (name, path, sec) in enumerate(entries):
            r = pygame.Rect(area.x, area.y + i * 44 - off, area.w - 6, 40)
            if r.bottom < area.top or r.top > area.bottom:
                continue
            if not sec:
                sec = "episode.json"
            right = time.strftime("%m-%d %H:%M",
                                  time.localtime(os.path.getmtime(path)))
            list_row(ctx, r, name, "rp_open", path, secondary=sec,
                     right=right)
        scroll_end(ctx, "rp_list", area)

    def _draw_replay_player(self, ctx: UIContext, body: pygame.Rect) -> None:
        app = self.app
        player = app.replay_player
        # info chip top-left of viewport: name · track · time · frame
        meta = player.metadata
        cur_t = 0.0
        dur = 0.0
        if player.frames:
            cur_t = player.frames[player.current_frame_idx].get("t", 0.0)
            dur = player.frames[-1].get("t", 0.0)
        info = (f"{os.path.basename(app.replay_path) if app.replay_path else meta.get('track_name','Recording')}"
                f"  ·  {meta.get('track_name','?')}"
                f"  ·  {cur_t:.1f}s / {dur:.1f}s"
                f"  ·  {player.current_frame_idx + 1}/{player.total_frames}"
                f"  ·  {player.playback_speed:.1f}x")
        _draw_text(ctx, ctx.fonts.small, info, T.C.text, 16, body.y + 12)

        # bottom transport bar
        r = pygame.Rect(0, body.bottom - 64, body.w, 64)
        pygame.draw.rect(ctx.surface, (T.C.bg_deep[0], T.C.bg_deep[1],
                                       T.C.bg_deep[2], 220), r)
        ctx.block(r)
        y = r.y + 8
        # transport buttons
        x = r.x + 14
        for glyph, act, tip in (("|<", "rp_first", "First frame"),
                                ("<", "rp_back", "Step back"),
                                (">", "rp_play", "Play / pause (Space)"),
                                (">>", "rp_fwd", "Step forward"),
                                (">|", "rp_last", "Last frame")):
            b = pygame.Rect(x, y, 34, 26)
            icon_button(ctx, b, glyph, act, tooltip=tip)
            x += 38
        # speed
        for sp in (0.5, 1.0, 2.0, 4.0):
            b = pygame.Rect(x + 6, y, 40, 26)
            button(ctx, b, f"{sp:g}×", "rp_speed", sp,
                   style="primary" if abs(player.playback_speed - sp) < 0.01
                   else "default")
            x += 46
        # scrub track
        sx = x + 12
        sw = r.right - sx - 120
        track = pygame.Rect(sx, r.centery - 6, sw, 8)
        pygame.draw.rect(ctx.surface, T.C.panel_alt, track, border_radius=4)
        if player.total_frames > 1:
            frac = player.current_frame_idx / (player.total_frames - 1)
            pygame.draw.rect(ctx.surface, T.C.accent,
                             (track.x, track.y,
                              int(track.w * frac), track.h), border_radius=4)
        ctx.hit(track, "rp_scrub", {"track": (track.x, track.w)})
        btn = pygame.Rect(r.right - 104, y, 92, 26)
        button(ctx, btn, "Close", "rp_close")

    # ----------------------------------------------------------------- actions

    def on_action(self, action: str, payload: Any) -> bool:
        app = self.app
        if action == "ws_home":
            app.go_home()
        elif action == "ws_tab":
            app.set_ws_tab(payload)
        elif action == "ws_save":
            app.save_project()
        elif action == "ws_cam":
            app.renderer.camera.mode = payload
        elif action == "ws_record_start":
            app.active_dialog = "record"
        elif action == "rec_browse":
            # choose destination dir for the pending recording — persists
            # so the last-chosen folder survives restarts.
            path = app._tk("askdirectory", initialdir=app.record_dir,
                           title="Recording destination")
            if path:
                app.record_dir = path
                app.settings.set_recordings_dir(path)
                app.settings.save()
                app.ui_ctx.inputs["rec_dir"]["text"] = path
        elif action == "ws_record_stop":
            app.stop_recording()
        elif action == "rp_browse":
            app.browse_recording_file()
        elif action == "rp_open":
            app.load_replay(payload)
        elif action == "rp_play":
            player = app.replay_player
            player.toggle_play()
        elif action == "rp_first":
            app.replay_player.seek(0)
        elif action == "rp_last":
            app.replay_player.seek(app.replay_player.total_frames - 1)
        elif action == "rp_back":
            app.replay_player.step_backward()
        elif action == "rp_fwd":
            app.replay_player.step_forward()
        elif action == "rp_speed":
            app.replay_player.playback_speed = float(payload)
        elif action == "rp_scrub":
            tx, tw = payload["track"]
            frac = (app.ui_ctx.mouse_pos[0] - tx) / max(1, tw)
            n = app.replay_player.total_frames
            app.replay_player.seek(int(frac * max(0, n - 1)))
        elif action == "rp_close":
            app.replay_player.frames = []
            app.replay_player.metadata = {}
        elif action.startswith("dyn_"):
            return handle_dynamics_action(app, action, payload)
        elif action == "ds_select":
            app.dataset_detail = payload
        elif action == "ds_open_folder":
            app.reveal_in_folder(payload)
        elif action == "ds_export":
            # Copy the selected dataset/recording to a user-chosen folder.
            dest = app._tk("askdirectory", initialdir=payload,
                           title="Export dataset to…")
            if dest:
                try:
                    import shutil as _sh
                    tgt = os.path.join(dest, os.path.basename(payload))
                    _sh.copytree(payload, tgt)
                    app._status(f"Dataset exported → {tgt}", "ok")
                    app.reveal_in_folder(tgt)
                except Exception as e:
                    app._status(f"Export failed: {e}", "error")
        elif action == "ds_view_replay":
            app.load_replay(payload)
        elif action == "ds_delete":
            app.active_dialog = "confirm_del_ds"
            app.dialog_payload = {"path": payload,
                                  "name": os.path.basename(payload)}
        elif action == "ds_open_source":
            # dataset inside an experiment run dir → select its experiment
            p = os.path.normpath(payload).split(os.sep)
            root = os.path.normpath(app.exp_mgr.root_dir).split(os.sep)
            if p[:len(root)] == root and len(p) > len(root):
                app.train_state["selected_experiment"] = p[len(root)]
                app.ws_tab = "EDIT"
                app.inspector.active_tab = "TRAIN"
            else:
                app._status("Source experiment not found", "warn")
        elif action == "exp_select":
            app.train_state["selected_experiment"] = payload
        elif action == "exp_launch":
            if payload:
                app.train_state["selected_experiment"] = payload
            app._train_action("trn_launch")
        elif action == "exp_open_env":
            app.open_experiment_env(payload)
        elif action == "exp_export":
            if payload:
                try:
                    dest = os.path.join(app.exp_mgr.root_dir, "exported",
                                        payload)
                    app.exp_mgr.export(payload, dest)
                    app._status(f"Exported → {dest}", "ok")
                    app.reveal_in_folder(dest)
                except Exception as e:
                    app._status(f"Export failed: {e}", "error")
        else:
            return False
        return True
