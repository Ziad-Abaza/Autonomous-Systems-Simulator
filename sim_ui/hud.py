"""
Technical simulation HUD panels, telemetry overlays, live reward decomposition bars,
observation inspector, and camera Picture-in-Picture (PiP).
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple
import pygame
import numpy as np


class SimulationHUD:
    """
    Renders professional telemetry, reward breakdown, and observation tables.
    """
    def __init__(self):
        self.show_obs_inspector = False
        self.show_reward_details = True

    def draw_top_bar(
        self,
        surface: pygame.Surface,
        width: int,
        active_mode: str,
        camera_mode: str,
        is_server_running: bool,
        is_client_connected: bool,
        steps_served: int,
        fonts: Dict[str, pygame.font.Font]
    ) -> List[Tuple[pygame.Rect, str]]:
        """
        Draws top header bar with mode tabs, camera buttons, and server status.
        Returns clickable button rects: [(rect, action_id), ...]
        """
        bar_height = 42
        pygame.draw.rect(surface, (20, 24, 32, 230), (0, 0, width, bar_height))
        pygame.draw.line(surface, (45, 55, 70), (0, bar_height), (width, bar_height), 1)

        f_title = fonts['title']
        f_bold = fonts['bold']
        f_small = fonts['small']

        # Title / Brand
        title_surf = f_title.render("AI SIMULATION STUDIO", True, (0, 200, 255))
        surface.blit(title_surf, (15, 10))

        buttons = []
        x_cursor = 240

        # Mode Tabs: [Simulation, Track Editor, Replay]
        modes = [("SIMULATION", "mode_sim"), ("TRACK EDITOR", "mode_editor"), ("REPLAY", "mode_replay")]
        for label, mode_id in modes:
            is_active = (active_mode == mode_id)
            btn_rect = pygame.Rect(x_cursor, 7, 120, 28)
            bg_col = (0, 130, 230) if is_active else (35, 42, 55)
            txt_col = (255, 255, 255) if is_active else (180, 190, 205)
            pygame.draw.rect(surface, bg_col, btn_rect, border_radius=4)
            lbl = f_bold.render(label, True, txt_col)
            surface.blit(lbl, (btn_rect.centerx - lbl.get_width() // 2, btn_rect.centery - lbl.get_height() // 2))
            buttons.append((btn_rect, mode_id))
            x_cursor += 128

        x_cursor += 20

        # Camera Modes: [Chase, Hood, Top-Down, Orbit]
        cam_modes = [("CHASE", "cam_chase"), ("HOOD", "cam_hood"), ("TOP-DOWN", "cam_top_down"), ("ORBIT", "cam_orbit")]
        for label, cam_id in cam_modes:
            active_cam = f"cam_{camera_mode}"
            is_active = (active_cam == cam_id)
            btn_rect = pygame.Rect(x_cursor, 8, 80, 26)
            bg_col = (50, 100, 150) if is_active else (30, 36, 48)
            txt_col = (255, 255, 255) if is_active else (150, 160, 175)
            pygame.draw.rect(surface, bg_col, btn_rect, border_radius=3)
            lbl = f_small.render(label, True, txt_col)
            surface.blit(lbl, (btn_rect.centerx - lbl.get_width() // 2, btn_rect.centery - lbl.get_height() // 2))
            buttons.append((btn_rect, cam_id))
            x_cursor += 86

        # Right-side: External AI Server Status badge
        server_x = width - 260
        srv_rect = pygame.Rect(server_x, 8, 245, 26)
        if is_client_connected:
            badge_col = (30, 140, 60)
            status_text = f"EXTERNAL AI CONNECTED ({steps_served} steps)"
        elif is_server_running:
            badge_col = (180, 130, 20)
            status_text = "AI SERVER LISTENING (Port 8765)"
        else:
            badge_col = (90, 35, 35)
            status_text = "AI SERVER STOPPED"

        pygame.draw.rect(surface, (25, 30, 40), srv_rect, border_radius=4)
        pygame.draw.circle(surface, badge_col, (server_x + 14, 21), 6)
        lbl_srv = f_small.render(status_text, True, (220, 225, 235))
        surface.blit(lbl_srv, (server_x + 26, 12))

        return buttons

    def draw_telemetry_hud(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        info: Dict[str, Any],
        speed: float,
        heading_err: float,
        lat_offset: float,
        action: List[float],
        fonts: Dict[str, pygame.font.Font]
    ) -> None:
        """Draws left-side vehicle telemetry overlay."""
        w, h = 260, 240
        pygame.draw.rect(surface, (18, 22, 30, 210), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=6)

        f_bold = fonts['bold']
        f_mono = fonts['mono']
        f_small = fonts['small']

        # Header
        surface.blit(f_bold.render("VEHICLE TELEMETRY", True, (0, 200, 255)), (x + 12, y + 10))

        # Speed (km/h and m/s)
        speed_kmh = speed * 3.6
        speed_str = f"{speed_kmh:5.1f} km/h  ({speed:4.1f} m/s)"
        surface.blit(f_small.render("Speed:", True, (160, 170, 185)), (x + 12, y + 36))
        surface.blit(f_mono.render(speed_str, True, (255, 255, 255)), (x + 85, y + 36))

        # Distance from center
        surface.blit(f_small.render("Center Dev:", True, (160, 170, 185)), (x + 12, y + 60))
        lat_col = (100, 255, 100) if abs(lat_offset) < 1.5 else ((255, 200, 50) if abs(lat_offset) < 3.5 else (255, 80, 80))
        surface.blit(f_mono.render(f"{lat_offset:+5.2f} m", True, lat_col), (x + 110, y + 60))

        # Heading error
        head_deg = math.degrees(heading_err)
        surface.blit(f_small.render("Heading Err:", True, (160, 170, 185)), (x + 12, y + 84))
        head_col = (100, 255, 100) if abs(head_deg) < 15.0 else ((255, 200, 50) if abs(head_deg) < 40.0 else (255, 80, 80))
        surface.blit(f_mono.render(f"{head_deg:+5.1f}°", True, head_col), (x + 110, y + 84))

        # Laps & Checkpoints
        laps = info.get('laps_completed', 0)
        cps = info.get('checkpoints_passed', 0)
        surface.blit(f_small.render("Laps / Gates:", True, (160, 170, 185)), (x + 12, y + 108))
        surface.blit(f_mono.render(f"Lap {laps}  (CP {cps})", True, (240, 240, 240)), (x + 110, y + 108))

        # Collision status
        is_col = info.get('is_colliding', False)
        surface.blit(f_small.render("Collision:", True, (160, 170, 185)), (x + 12, y + 132))
        col_str = "CRASH!" if is_col else "CLEAR"
        surface.blit(f_bold.render(col_str, True, (255, 50, 50) if is_col else (50, 220, 80)), (x + 110, y + 130))

        # Action inputs meters
        steer, throttle, brake = action[0], action[1], action[2]
        surface.blit(f_small.render("Steer:", True, (140, 150, 165)), (x + 12, y + 160))
        self._draw_horizontal_meter(surface, x + 65, y + 165, 80, 8, (steer + 1.0) / 2.0, (80, 160, 255))
        surface.blit(f_mono.render(f"{steer:+4.2f}", True, (200, 210, 220)), (x + 155, y + 158))

        surface.blit(f_small.render("Throttle:", True, (140, 150, 165)), (x + 12, y + 182))
        self._draw_horizontal_meter(surface, x + 65, y + 187, 80, 8, throttle, (50, 220, 100))
        surface.blit(f_mono.render(f"{throttle:4.2f}", True, (200, 210, 220)), (x + 155, y + 180))

        surface.blit(f_small.render("Brake:", True, (140, 150, 165)), (x + 12, y + 204))
        self._draw_horizontal_meter(surface, x + 65, y + 209, 80, 8, brake, (255, 70, 70))
        surface.blit(f_mono.render(f"{brake:4.2f}", True, (200, 210, 220)), (x + 155, y + 202))

    def draw_reward_inspector(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        breakdown: Dict[str, float],
        total_accumulated: float,
        fonts: Dict[str, pygame.font.Font]
    ) -> None:
        """Draws live reward decomposition inspector panel."""
        w, h = 300, 280
        pygame.draw.rect(surface, (18, 22, 30, 210), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=6)

        f_bold = fonts['bold']
        f_mono = fonts['mono']
        f_small = fonts['small']

        # Header with Total Reward
        surface.blit(f_bold.render("REWARD DECOMPOSITION", True, (0, 200, 255)), (x + 12, y + 10))
        tot_str = f"Total: {total_accumulated:+7.2f}"
        surface.blit(f_mono.render(tot_str, True, (255, 230, 50)), (x + 175, y + 10))

        # Reward terms table
        row_y = y + 38
        terms = [
            ("Progress (ds)", breakdown.get('progress', 0.0), (80, 220, 120)),
            ("Centering", breakdown.get('centering', 0.0), (80, 200, 255)),
            ("Speed target", breakdown.get('speed', 0.0), (120, 180, 255)),
            ("Heading align", breakdown.get('heading', 0.0), (160, 140, 255)),
            ("Smoothness", breakdown.get('smoothness', 0.0), (220, 150, 100)),
            ("Checkpoint bonus", breakdown.get('checkpoint', 0.0), (255, 215, 0)),
            ("Lap bonus", breakdown.get('lap', 0.0), (255, 225, 50)),
            ("Collision penalty", breakdown.get('collision', 0.0), (255, 60, 60)),
            ("Off-road penalty", breakdown.get('off_road', 0.0), (255, 80, 80)),
        ]

        for label, val, col in terms:
            surface.blit(f_small.render(label, True, (170, 180, 195)), (x + 12, row_y))
            # Mini bar
            bar_w = int(min(60.0, max(-60.0, val * 15.0)))
            bar_center = x + 180
            if bar_w >= 0:
                pygame.draw.rect(surface, col, (bar_center, row_y + 3, bar_w, 8))
            else:
                pygame.draw.rect(surface, (255, 60, 60), (bar_center + bar_w, row_y + 3, -bar_w, 8))

            val_str = f"{val:+6.2f}"
            surface.blit(f_mono.render(val_str, True, col if val != 0.0 else (120, 130, 140)), (x + 245, row_y))
            row_y += 24

        # Step total
        step_tot = breakdown.get('total', 0.0)
        pygame.draw.line(surface, (50, 60, 80), (x + 10, row_y + 4), (x + w - 10, row_y + 4), 1)
        surface.blit(f_bold.render("Step Total:", True, (240, 240, 240)), (x + 12, row_y + 8))
        tot_col = (80, 255, 120) if step_tot >= 0 else (255, 80, 80)
        surface.blit(f_mono.render(f"{step_tot:+7.2f}", True, tot_col), (x + 235, row_y + 8))

    def draw_camera_pip(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        camera_sensor: Any,
        fonts: Dict[str, pygame.font.Font]
    ) -> None:
        """Renders Picture-in-Picture synthetic camera sensor preview."""
        w, h = 140, 140
        pygame.draw.rect(surface, (18, 22, 30, 230), (x, y, w, h + 24), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h + 24), 1, border_radius=6)

        surface.blit(fonts['bold'].render("CAMERA SENSOR", True, (0, 200, 255)), (x + 10, y + 6))

        if camera_sensor and hasattr(camera_sensor, '_last_image') and camera_sensor._last_image is not None:
            img = camera_sensor._last_image
            # Convert RGB array to Pygame Surface
            cam_surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            cam_scaled = pygame.transform.scale(cam_surf, (w - 8, h - 8))
            surface.blit(cam_scaled, (x + 4, y + 26))

    def draw_bottom_bar(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
        is_recording: bool,
        frames_recorded: int,
        replay_mode: bool,
        current_frame: int,
        total_frames: int,
        fonts: Dict[str, pygame.font.Font]
    ) -> List[Tuple[pygame.Rect, str]]:
        """Renders bottom episode recording & replay scrubber controls."""
        bar_height = 45
        y = height - bar_height
        pygame.draw.rect(surface, (20, 24, 32, 230), (0, y, width, bar_height))
        pygame.draw.line(surface, (45, 55, 70), (0, y), (width, y), 1)

        f_bold = fonts['bold']
        f_small = fonts['small']
        buttons = []

        # Record Button
        rec_rect = pygame.Rect(15, y + 8, 120, 28)
        rec_bg = (180, 40, 40) if is_recording else (40, 50, 65)
        pygame.draw.rect(surface, rec_bg, rec_rect, border_radius=4)
        rec_text = "REC ● STOP" if is_recording else "RECORD ●"
        lbl_rec = f_bold.render(rec_text, True, (255, 255, 255))
        surface.blit(lbl_rec, (rec_rect.centerx - lbl_rec.get_width() // 2, rec_rect.centery - lbl_rec.get_height() // 2))
        buttons.append((rec_rect, "btn_record"))

        # Frames count / status
        stat_text = f"Recorded: {frames_recorded} frames" if is_recording else "Ready"
        surface.blit(f_small.render(stat_text, True, (170, 180, 195)), (145, y + 14))

        # Replay controls if in replay mode
        if replay_mode:
            # Play / Pause
            play_rect = pygame.Rect(320, y + 8, 80, 28)
            pygame.draw.rect(surface, (0, 130, 220), play_rect, border_radius=4)
            surface.blit(f_bold.render("PLAY / ||", True, (255, 255, 255)), (play_rect.centerx - 26, play_rect.centery - 8))
            buttons.append((play_rect, "btn_replay_play"))

            # Scrubber track
            scrub_x = 420
            scrub_w = width - scrub_x - 120
            pygame.draw.rect(surface, (50, 60, 75), (scrub_x, y + 18, scrub_w, 8), border_radius=4)
            if total_frames > 0:
                frac = current_frame / max(1, total_frames - 1)
                knob_x = int(scrub_x + frac * scrub_w)
                pygame.draw.circle(surface, (0, 200, 255), (knob_x, y + 22), 8)
                lbl_f = f_small.render(f"{current_frame}/{total_frames}", True, (220, 220, 220))
                surface.blit(lbl_f, (scrub_x + scrub_w + 12, y + 14))

        # Keyboard Guide Hint
        hint = f_small.render("Drive: WASD / Arrow Keys | Reset: R | Camera: C | Toggle Obs: TAB", True, (130, 140, 155))
        surface.blit(hint, (width - hint.get_width() - 20, y + 14))

        return buttons

    def draw_editor_palette(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        selected_cp_idx: Optional[int],
        selected_cp: Any,
        fonts: Dict[str, pygame.font.Font]
    ) -> List[Tuple[pygame.Rect, str]]:
        """Editor side panel showing selected control point properties and actions."""
        w, h = 260, 310
        pygame.draw.rect(surface, (18, 22, 30, 220), (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=6)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']
        buttons = []

        surface.blit(f_bold.render("TRACK PROPERTIES", True, (0, 200, 255)), (x + 12, y + 10))

        if selected_cp is not None and selected_cp_idx is not None:
            surface.blit(f_small.render(f"Control Point #{selected_cp_idx}", True, (255, 255, 255)), (x + 12, y + 36))
            surface.blit(f_mono.render(f"X: {selected_cp.x:6.1f} m  Y: {selected_cp.y:6.1f} m", True, (180, 200, 220)), (x + 12, y + 58))
            surface.blit(f_mono.render(f"Z (Elev): {selected_cp.z:4.1f} m", True, (180, 200, 220)), (x + 12, y + 80))
            surface.blit(f_mono.render(f"Width:   {selected_cp.width:4.1f} m", True, (180, 200, 220)), (x + 12, y + 102))

            # Width Adjustment Buttons [-] [+]
            btn_minus_w = pygame.Rect(x + 12, y + 130, 110, 26)
            pygame.draw.rect(surface, (35, 45, 60), btn_minus_w, border_radius=3)
            surface.blit(f_small.render("Width -1m", True, (220, 220, 220)), (btn_minus_w.centerx - 30, btn_minus_w.centery - 8))
            buttons.append((btn_minus_w, "cp_width_minus"))

            btn_plus_w = pygame.Rect(x + 130, y + 130, 110, 26)
            pygame.draw.rect(surface, (35, 45, 60), btn_plus_w, border_radius=3)
            surface.blit(f_small.render("Width +1m", True, (220, 220, 220)), (btn_plus_w.centerx - 30, btn_plus_w.centery - 8))
            buttons.append((btn_plus_w, "cp_width_plus"))

            # Elevation Adjustment Buttons [-] [+]
            btn_minus_z = pygame.Rect(x + 12, y + 164, 110, 26)
            pygame.draw.rect(surface, (35, 45, 60), btn_minus_z, border_radius=3)
            surface.blit(f_small.render("Elev -0.5m", True, (220, 220, 220)), (btn_minus_z.centerx - 30, btn_minus_z.centery - 8))
            buttons.append((btn_minus_z, "cp_elev_minus"))

            btn_plus_z = pygame.Rect(x + 130, y + 164, 110, 26)
            pygame.draw.rect(surface, (35, 45, 60), btn_plus_z, border_radius=3)
            surface.blit(f_small.render("Elev +0.5m", True, (220, 220, 220)), (btn_plus_z.centerx - 30, btn_plus_z.centery - 8))
            buttons.append((btn_plus_z, "cp_elev_plus"))

            # Delete Point Button
            btn_del = pygame.Rect(x + 12, y + 200, 228, 28)
            pygame.draw.rect(surface, (150, 40, 40), btn_del, border_radius=4)
            surface.blit(f_bold.render("DELETE POINT", True, (255, 255, 255)), (btn_del.centerx - 48, btn_del.centery - 8))
            buttons.append((btn_del, "cp_delete"))
        else:
            surface.blit(f_small.render("Click canvas to add points.", True, (160, 170, 180)), (x + 12, y + 45))
            surface.blit(f_small.render("Click a point to drag/edit.", True, (160, 170, 180)), (x + 12, y + 70))
            surface.blit(f_small.render("Right click to delete point.", True, (160, 170, 180)), (x + 12, y + 95))

        # Rebuild 3D Track button
        btn_rebuild = pygame.Rect(x + 12, y + 250, 228, 36)
        pygame.draw.rect(surface, (0, 150, 100), btn_rebuild, border_radius=4)
        lbl_rb = f_bold.render("REBUILD 3D MESH", True, (255, 255, 255))
        surface.blit(lbl_rb, (btn_rebuild.centerx - lbl_rb.get_width() // 2, btn_rebuild.centery - lbl_rb.get_height() // 2))
        buttons.append((btn_rebuild, "track_rebuild"))

        return buttons

    def _draw_horizontal_meter(self, surface: pygame.Surface, x: int, y: int, w: int, h: int, value: float, color: tuple) -> None:
        """Draws small progress/value meter."""
        pygame.draw.rect(surface, (40, 48, 60), (x, y, w, h), border_radius=2)
        fill_w = int(max(0.0, min(1.0, value)) * w)
        if fill_w > 0:
            pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=2)

    def draw_observation_inspector(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        obs: Any,
        schema: Any,
        info: Dict[str, Any],
        fonts: Dict[str, pygame.font.Font]
    ) -> None:
        """
        Renders detailed Observation Inspector overlay explicitly separating
        Agent Observation (AI perception) from Privileged Oracle Telemetry.
        """
        w, h = 620, 360
        pygame.draw.rect(surface, (14, 18, 26, 240), (x, y, w, h), border_radius=8)
        pygame.draw.rect(surface, (45, 55, 75), (x, y, w, h), 1, border_radius=8)

        f_bold = fonts['bold']
        f_small = fonts['small']
        f_mono = fonts['mono']

        # Title
        surface.blit(f_bold.render("OBSERVATION & TELEMETRY INSPECTOR (TAB to close)", True, (0, 210, 255)), (x + 15, y + 10))
        pygame.draw.line(surface, (45, 55, 75), (x + 10, y + 34), (x + w - 10, y + 34), 1)

        col_w = (w - 30) // 2

        # --- LEFT: AGENT OBSERVATION ---
        lx = x + 15
        surface.blit(f_bold.render("[AI OBSERVATION (Policy Input)]", True, (80, 220, 120)), (lx, y + 42))

        # Check observation format
        if isinstance(obs, np.ndarray):
            dim = obs.shape[0] if obs.ndim > 0 else 1
            surface.blit(f_small.render(f"Type: Box({dim},)  |  Dtype: {obs.dtype}", True, (160, 175, 190)), (lx, y + 64))

            # Display first 10-12 feature values
            row_y = y + 88
            feature_names = []
            if getattr(schema, 'include_speed', True): feature_names.append("speed_norm")
            if getattr(schema, 'include_velocity', True): feature_names.extend(["vel_x", "vel_y"])
            if getattr(schema, 'include_yaw_rate', True): feature_names.append("yaw_rate")
            if getattr(schema, 'include_steering_angle', True): feature_names.append("steering")
            if getattr(schema, 'include_distance_from_center', True): feature_names.append("lat_offset")
            if getattr(schema, 'include_heading_error', True): feature_names.append("heading_err")
            if getattr(schema, 'include_distance_to_checkpoint', True): feature_names.append("dist_cp")
            if getattr(schema, 'include_lidar_rays', True):
                for k in range(min(7, dim - len(feature_names))):
                    feature_names.append(f"lidar_{k}")

            for idx in range(min(10, len(feature_names), len(obs))):
                val = float(obs[idx])
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                surface.blit(f_small.render(name, True, (170, 185, 200)), (lx, row_y))
                surface.blit(f_mono.render(f"{val:+6.3f}", True, (255, 255, 255)), (lx + 130, row_y))
                row_y += 22

            if dim > 10:
                surface.blit(f_small.render(f"... + {dim - 10} more features (LiDAR / image)", True, (130, 140, 155)), (lx, row_y + 4))

        elif isinstance(obs, dict):
            surface.blit(f_small.render("Type: Dict Observation", True, (160, 175, 190)), (lx, y + 64))
            row_y = y + 88
            for k, v in list(obs.items())[:8]:
                val_str = f"array{v.shape}" if isinstance(v, np.ndarray) else f"{v}"
                surface.blit(f_small.render(k, True, (170, 185, 200)), (lx, row_y))
                surface.blit(f_mono.render(val_str, True, (255, 255, 255)), (lx + 130, row_y))
                row_y += 22

        # --- RIGHT: ORACLE TELEMETRY (Privileged Ground Truth) ---
        rx = x + col_w + 15
        surface.blit(f_bold.render("[ORACLE / DEBUG TELEMETRY]", True, (255, 180, 50)), (rx, y + 42))
        surface.blit(f_small.render("Privileged state (NEVER sent to AI)", True, (220, 120, 80)), (rx, y + 64))

        oracle_rows = [
            ("Speed (Exact)", f"{info.get('speed', 0.0):.2f} m/s"),
            ("Heading Error", f"{math.degrees(info.get('heading_error', 0.0)):+.1f}°"),
            ("Lateral Offset", f"{info.get('lateral_offset', 0.0):+.2f} m"),
            ("Road Width", f"{info.get('road_width', 12.0):.1f} m"),
            ("On Road", "YES" if info.get('is_on_road', True) else "OFF ROAD"),
            ("Is Colliding", "COLLISION" if info.get('is_colliding', False) else "CLEAR"),
            ("Checkpoints", f"{info.get('checkpoints_passed', 0)} (CP {info.get('current_checkpoint', 0)})"),
            ("Laps Done", f"{info.get('laps_completed', 0)}"),
            ("Sim Time", f"{info.get('sim_time', 0.0):.2f} s"),
            ("Total Reward", f"{info.get('total_reward', 0.0):+.2f}"),
        ]

        row_y = y + 88
        for label, val_str in oracle_rows:
            surface.blit(f_small.render(label, True, (170, 185, 200)), (rx, row_y))
            surface.blit(f_mono.render(val_str, True, (255, 230, 80)), (rx + 135, row_y))
            row_y += 22

