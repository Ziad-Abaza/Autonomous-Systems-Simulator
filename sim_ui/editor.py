"""
Interactive visual Track Editor supporting spline control points, variable road width,
elevation, banking, boundary selection, spawn placement, and 3D mesh regeneration.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import pygame
from sim_core.track.road_definition import RoadDefinition, ControlPoint, RoadBoundaryConfig, SpawnPoint
from sim_core.math_utils import Vec2, Vec3, clamp


class VisualTrackEditor:
    """
    Top-down 2D interactive editor for authoring tracks with control points,
    checkpoints, spawn location, and spline curves.
    """
    def __init__(self, road_def: RoadDefinition):
        self.road_def = road_def
        self.selected_point_idx: Optional[int] = 0 if road_def.control_points else None
        self.is_dragging_point = False
        self.is_panning = False
        self.is_placing_spawn = False
        self._pan_start = (0, 0)
        self._orig_view_offset = (0.0, 0.0)

        # 2D view transformation (world coordinates to screen pixels)
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.zoom = 4.0  # pixels per meter

    def world_to_screen(self, wx: float, wy: float, screen_center_x: float, screen_center_y: float) -> Tuple[int, int]:
        sx = int(screen_center_x + (wx + self.view_offset_x) * self.zoom)
        sy = int(screen_center_y - (wy + self.view_offset_y) * self.zoom)  # Y inverted for screen
        return sx, sy

    def screen_to_world(self, sx: int, sy: int, screen_center_x: float, screen_center_y: float) -> Tuple[float, float]:
        wx = (sx - screen_center_x) / self.zoom - self.view_offset_x
        wy = -(sy - screen_center_y) / self.zoom - self.view_offset_y
        return wx, wy

    def handle_mouse_down(self, pos: Tuple[int, int], button: int, screen_center: Tuple[float, float]) -> bool:
        """Handles mouse clicks for point selection, dragging, spawn placement, and panning."""
        sc_x, sc_y = screen_center

        # Middle click or Right click on canvas: Pan
        if button == 2:  # Middle button
            self.is_panning = True
            self._pan_start = pos
            self._orig_view_offset = (self.view_offset_x, self.view_offset_y)
            return True

        # Mode: Placing Spawn
        if self.is_placing_spawn and button == 1:
            wx, wy = self.screen_to_world(pos[0], pos[1], sc_x, sc_y)
            self.road_def.spawn_point.x = round(wx, 1)
            self.road_def.spawn_point.y = round(wy, 1)
            self.is_placing_spawn = False
            return True

        # Check if clicked on Spawn Point
        sp = self.road_def.spawn_point
        sp_sx, sp_sy = self.world_to_screen(sp.x, sp.y, sc_x, sc_y)
        if (pos[0] - sp_sx) ** 2 + (pos[1] - sp_sy) ** 2 <= 144:
            if button == 1:
                self.selected_point_idx = None
                return True

        # Check if clicked on any control point
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = self.world_to_screen(cp.x, cp.y, sc_x, sc_y)
            dx = pos[0] - sx
            dy = pos[1] - sy
            if dx * dx + dy * dy <= 144:  # 12px radius
                self.selected_point_idx = i
                if button == 1:  # Left click: start drag
                    self.is_dragging_point = True
                    return True
                elif button == 3:  # Right click: delete point if >= 4 points
                    if len(self.road_def.control_points) > 3:
                        self.road_def.control_points.pop(i)
                        self.selected_point_idx = max(0, i - 1)
                        return True

        # Right click on empty space: pan
        if button == 3:
            self.is_panning = True
            self._pan_start = pos
            self._orig_view_offset = (self.view_offset_x, self.view_offset_y)
            return True

        # Left click on empty space: add new control point
        if button == 1:
            wx, wy = self.screen_to_world(pos[0], pos[1], sc_x, sc_y)
            insert_idx = (self.selected_point_idx + 1) if self.selected_point_idx is not None else len(self.road_def.control_points)
            new_cp = ControlPoint(x=round(wx, 1), y=round(wy, 1), z=0.0, width=12.0)
            self.road_def.control_points.insert(insert_idx, new_cp)
            self.selected_point_idx = insert_idx
            self.is_dragging_point = True
            return True

        return False

    def handle_mouse_up(self) -> None:
        self.is_dragging_point = False
        self.is_panning = False

    def handle_mouse_wheel(self, y_offset: int) -> None:
        """Zooms centered at current view."""
        factor = 1.15 if y_offset > 0 else (1.0 / 1.15)
        self.zoom = clamp(self.zoom * factor, 0.5, 30.0)

    def handle_mouse_move(self, pos: Tuple[int, int], screen_center: Tuple[float, float]) -> None:
        sc_x, sc_y = screen_center

        if self.is_panning:
            dx = (pos[0] - self._pan_start[0]) / self.zoom
            dy = -(pos[1] - self._pan_start[1]) / self.zoom
            self.view_offset_x = self._orig_view_offset[0] + dx
            self.view_offset_y = self._orig_view_offset[1] + dy

        elif self.is_dragging_point and self.selected_point_idx is not None:
            if 0 <= self.selected_point_idx < len(self.road_def.control_points):
                wx, wy = self.screen_to_world(pos[0], pos[1], sc_x, sc_y)
                self.road_def.control_points[self.selected_point_idx].x = round(wx, 1)
                self.road_def.control_points[self.selected_point_idx].y = round(wy, 1)

    def draw_editor(
        self,
        surface: pygame.Surface,
        screen_rect: pygame.Rect,
        font: pygame.font.Font,
        track_spline: Any,
        checkpoints: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Renders 2D top-down track geometry, control points, checkpoints, and spawn."""
        sc_x = screen_rect.centerx
        sc_y = screen_rect.centery

        # Grid lines
        grid_step = int(10.0 * self.zoom)
        if grid_step > 15:
            ox = int(sc_x + self.view_offset_x * self.zoom) % grid_step
            oy = int(sc_y - self.view_offset_y * self.zoom) % grid_step
            for x in range(screen_rect.left + ox, screen_rect.right, grid_step):
                pygame.draw.line(surface, (35, 40, 50, 100), (x, screen_rect.top), (x, screen_rect.bottom), 1)
            for y in range(screen_rect.top + oy, screen_rect.bottom, grid_step):
                pygame.draw.line(surface, (35, 40, 50, 100), (screen_rect.left, y), (screen_rect.right, y), 1)

        # 1. Draw spline centerline and boundary ribbons if samples exist
        if track_spline and track_spline.samples:
            pts_center = []
            pts_left = []
            pts_right = []
            for s in track_spline.samples:
                sx, sy = self.world_to_screen(s.pos.x, s.pos.y, sc_x, sc_y)
                pts_center.append((sx, sy))

                half_w = s.width * 0.5
                lx = s.pos.x + s.normal.x * half_w
                ly = s.pos.y + s.normal.y * half_w
                rx = s.pos.x - s.normal.x * half_w
                ry = s.pos.y - s.normal.y * half_w
                pts_left.append(self.world_to_screen(lx, ly, sc_x, sc_y))
                pts_right.append(self.world_to_screen(rx, ry, sc_x, sc_y))

            if len(pts_center) > 2:
                # Road boundaries
                pygame.draw.lines(surface, (170, 175, 185, 200), track_spline.is_closed, pts_left, 2)
                pygame.draw.lines(surface, (170, 175, 185, 200), track_spline.is_closed, pts_right, 2)
                # Centerline
                pygame.draw.lines(surface, (255, 215, 0, 180), track_spline.is_closed, pts_center, 1)

        # 2. Draw Checkpoints gates and direction arrows
        if checkpoints:
            for cp in checkpoints:
                gl = cp['gate_left']
                gr = cp['gate_right']
                slx, sly = self.world_to_screen(gl.x, gl.y, sc_x, sc_y)
                srx, sry = self.world_to_screen(gr.x, gr.y, sc_x, sc_y)

                # Gate line
                cp_color = (0, 230, 200) if cp['index'] == 0 else (80, 200, 140, 160)
                pygame.draw.line(surface, cp_color, (slx, sly), (srx, sry), 2)

                # Direction arrow in the center of gate
                c_pos = cp.get('pos')
                c_tan = cp.get('tangent')
                if c_pos and c_tan:
                    cx, cy = self.world_to_screen(c_pos.x, c_pos.y, sc_x, sc_y)
                    arr_len = int(14 * (self.zoom / 4.0))
                    arr_len = max(8, min(24, arr_len))
                    tan_2d = Vec2(c_tan.x, c_tan.y).normalized()
                    ax = int(cx + tan_2d.x * arr_len)
                    ay = int(cy - tan_2d.y * arr_len)
                    pygame.draw.line(surface, cp_color, (cx, cy), (ax, ay), 2)

        # 3. Draw control points and connecting polyline
        cp_screen_pts = []
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = self.world_to_screen(cp.x, cp.y, sc_x, sc_y)
            cp_screen_pts.append((sx, sy))

            is_sel = (i == self.selected_point_idx)
            color = (255, 80, 80) if is_sel else (80, 180, 255)
            radius = 9 if is_sel else 6

            pygame.draw.circle(surface, color, (sx, sy), radius)
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), radius, 2)

            # Label index and width
            lbl = font.render(f"P{i} ({cp.width:.1f}m)", True, (240, 240, 240))
            surface.blit(lbl, (sx + 10, sy - 12))

        # Connecting straight reference lines between control points
        if len(cp_screen_pts) > 1:
            pygame.draw.lines(surface, (80, 130, 200, 90), self.road_def.is_closed, cp_screen_pts, 1)

        # 4. Draw Spawn Point Indicator
        sp = self.road_def.spawn_point
        spx, spy = self.world_to_screen(sp.x, sp.y, sc_x, sc_y)
        pygame.draw.circle(surface, (0, 255, 120), (spx, spy), 8)
        arrow_len = 24
        ax = int(spx + math.cos(sp.yaw) * arrow_len)
        ay = int(spy - math.sin(sp.yaw) * arrow_len)
        pygame.draw.line(surface, (0, 255, 120), (spx, spy), (ax, ay), 3)
        lbl_sp = font.render(f"SPAWN ({math.degrees(sp.yaw):.0f}°)", True, (0, 255, 120))
        surface.blit(lbl_sp, (spx + 12, spy - 10))

        # Placement Mode banner
        if self.is_placing_spawn:
            hint = font.render(">> CLICK CANVAS TO PLACE SPAWN POINT <<", True, (255, 220, 0))
            surface.blit(hint, (sc_x - hint.get_width() // 2, screen_rect.top + 20))
