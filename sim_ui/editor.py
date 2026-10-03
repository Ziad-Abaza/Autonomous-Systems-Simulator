"""
Interactive visual Track Editor supporting spline control points, variable road width,
elevation, banking, boundary selection, and 3D mesh regeneration.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import pygame
from sim_core.track.road_definition import RoadDefinition, ControlPoint, RoadBoundaryConfig
from sim_core.math_utils import Vec2, Vec3


class VisualTrackEditor:
    """
    Top-down 2D interactive editor for authoring tracks with control points and spline curves.
    """
    def __init__(self, road_def: RoadDefinition):
        self.road_def = road_def
        self.selected_point_idx: Optional[int] = None
        self.is_dragging = False

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
        """Handles mouse clicks for point selection and dragging."""
        sc_x, sc_y = screen_center
        # Check if clicked on any control point
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = self.world_to_screen(cp.x, cp.y, sc_x, sc_y)
            dx = pos[0] - sx
            dy = pos[1] - sy
            if dx * dx + dy * dy <= 144:  # 12px radius
                self.selected_point_idx = i
                if button == 1:  # Left click: start drag
                    self.is_dragging = True
                    return True
                elif button == 3:  # Right click: delete point if >= 4 points
                    if len(self.road_def.control_points) > 3:
                        self.road_def.control_points.pop(i)
                        self.selected_point_idx = None
                        return True

        if button == 1:
            # Clicked empty space: add new point at mouse location
            wx, wy = self.screen_to_world(pos[0], pos[1], sc_x, sc_y)
            # Insert after selected point, or at end
            insert_idx = (self.selected_point_idx + 1) if self.selected_point_idx is not None else len(self.road_def.control_points)
            new_cp = ControlPoint(x=wx, y=wy, z=0.0, width=12.0)
            self.road_def.control_points.insert(insert_idx, new_cp)
            self.selected_point_idx = insert_idx
            self.is_dragging = True
            return True

        return False

    def handle_mouse_up(self) -> None:
        self.is_dragging = False

    def handle_mouse_move(self, pos: Tuple[int, int], screen_center: Tuple[float, float]) -> None:
        if self.is_dragging and self.selected_point_idx is not None:
            if 0 <= self.selected_point_idx < len(self.road_def.control_points):
                sc_x, sc_y = screen_center
                wx, wy = self.screen_to_world(pos[0], pos[1], sc_x, sc_y)
                self.road_def.control_points[self.selected_point_idx].x = wx
                self.road_def.control_points[self.selected_point_idx].y = wy

    def draw_editor(self, surface: pygame.Surface, screen_rect: pygame.Rect, font: pygame.font.Font, track_spline: Any) -> None:
        """Renders 2D top-down track geometry, control points, and handles."""
        sc_x = screen_rect.centerx
        sc_y = screen_rect.centery

        # Grid lines
        grid_step = int(10.0 * self.zoom)
        if grid_step > 15:
            ox = int(sc_x + self.view_offset_x * self.zoom) % grid_step
            oy = int(sc_y - self.view_offset_y * self.zoom) % grid_step
            for x in range(screen_rect.left + ox, screen_rect.right, grid_step):
                pygame.draw.line(surface, (40, 45, 55, 120), (x, screen_rect.top), (x, screen_rect.bottom), 1)
            for y in range(screen_rect.top + oy, screen_rect.bottom, grid_step):
                pygame.draw.line(surface, (40, 45, 55, 120), (screen_rect.left, y), (screen_rect.right, y), 1)

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
                # Boundaries
                pygame.draw.lines(surface, (180, 180, 180, 200), track_spline.is_closed, pts_left, 2)
                pygame.draw.lines(surface, (180, 180, 180, 200), track_spline.is_closed, pts_right, 2)
                # Centerline
                pygame.draw.lines(surface, (255, 215, 0, 180), track_spline.is_closed, pts_center, 1)

        # 2. Draw control points and connecting polyline
        cp_screen_pts = []
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = self.world_to_screen(cp.x, cp.y, sc_x, sc_y)
            cp_screen_pts.append((sx, sy))

            is_sel = (i == self.selected_point_idx)
            color = (255, 80, 80) if is_sel else (80, 180, 255)
            radius = 9 if is_sel else 6

            # Circle for control point
            pygame.draw.circle(surface, color, (sx, sy), radius)
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), radius, 2)

            # Label index and width
            lbl = font.render(f"P{i} ({cp.width:.1f}m)", True, (240, 240, 240))
            surface.blit(lbl, (sx + 10, sy - 12))

        # Connecting straight reference lines between control points
        if len(cp_screen_pts) > 1:
            pygame.draw.lines(surface, (80, 130, 200, 100), self.road_def.is_closed, cp_screen_pts, 1)

        # Draw Spawn Point Indicator
        sp = self.road_def.spawn_point
        spx, spy = self.world_to_screen(sp.x, sp.y, sc_x, sc_y)
        pygame.draw.circle(surface, (0, 255, 120), (spx, spy), 8)
        # Arrow pointing along spawn yaw
        arrow_len = 22
        ax = int(spx + math.cos(sp.yaw) * arrow_len)
        ay = int(spy - math.sin(sp.yaw) * arrow_len)
        pygame.draw.line(surface, (0, 255, 120), (spx, spy), (ax, ay), 3)
        lbl_sp = font.render("START / SPAWN", True, (0, 255, 120))
        surface.blit(lbl_sp, (spx + 12, spy - 10))
