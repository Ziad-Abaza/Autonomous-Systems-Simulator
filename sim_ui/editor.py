"""
Interactive visual Environment Editor supporting track spline geometry, road width,
elevation, banking, curvature analysis, entity placement (obstacles, barriers, cones, signs, lights),
visual gizmos, and bidirectional scene selection.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
import pygame
from sim_core.track.road_definition import RoadDefinition, ControlPoint, RoadBoundaryConfig, SpawnPoint
from sim_core.world.entity import (
    WorldEntity, StaticObstacle, Barrier, TrafficCone,
    TrafficSign, TrafficLight, create_entity
)
from sim_core.math_utils import Vec2, Vec3, clamp, normalize_angle


class VisualTrackEditor:
    """
    Top-down 2D interactive editor for authoring tracks and placing environment entities.
    """
    def __init__(self, road_def: RoadDefinition, entities: Optional[List[WorldEntity]] = None):
        self.road_def = road_def
        self.entities = entities if entities is not None else []

        # Selection state
        self.selected_point_idx: Optional[int] = 0 if road_def.control_points else None
        self.selected_entity_id: Optional[str] = None
        self.selected_checkpoint_idx: Optional[int] = None
        self.is_spawn_selected: bool = False

        # Dragging / Interaction state
        self.is_dragging_point = False
        self.is_dragging_entity = False
        self.is_rotating_entity = False
        self.is_dragging_width = False
        self.is_panning = False

        # Placement tool mode: None, "obstacle", "barrier", "cone", "traffic_sign", "traffic_light", "spawn"
        self.active_tool: Optional[str] = None

        self._pan_start = (0, 0)
        self._orig_view_offset = (0.0, 0.0)
        self._drag_start_pos = (0.0, 0.0)

        # 2D view transformation (world coordinates to screen pixels)
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.zoom = 4.0  # pixels per meter
        # The drawable canvas region in window coordinates — set by the
        # editor UI each frame before input/draw. World (0,0) maps to its
        # center, NOT the window center.
        self.canvas_rect = pygame.Rect(0, 0, 800, 600)

        # Interaction tool: "select" (no geometry mutation) or "draw"
        # (control-point creation/insertion). Placement tools still use
        # self.active_tool.
        self.tool = "select"

        # Snapping
        self.snap_enabled = False
        self.snap_step = 1.0  # meters

        # Change notification — set by the app: called after any committed
        # mutation (for dirty tracking + undo snapshots)
        self.on_change = None

        # Display toggles
        self.show_curvature = True
        self.show_tangents = True
        self.show_width_handles = True
        self.show_grid = True

    def get_selected_entity(self) -> Optional[WorldEntity]:
        if self.selected_entity_id is None:
            return None
        for ent in self.entities:
            if ent.entity_id == self.selected_entity_id:
                return ent
        return None

    def select_entity(self, entity_id: Optional[str]) -> None:
        self.selected_entity_id = entity_id
        if entity_id is not None:
            self.selected_point_idx = None
            self.is_spawn_selected = False
            self.selected_checkpoint_idx = None

    def select_control_point(self, idx: Optional[int]) -> None:
        self.selected_point_idx = idx
        if idx is not None:
            self.selected_entity_id = None
            self.is_spawn_selected = False
            self.selected_checkpoint_idx = None

    def select_spawn(self) -> None:
        self.is_spawn_selected = True
        self.selected_point_idx = None
        self.selected_entity_id = None
        self.selected_checkpoint_idx = None

    # ------------------------------------------------------- transforms

    def _center(self) -> Tuple[float, float]:
        return (self.canvas_rect.centerx, self.canvas_rect.centery)

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        cx, cy = self._center()
        sx = int(cx + (wx + self.view_offset_x) * self.zoom)
        sy = int(cy - (wy + self.view_offset_y) * self.zoom)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        cx, cy = self._center()
        wx = (sx - cx) / self.zoom - self.view_offset_x
        wy = -(sy - cy) / self.zoom - self.view_offset_y
        return wx, wy

    # ------------------------------------------------ bounds & framing

    def compute_bounds(self, margin: float = 8.0) -> Tuple[float, float, float, float]:
        """World-space bounds of everything editable on the track."""
        xs, ys = [], []
        for cp in self.road_def.control_points:
            hw = getattr(cp, "width", 12.0) * 0.5
            xs += [cp.x - hw, cp.x + hw]
            ys += [cp.y - hw, cp.y + hw]
        sp = self.road_def.spawn_point
        xs += [sp.x - 4.0, sp.x + 4.0]
        ys += [sp.y - 4.0, sp.y + 4.0]
        for ent in self.entities:
            ext = max(getattr(ent, "length", 2.0), getattr(ent, "width", 2.0),
                      getattr(ent, "radius", 1.0) * 2.0) * 0.5
            xs += [ent.pos.x - ext, ent.pos.x + ext]
            ys += [ent.pos.y - ext, ent.pos.y + ext]
        if not xs:
            return (-50.0, -50.0, 50.0, 50.0)
        return (min(xs) - margin, min(ys) - margin,
                max(xs) + margin, max(ys) + margin)

    def frame_bounds(self, bounds: Tuple[float, float, float, float],
                     pad_frac: float = 0.12) -> None:
        """Fit the given world bounds inside the canvas with padding."""
        minx, miny, maxx, maxy = bounds
        span_x = max(maxx - minx, 1e-3)
        span_y = max(maxy - miny, 1e-3)
        vw = max(40, self.canvas_rect.w)
        vh = max(40, self.canvas_rect.h)
        self.zoom = clamp(
            min(vw * (1.0 - 2 * pad_frac) / span_x,
                vh * (1.0 - 2 * pad_frac) / span_y),
            0.05, 35.0)
        # world_to_screen uses (w + offset): center world point → offset=-c
        self.view_offset_x = -(minx + maxx) * 0.5
        self.view_offset_y = -(miny + maxy) * 0.5

    def frame_all(self) -> None:
        self.frame_bounds(self.compute_bounds())

    def frame_point(self, wx: float, wy: float, zoom: Optional[float] = None) -> None:
        if zoom is not None:
            self.zoom = clamp(zoom, 0.05, 35.0)
        self.view_offset_x = -wx
        self.view_offset_y = -wy

    def frame_selected(self) -> None:
        ent = self.get_selected_entity()
        if ent is not None:
            ext = max(getattr(ent, "length", 2.0), getattr(ent, "width", 2.0), 4.0)
            self.frame_bounds((ent.pos.x - ext, ent.pos.y - ext,
                               ent.pos.x + ext, ent.pos.y + ext), 0.25)
            return
        if self.is_spawn_selected:
            sp = self.road_def.spawn_point
            self.frame_bounds((sp.x - 20, sp.y - 20, sp.x + 20, sp.y + 20), 0.2)
            return
        if self.selected_point_idx is not None and \
                0 <= self.selected_point_idx < len(self.road_def.control_points):
            cp = self.road_def.control_points[self.selected_point_idx]
            hw = max(cp.width, 24.0)
            self.frame_bounds((cp.x - hw, cp.y - hw, cp.x + hw, cp.y + hw), 0.25)

    def _snap(self, v: float) -> float:
        if self.snap_enabled and self.snap_step > 0:
            return round(round(v / self.snap_step) * self.snap_step, 3)
        return round(v, 1)

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()

    # ---------------------------------------------------------- editing

    def delete_selected(self) -> bool:
        """Delete selected entity or control point. True if mutated."""
        if self.selected_entity_id is not None:
            ent = self.get_selected_entity()
            if ent is not None:
                self.entities.remove(ent)
                self.selected_entity_id = None
                self._changed()
                return True
            return False
        if self.selected_point_idx is not None and \
                len(self.road_def.control_points) > 3:
            self.road_def.control_points.pop(self.selected_point_idx)
            self.selected_point_idx = min(
                self.selected_point_idx, len(self.road_def.control_points) - 1)
            self._changed()
            return True
        return False

    def deselect_all(self) -> None:
        self.selected_point_idx = None
        self.selected_entity_id = None
        self.selected_checkpoint_idx = None
        self.is_spawn_selected = False

    def handle_mouse_down(self, pos: Tuple[int, int], button: int) -> bool:
        """Handles mouse clicks for selection, dragging, panning, and entity placement."""
        wx, wy = self.screen_to_world(pos[0], pos[1])

        # 1. Middle button: Pan
        if button == 2:
            self.is_panning = True
            self._pan_start = pos
            self._orig_view_offset = (self.view_offset_x, self.view_offset_y)
            return True

        # 2. Tool placement mode (Click canvas to place entity)
        if self.active_tool and button == 1:
            if self.active_tool == "spawn":
                self.road_def.spawn_point.x = self._snap(wx)
                self.road_def.spawn_point.y = self._snap(wy)
                self.select_spawn()
            elif self.active_tool in ("obstacle", "barrier", "cone", "traffic_sign", "traffic_light"):
                new_ent = create_entity(
                    entity_type=self.active_tool,
                    pos=Vec3(self._snap(wx), self._snap(wy), 0.0),
                    yaw=0.0
                )
                self.entities.append(new_ent)
                self.select_entity(new_ent.entity_id)
            self.active_tool = None
            self._changed()
            return True

        # 3. Check click on placed entities
        for ent in reversed(self.entities):
            ex, ey = ent.pos.x, ent.pos.y
            esx, esy = self.world_to_screen(ex, ey)
            dx = pos[0] - esx
            dy = pos[1] - esy
            hit_r = max(14, int(max(getattr(ent, 'length', 2.0), getattr(ent, 'width', 2.0)) * 0.5 * self.zoom))
            if dx * dx + dy * dy <= (hit_r * hit_r):
                if button == 1:
                    self.select_entity(ent.entity_id)
                    self.is_dragging_entity = True
                    self._drag_start_pos = (wx, wy)
                    return True
                elif button == 3:  # Right click: Delete entity
                    self.entities.remove(ent)
                    if self.selected_entity_id == ent.entity_id:
                        self.selected_entity_id = None
                    self._changed()
                    return True

        # 4. Check click on Spawn Point
        sp = self.road_def.spawn_point
        sp_sx, sp_sy = self.world_to_screen(sp.x, sp.y)
        if (pos[0] - sp_sx) ** 2 + (pos[1] - sp_sy) ** 2 <= 144:
            if button == 1:
                self.select_spawn()
                return True

        # 5. Check click on control points
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = self.world_to_screen(cp.x, cp.y)
            dx = pos[0] - sx
            dy = pos[1] - sy
            if dx * dx + dy * dy <= 169:  # 13px radius
                self.select_control_point(i)
                if button == 1:
                    self.is_dragging_point = True
                    return True
                elif button == 3:  # Right click: Delete control point if > 3 points
                    if len(self.road_def.control_points) > 3:
                        self.road_def.control_points.pop(i)
                        self.selected_point_idx = max(0, i - 1)
                        self._changed()
                        return True

        # 6. Draw tool: click on a segment between control points INSERTS a
        #    point there (only when the draw tool is active — a stray click
        #    in select mode must never create geometry)
        if button == 1 and self.tool == "draw" and len(self.road_def.control_points) >= 2:
            cps = self.road_def.control_points
            n = len(cps)
            num_segs = n if self.road_def.is_closed else (n - 1)
            for i in range(num_segs):
                p1 = cps[i]
                p2 = cps[(i + 1) % n]
                s1x, s1y = self.world_to_screen(p1.x, p1.y)
                s2x, s2y = self.world_to_screen(p2.x, p2.y)
                dist_to_seg = self._dist_point_to_line_segment(pos[0], pos[1], s1x, s1y, s2x, s2y)
                if dist_to_seg < 10.0:  # Within 10 pixels of line
                    insert_idx = i + 1
                    interp_width = (p1.width + p2.width) * 0.5
                    new_cp = ControlPoint(x=self._snap(wx), y=self._snap(wy), z=(p1.z + p2.z) * 0.5, width=interp_width)
                    self.road_def.control_points.insert(insert_idx, new_cp)
                    self.select_control_point(insert_idx)
                    self.is_dragging_point = True
                    self._changed()
                    return True

        # 7. Right click on empty space: Pan view
        if button == 3:
            self.is_panning = True
            self._pan_start = pos
            self._orig_view_offset = (self.view_offset_x, self.view_offset_y)
            return True

        # 8. Left click on empty space
        if button == 1:
            if self.tool == "draw":
                # Append a control point after the selected one
                insert_idx = (self.selected_point_idx + 1) if self.selected_point_idx is not None else len(self.road_def.control_points)
                new_cp = ControlPoint(x=self._snap(wx), y=self._snap(wy), z=0.0, width=12.0)
                self.road_def.control_points.insert(insert_idx, new_cp)
                self.select_control_point(insert_idx)
                self.is_dragging_point = True
                self._changed()
            else:
                # Select tool: click empty space = deselect
                self.deselect_all()
            return True

        return False

    def handle_mouse_up(self) -> None:
        was_editing = (self.is_dragging_point or self.is_dragging_entity
                       or self.is_dragging_width)
        self.is_dragging_point = False
        self.is_dragging_entity = False
        self.is_rotating_entity = False
        self.is_dragging_width = False
        self.is_panning = False
        if was_editing:
            self._changed()  # committed drag → history/dirty hook

    def handle_mouse_wheel(self, y_offset: int,
                           pos: Optional[Tuple[int, int]] = None) -> None:
        """Zoom anchored at the cursor (or canvas center)."""
        factor = 1.15 if y_offset > 0 else (1.0 / 1.15)
        ax, ay = pos if pos else (self.canvas_rect.centerx,
                                  self.canvas_rect.centery)
        wx, wy = self.screen_to_world(ax, ay)
        self.zoom = clamp(self.zoom * factor, 0.05, 60.0)
        # keep the world point under the cursor fixed:
        #   w = (s - c)/z - o  ⇒  o = (s - c)/z - w
        cx, cy = self._center()
        self.view_offset_x = (ax - cx) / self.zoom - wx
        self.view_offset_y = -(ay - cy) / self.zoom - wy

    def handle_mouse_move(self, pos: Tuple[int, int]) -> None:
        if self.is_panning:
            dx = (pos[0] - self._pan_start[0]) / self.zoom
            dy = -(pos[1] - self._pan_start[1]) / self.zoom
            self.view_offset_x = self._orig_view_offset[0] + dx
            self.view_offset_y = self._orig_view_offset[1] + dy

        elif self.is_dragging_point and self.selected_point_idx is not None:
            if 0 <= self.selected_point_idx < len(self.road_def.control_points):
                wx, wy = self.screen_to_world(pos[0], pos[1])
                self.road_def.control_points[self.selected_point_idx].x = self._snap(wx)
                self.road_def.control_points[self.selected_point_idx].y = self._snap(wy)

        elif self.is_dragging_entity and self.selected_entity_id is not None:
            ent = self.get_selected_entity()
            if ent:
                wx, wy = self.screen_to_world(pos[0], pos[1])
                ent.pos.x = self._snap(wx)
                ent.pos.y = self._snap(wy)

    def _dist_point_to_line_segment(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0.0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def draw_editor(
        self,
        surface: pygame.Surface,
        screen_rect: pygame.Rect,
        font: pygame.font.Font,
        track_spline: Any,
        checkpoints: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Renders comprehensive 2D top-down track geometry, visual gizmos, and entities."""
        self.canvas_rect = screen_rect
        sc_x, sc_y = self._center()
        surface.set_clip(screen_rect)

        # 1. Background Grid Lines
        if self.show_grid:
            grid_step = int(10.0 * self.zoom)
            if grid_step > 15:
                ox = int(sc_x + self.view_offset_x * self.zoom) % grid_step
                oy = int(sc_y - self.view_offset_y * self.zoom) % grid_step
                for x in range(screen_rect.left + ox, screen_rect.right, grid_step):
                    pygame.draw.line(surface, (30, 36, 46, 120), (x, screen_rect.top), (x, screen_rect.bottom), 1)
                for y in range(screen_rect.top + oy, screen_rect.bottom, grid_step):
                    pygame.draw.line(surface, (30, 36, 46, 120), (screen_rect.left, y), (screen_rect.right, y), 1)

        # 2. Spline Centerline with Curvature Visualization & Road Boundaries
        if track_spline and track_spline.samples:
            samples = track_spline.samples
            m = len(samples)

            pts_center = []
            pts_left = []
            pts_right = []
            curvatures = []

            for i, s in enumerate(samples):
                sx, sy = self.world_to_screen(s.pos.x, s.pos.y)
                pts_center.append((sx, sy))

                half_w = s.width * 0.5
                lx = s.pos.x + s.normal.x * half_w
                ly = s.pos.y + s.normal.y * half_w
                rx = s.pos.x - s.normal.x * half_w
                ry = s.pos.y - s.normal.y * half_w
                pts_left.append(self.world_to_screen(lx, ly))
                pts_right.append(self.world_to_screen(rx, ry))

                # Curvature estimate: rate of heading change
                next_s = samples[(i + 1) % m]
                prev_s = samples[(i - 1) % m]
                t_next = math.atan2(next_s.tangent.y, next_s.tangent.x)
                t_prev = math.atan2(prev_s.tangent.y, prev_s.tangent.x)
                d_theta = abs(normalize_angle(t_next - t_prev))
                ds = max(0.1, (next_s.s - prev_s.s) % track_spline.total_length if track_spline.is_closed else 2.0)
                curvatures.append(d_theta / ds)

            if len(pts_center) > 2:
                # Road Boundaries
                pygame.draw.lines(surface, (160, 170, 185, 210), track_spline.is_closed, pts_left, 2)
                pygame.draw.lines(surface, (160, 170, 185, 210), track_spline.is_closed, pts_right, 2)

                # Curvature-colored centerline segments
                num_draw = m if track_spline.is_closed else (m - 1)
                for i in range(num_draw):
                    k = curvatures[i]
                    p1 = pts_center[i]
                    p2 = pts_center[(i + 1) % m]
                    if not self.show_curvature or k < 0.02:
                        color = (0, 220, 255)  # Gentle / Straight
                    elif k < 0.05:
                        color = (255, 200, 40)  # Moderate corner
                    else:
                        color = (255, 60, 60)   # Sharp turn / Hairpin
                    pygame.draw.line(surface, color, p1, p2, 2)

                # Centerline Direction Arrows (Chevrons) every ~20 samples
                if self.show_tangents:
                    step_arrow = max(8, int(25.0 / max(0.5, track_spline.total_length / max(1, m))))
                    for i in range(0, m, step_arrow):
                        s = samples[i]
                        cx, cy = pts_center[i]
                        tan = s.tangent
                        arr_len = int(12 * (self.zoom / 4.0))
                        arr_len = max(8, min(20, arr_len))
                        ax = int(cx + tan.x * arr_len)
                        ay = int(cy - tan.y * arr_len)
                        pygame.draw.line(surface, (0, 255, 200), (cx, cy), (ax, ay), 2)

        # 3. Checkpoints Gates Gizmo
        if checkpoints:
            for cp in checkpoints:
                gl = cp['gate_left']
                gr = cp['gate_right']
                slx, sly = self.world_to_screen(gl.x, gl.y)
                srx, sry = self.world_to_screen(gr.x, gr.y)

                is_start = (cp['index'] == 0)
                cp_color = (0, 255, 180) if is_start else (70, 190, 130, 180)
                pygame.draw.line(surface, cp_color, (slx, sly), (srx, sry), 2 if not is_start else 3)

                c_pos = cp.get('pos')
                c_tan = cp.get('tangent')
                if c_pos and c_tan:
                    cx, cy = self.world_to_screen(c_pos.x, c_pos.y)
                    arr_len = 16
                    tan_2d = Vec2(c_tan.x, c_tan.y).normalized()
                    ax = int(cx + tan_2d.x * arr_len)
                    ay = int(cy - tan_2d.y * arr_len)
                    pygame.draw.line(surface, cp_color, (cx, cy), (ax, ay), 2)

                    # Gate badge
                    badge = font.render(f"CP {cp['index']}", True, cp_color)
                    surface.blit(badge, (srx + 6, sry - 8))

        # 4. Connecting Reference Lines Between Control Points
        cp_screen_pts = []
        for cp in self.road_def.control_points:
            cp_screen_pts.append(self.world_to_screen(cp.x, cp.y))
        if len(cp_screen_pts) > 1:
            pygame.draw.lines(surface, (70, 110, 170, 70), self.road_def.is_closed, cp_screen_pts, 1)

        # 5. Render Placed World Entities
        for ent in self.entities:
            self._draw_entity_gizmo(surface, ent, font)

        # 6. Render Control Points & Visual Handles
        for i, cp in enumerate(self.road_def.control_points):
            sx, sy = cp_screen_pts[i]
            is_sel = (i == self.selected_point_idx)

            color = (255, 80, 80) if is_sel else (80, 180, 255)
            radius = 10 if is_sel else 7

            # Outer ring
            pygame.draw.circle(surface, color, (sx, sy), radius)
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), radius, 2)

            # Selected Control Point: Width handles & Banking badge
            if is_sel and self.show_width_handles:
                # Selection indicator ring
                pygame.draw.circle(surface, (255, 220, 0), (sx, sy), radius + 5, 1)

                # Road width preview circle
                w_pixels = int(cp.width * 0.5 * self.zoom)
                pygame.draw.circle(surface, (255, 255, 255, 80), (sx, sy), w_pixels, 1)

                # Detailed metadata label
                elev_str = f"Z: {cp.z:+.1f}m" if cp.z != 0 else ""
                bank_str = f"Bank: {cp.banking:+.0f}°" if cp.banking != 0 else ""
                badge_text = f"P{i} ({cp.width:.1f}m) {elev_str} {bank_str}".strip()
                lbl = font.render(badge_text, True, (255, 240, 150))
                surface.blit(lbl, (sx + 12, sy - 14))
            else:
                lbl = font.render(f"P{i}", True, (220, 225, 235))
                surface.blit(lbl, (sx + 10, sy - 10))

        # 7. Render Spawn Point Indicator
        sp = self.road_def.spawn_point
        spx, spy = self.world_to_screen(sp.x, sp.y)
        is_sp_sel = self.is_spawn_selected

        col_sp = (0, 255, 120)
        pygame.draw.circle(surface, col_sp, (spx, spy), 9)
        if is_sp_sel:
            pygame.draw.circle(surface, (255, 220, 0), (spx, spy), 14, 2)

        # Vehicle footprint box preview (4.5m x 1.8m)
        v_hl = 2.25 * self.zoom
        v_hw = 0.9 * self.zoom
        cos_y = math.cos(sp.yaw)
        sin_y = math.sin(sp.yaw)
        corners = [
            (spx + cos_y * v_hl - sin_y * v_hw, spy - (sin_y * v_hl + cos_y * v_hw)),
            (spx - cos_y * v_hl - sin_y * v_hw, spy - (-sin_y * v_hl + cos_y * v_hw)),
            (spx - cos_y * v_hl + sin_y * v_hw, spy - (-sin_y * v_hl - cos_y * v_hw)),
            (spx + cos_y * v_hl + sin_y * v_hw, spy - (sin_y * v_hl - cos_y * v_hw)),
        ]
        pygame.draw.polygon(surface, (0, 255, 120, 90), [(int(x), int(y)) for x, y in corners], 1)

        # Forward orientation arrow
        arrow_len = int(28 * (self.zoom / 4.0))
        ax = int(spx + math.cos(sp.yaw) * arrow_len)
        ay = int(spy - math.sin(sp.yaw) * arrow_len)
        pygame.draw.line(surface, col_sp, (spx, spy), (ax, ay), 3)
        lbl_sp = font.render(f"SPAWN ({math.degrees(sp.yaw):.0f}°)", True, col_sp)
        surface.blit(lbl_sp, (spx + 14, spy - 10))

        # 8. Active Tool Banner
        if self.active_tool:
            tool_name = self.active_tool.replace('_', ' ').upper()
            hint = font.render(f">> CLICK CANVAS TO PLACE: {tool_name} (ESC to cancel) <<", True, (255, 220, 0))
            surface.blit(hint, (sc_x - hint.get_width() // 2, screen_rect.top + 16))

        surface.set_clip(None)

    def _draw_entity_gizmo(
        self,
        surface: pygame.Surface,
        ent: WorldEntity,
        font: pygame.font.Font
    ) -> None:
        """Draws individual environment entity icon and collision footprint."""
        sx, sy = self.world_to_screen(ent.pos.x, ent.pos.y)
        is_sel = (ent.entity_id == self.selected_entity_id)

        # Highlight ring if selected
        if is_sel:
            pygame.draw.circle(surface, (255, 220, 0), (sx, sy), 18, 2)

        cos_y = math.cos(ent.yaw)
        sin_y = math.sin(ent.yaw)

        # Entity type specific visuals
        if isinstance(ent, (StaticObstacle, Barrier)):
            hl = max(0.2, ent.length * 0.5) * self.zoom
            hw = max(0.2, ent.width * 0.5) * self.zoom
            corners = [
                (sx + cos_y * hl - sin_y * hw, sy - (sin_y * hl + cos_y * hw)),
                (sx - cos_y * hl - sin_y * hw, sy - (-sin_y * hl + cos_y * hw)),
                (sx - cos_y * hl + sin_y * hw, sy - (-sin_y * hl - cos_y * hw)),
                (sx + cos_y * hl + sin_y * hw, sy - (sin_y * hl - cos_y * hw)),
            ]
            fill_col = (180, 50, 50) if isinstance(ent, StaticObstacle) else (110, 130, 150)
            pygame.draw.polygon(surface, fill_col, [(int(x), int(y)) for x, y in corners])
            pygame.draw.polygon(surface, (255, 255, 255), [(int(x), int(y)) for x, y in corners], 2)
            lbl = font.render(ent.name, True, (240, 240, 240))
            surface.blit(lbl, (sx + 10, sy - 8))

        elif isinstance(ent, TrafficCone):
            r_px = max(4, int(ent.radius * self.zoom))
            pygame.draw.circle(surface, (255, 120, 0), (sx, sy), r_px)
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy), max(2, r_px // 2), 1)
            lbl = font.render("CONE", True, (255, 160, 50))
            surface.blit(lbl, (sx + 8, sy - 8))

        elif isinstance(ent, TrafficSign):
            # Diamond / Octagon icon
            r = 10
            diamond = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)]
            col = (220, 40, 40) if "stop" in ent.sign_type else (230, 190, 20)
            pygame.draw.polygon(surface, col, diamond)
            pygame.draw.polygon(surface, (255, 255, 255), diamond, 2)
            lbl = font.render(ent.sign_type.upper(), True, (255, 255, 255))
            surface.blit(lbl, (sx + 12, sy - 8))

        elif isinstance(ent, TrafficLight):
            # Traffic light box
            bw, bh = 14, 26
            pygame.draw.rect(surface, (20, 20, 25), (sx - bw//2, sy - bh//2, bw, bh), border_radius=3)
            # Active light color
            l_col = (50, 220, 80) if ent.state == "green" else ((255, 200, 30) if ent.state == "yellow" else (240, 40, 40))
            pygame.draw.circle(surface, l_col, (sx, sy), 5)
            lbl = font.render(f"TL ({ent.state.upper()})", True, l_col)
            surface.blit(lbl, (sx + 12, sy - 8))

        else:
            pygame.draw.circle(surface, (150, 150, 200), (sx, sy), 8)
            lbl = font.render(ent.name, True, (200, 200, 220))
            surface.blit(lbl, (sx + 10, sy - 8))
