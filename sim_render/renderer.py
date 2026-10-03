"""
Master 3D Simulation Renderer using ModernGL.
Handles high-performance procedural track rendering, dynamic vehicle geometry,
materials, lighting, and real-time RL debug gizmos.
"""

from __future__ import annotations
import math
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import moderngl

from sim_core.math_utils import Vec2, Vec3
from sim_core.track.mesh_generator import GeneratedTrack
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_render.camera import SimulationCamera, CameraMode
from sim_render.shaders import (
    STANDARD_VS, STANDARD_FS,
    ROAD_VS, ROAD_FS,
    COLOR_VS, COLOR_FS,
    LINE_VS, LINE_FS
)
from sim_render.mesh import GLMesh, MeshBuilder
from sim_render.offscreen import OffscreenFBO


class SimulationRenderer3D:
    """
    3D Renderer for simulation visualization, editor view, and RL telemetry debug overlays.
    """
    def __init__(self, ctx: Optional[moderngl.Context] = None):
        self.ctx = ctx or moderngl.create_context()
        self.camera = SimulationCamera()

        # OpenGL states
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)

        # Compile shaders
        self.prog_standard = self.ctx.program(vertex_shader=STANDARD_VS, fragment_shader=STANDARD_FS)
        self.prog_road = self.ctx.program(vertex_shader=ROAD_VS, fragment_shader=ROAD_FS)
        self.prog_color = self.ctx.program(vertex_shader=COLOR_VS, fragment_shader=COLOR_FS)
        self.prog_line = self.ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)

        # Meshes
        self.mesh_road: Optional[GLMesh] = None
        self.mesh_curbs: Optional[GLMesh] = None
        self.mesh_barriers: Optional[GLMesh] = None
        self.mesh_ground: Optional[GLMesh] = None
        self.mesh_car_body: Optional[GLMesh] = None
        self.mesh_car_cabin: Optional[GLMesh] = None
        self.mesh_wheel: Optional[GLMesh] = None
        self.mesh_obstacle: Optional[GLMesh] = None

        # Debug Trajectory
        self.trajectory_history: List[Vec3] = []
        self.max_trajectory_pts = 300

        # Offscreen FBO for synthetic camera sensor
        self.offscreen_fbo: Optional[OffscreenFBO] = None

        self._init_primitives()

    def _init_primitives(self) -> None:
        # Ground plane
        self.mesh_ground = MeshBuilder.create_ground_plane(self.ctx, self.prog_standard, size=1200.0)

        # Vehicle meshes
        # Lower chassis
        self.mesh_car_body = MeshBuilder.create_box_primitive(self.ctx, self.prog_standard, length=4.2, width=1.8, height=0.6)
        # Upper cabin
        self.mesh_car_cabin = MeshBuilder.create_box_primitive(self.ctx, self.prog_standard, length=2.2, width=1.4, height=0.55)
        # Wheel
        self.mesh_wheel = MeshBuilder.create_box_primitive(self.ctx, self.prog_standard, length=0.68, width=0.28, height=0.68)
        # Obstacle
        self.mesh_obstacle = MeshBuilder.create_box_primitive(self.ctx, self.prog_standard, length=2.0, width=0.8, height=1.0)

    def load_track(self, track: GeneratedTrack) -> None:
        """Uploads newly generated track geometry to GPU."""
        if self.mesh_road:
            self.mesh_road.release()
        if self.mesh_curbs:
            self.mesh_curbs.release()
        if self.mesh_barriers:
            self.mesh_barriers.release()

        # Road
        self.mesh_road = MeshBuilder.create_road_mesh(
            self.ctx, self.prog_road,
            track.road_vertices, track.road_normals, track.road_uvs, track.road_indices
        )

        # Curbs
        if len(track.curb_vertices) > 0:
            self.mesh_curbs = MeshBuilder.create_color_mesh(
                self.ctx, self.prog_color,
                track.curb_vertices, track.curb_colors, track.curb_indices
            )
        else:
            self.mesh_curbs = None

        # Barriers
        if len(track.barrier_vertices) > 0:
            norms = np.tile([0.0, 0.0, 1.0], (len(track.barrier_vertices), 1)).astype(np.float32)
            self.mesh_barriers = MeshBuilder.create_standard_mesh(
                self.ctx, self.prog_standard,
                track.barrier_vertices, norms, track.barrier_indices
            )
        else:
            self.mesh_barriers = None

    def render_frame(
        self,
        vehicle: VehicleModel,
        track: GeneratedTrack,
        obstacles: List[Any],
        sensors: Any,
        checkpoints: List[Dict[str, Any]],
        current_cp_idx: int,
        viewport_width: int,
        viewport_height: int,
        ambient_light: float = 1.0,
        show_lidar_rays: bool = True,
        show_trajectory: bool = True,
        show_checkpoints: bool = True
    ) -> None:
        """Renders complete 3D viewport."""
        self.ctx.viewport = (0, 0, viewport_width, viewport_height)
        self.ctx.clear(0.53, 0.81, 0.92, 1.0)  # Sky blue

        self.camera.set_aspect_ratio(viewport_width, viewport_height)
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        vp = proj @ view

        light_dir = (-0.4, 0.5, 0.8)

        # 1. Render Ground Plane
        if self.mesh_ground:
            model_g = np.identity(4, dtype=np.float32)
            mvp_g = vp @ model_g
            self.prog_standard['u_mvp'].write(mvp_g.T.tobytes())
            self.prog_standard['u_model'].write(model_g.T.tobytes())
            self.prog_standard['u_color'].value = (0.22, 0.55, 0.24)  # Grass green
            self.prog_standard['u_light_dir'].value = light_dir
            self.prog_standard['u_ambient'].value = ambient_light
            self.mesh_ground.render()

        # 2. Render Road Surface
        if self.mesh_road:
            model_r = np.identity(4, dtype=np.float32)
            mvp_r = vp @ model_r
            self.prog_road['u_mvp'].write(mvp_r.T.tobytes())
            self.prog_road['u_light_dir'].value = light_dir
            self.prog_road['u_ambient'].value = ambient_light
            self.mesh_road.render()

        # 3. Render Curbs
        if self.mesh_curbs:
            model_c = np.identity(4, dtype=np.float32)
            mvp_c = vp @ model_c
            self.prog_color['u_mvp'].write(mvp_c.T.tobytes())
            self.mesh_curbs.render()

        # 4. Render Guardrail Barriers
        if self.mesh_barriers:
            model_b = np.identity(4, dtype=np.float32)
            mvp_b = vp @ model_b
            self.prog_standard['u_mvp'].write(mvp_b.T.tobytes())
            self.prog_standard['u_model'].write(model_b.T.tobytes())
            self.prog_standard['u_color'].value = (0.75, 0.78, 0.82)  # Metallic guardrail
            self.prog_standard['u_light_dir'].value = light_dir
            self.prog_standard['u_ambient'].value = ambient_light
            self.mesh_barriers.render()

        # 5. Render Obstacles
        if self.mesh_obstacle and obstacles:
            for obs in obstacles:
                m_obs = self._create_transform_matrix(obs.pos.x, obs.pos.y, obs.pos.z + obs.height * 0.5, obs.yaw)
                mvp_obs = vp @ m_obs
                self.prog_standard['u_mvp'].write(mvp_obs.T.tobytes())
                self.prog_standard['u_model'].write(m_obs.T.tobytes())
                if getattr(obs, 'obstacle_type', '') == 'cone':
                    self.prog_standard['u_color'].value = (1.0, 0.45, 0.0)  # Orange cone
                else:
                    self.prog_standard['u_color'].value = (0.85, 0.25, 0.25)  # Red barricade
                self.prog_standard['u_light_dir'].value = light_dir
                self.prog_standard['u_ambient'].value = ambient_light
                self.mesh_obstacle.render()

        # 6. Render Vehicle Chassis and Wheels
        self._render_vehicle(vehicle, vp, light_dir, ambient_light)

        # 7. Debug Visualizations
        # Trajectory trace
        st = vehicle.state
        self.trajectory_history.append(Vec3(st.pos.x, st.pos.y, st.pos.z + 0.1))
        if len(self.trajectory_history) > self.max_trajectory_pts:
            self.trajectory_history.pop(0)

        if show_trajectory and len(self.trajectory_history) >= 2:
            self._render_trajectory(vp)

        # Checkpoints
        if show_checkpoints and checkpoints:
            self._render_checkpoints(checkpoints, current_cp_idx, vp)

        # LiDAR Rays
        if show_lidar_rays:
            lidar_sensor = sensors.get_sensor("lidar_rays")
            if lidar_sensor and hasattr(lidar_sensor, 'ray_visuals') and lidar_sensor.ray_visuals:
                self._render_lidar_rays(lidar_sensor.ray_visuals, st.pos.z + 0.35, vp)

    def _render_vehicle(self, vehicle: VehicleModel, vp: np.ndarray, light_dir: tuple, ambient: float) -> None:
        st = vehicle.state

        # Vehicle body color: changes to bright orange-red on collision!
        car_color = (0.9, 0.15, 0.15) if st.is_colliding else (0.1, 0.55, 0.95)  # Bright blue or collision red

        # 1. Lower chassis
        m_chassis = self._create_transform_matrix(st.pos.x, st.pos.y, st.pos.z + 0.35, st.yaw)
        mvp_chassis = vp @ m_chassis
        self.prog_standard['u_mvp'].write(mvp_chassis.T.tobytes())
        self.prog_standard['u_model'].write(m_chassis.T.tobytes())
        self.prog_standard['u_color'].value = car_color
        self.prog_standard['u_light_dir'].value = light_dir
        self.prog_standard['u_ambient'].value = ambient
        if self.mesh_car_body:
            self.mesh_car_body.render()

        # 2. Upper cabin
        cos_y = math.cos(st.yaw)
        sin_y = math.sin(st.yaw)
        cabin_x = st.pos.x - cos_y * 0.4
        cabin_y = st.pos.y - sin_y * 0.4
        m_cabin = self._create_transform_matrix(cabin_x, cabin_y, st.pos.z + 0.85, st.yaw)
        mvp_cabin = vp @ m_cabin
        self.prog_standard['u_mvp'].write(mvp_cabin.T.tobytes())
        self.prog_standard['u_model'].write(m_cabin.T.tobytes())
        self.prog_standard['u_color'].value = (0.2, 0.25, 0.32)  # Tinted glass
        if self.mesh_car_cabin:
            self.mesh_car_cabin.render()

        # 3. Four Wheels
        if self.mesh_wheel:
            wheel_transforms = vehicle.get_wheel_transforms()
            for (w_pos, steer_angle) in wheel_transforms:
                m_wheel = self._create_transform_matrix(w_pos.x, w_pos.y, w_pos.z + 0.25, st.yaw + steer_angle)
                mvp_w = vp @ m_wheel
                self.prog_standard['u_mvp'].write(mvp_w.T.tobytes())
                self.prog_standard['u_model'].write(m_wheel.T.tobytes())
                self.prog_standard['u_color'].value = (0.12, 0.12, 0.12)  # Dark rubber
                self.mesh_wheel.render()

    def _render_lidar_rays(self, ray_visuals: List[Dict[str, Any]], z_height: float, vp: np.ndarray) -> None:
        """Renders LiDAR range beams with color coding."""
        lines_pts = []
        for r in ray_visuals:
            o = r['origin']
            e = r['end']
            lines_pts.append([o.x, o.y, z_height])
            lines_pts.append([e.x, e.y, z_height])

        if not lines_pts:
            return

        line_arr = np.array(lines_pts, dtype=np.float32)
        vbo = self.ctx.buffer(line_arr.tobytes())
        vao = self.ctx.vertex_array(self.prog_line, [(vbo, '3f', 'in_position')])

        self.prog_line['u_mvp'].write(vp.T.tobytes())
        self.prog_line['u_color'].value = (0.1, 1.0, 0.2, 0.8)  # Neon green LiDAR
        vao.render(mode=moderngl.LINES)
        vao.release()
        vbo.release()

    def _render_trajectory(self, vp: np.ndarray) -> None:
        pts = [[p.x, p.y, p.z] for p in self.trajectory_history]
        line_arr = np.array(pts, dtype=np.float32)
        vbo = self.ctx.buffer(line_arr.tobytes())
        vao = self.ctx.vertex_array(self.prog_line, [(vbo, '3f', 'in_position')])

        self.prog_line['u_mvp'].write(vp.T.tobytes())
        self.prog_line['u_color'].value = (1.0, 0.85, 0.0, 0.9)  # Golden trajectory
        vao.render(mode=moderngl.LINE_STRIP)
        vao.release()
        vbo.release()

    def _render_checkpoints(self, checkpoints: List[Dict[str, Any]], current_idx: int, vp: np.ndarray) -> None:
        gate_lines = []
        for cp in checkpoints:
            gl = cp['gate_left']
            gr = cp['gate_right']
            z = cp['pos'].z
            # Base line
            gate_lines.append([gl.x, gl.y, z + 0.1])
            gate_lines.append([gr.x, gr.y, z + 0.1])
            # Arch top
            gate_lines.append([gl.x, gl.y, z + 3.0])
            gate_lines.append([gr.x, gr.y, z + 3.0])
            # Left post
            gate_lines.append([gl.x, gl.y, z + 0.1])
            gate_lines.append([gl.x, gl.y, z + 3.0])
            # Right post
            gate_lines.append([gr.x, gr.y, z + 0.1])
            gate_lines.append([gr.x, gr.y, z + 3.0])

        if not gate_lines:
            return

        arr = np.array(gate_lines, dtype=np.float32)
        vbo = self.ctx.buffer(arr.tobytes())
        vao = self.ctx.vertex_array(self.prog_line, [(vbo, '3f', 'in_position')])

        self.prog_line['u_mvp'].write(vp.T.tobytes())
        self.prog_line['u_color'].value = (0.2, 0.8, 1.0, 0.6)  # Cyan checkpoints
        vao.render(mode=moderngl.LINES)
        vao.release()
        vbo.release()

    def _create_transform_matrix(self, x: float, y: float, z: float, yaw: float) -> np.ndarray:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        m = np.identity(4, dtype=np.float32)
        m[0, 0] = cos_y
        m[0, 1] = -sin_y
        m[1, 0] = sin_y
        m[1, 1] = cos_y
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m
