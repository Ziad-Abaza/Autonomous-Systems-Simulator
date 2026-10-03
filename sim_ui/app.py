"""
Main Simulation Platform Application.
Combines 3D ModernGL rendering, Pygame SDL2 windowing, track editor,
RL simulation engine, TCP external AI server, and episode recorder.
"""

from __future__ import annotations
import sys
import os
import time
import math
from typing import Optional, Dict, Any, Tuple

if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import pygame
import numpy as np
import moderngl

from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition
from sim_env.environment import SimulationEnvironment
from sim_net.server import SimulationServer
from sim_render.renderer import SimulationRenderer3D
from sim_render.camera import CameraMode
from sim_render.offscreen import OffscreenFBO
from sim_ui.ui_overlay import UIOverlayRenderer
from sim_ui.hud import SimulationHUD
from sim_ui.editor import VisualTrackEditor
from sim_ui.inspector import EnvironmentInspector
from sim_recorder.recorder import EpisodeRecorder
from sim_recorder.replay import EpisodeReplayPlayer
from sim_project.serializer import EnvironmentProject
from sim_project.presets import (
    create_oval_circuit,
    create_serpentine_track,
    create_obstacle_challenge
)


class SimulationStudioApp:
    """
    The complete Standalone AI Environment Studio application.
    """
    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False, port: int = 8765):
        self.width = width
        self.height = height
        self.headless = headless
        self.port = port
        self.is_running = True

        # Active Mode: "mode_sim", "mode_editor", "mode_replay"
        self.active_mode = "mode_sim"

        # 1. Initialize Simulation Environment
        self.project = create_oval_circuit()
        self.env = SimulationEnvironment(road_def=self.project.road_def)
        self.obs, self.step_info = self.env.reset()

        # 2. Start TCP Simulation Server for External AI
        self.server = SimulationServer(self.env, host="127.0.0.1", port=self.port)
        self.server_started = self.server.start()
        if self.server_started:
            print(f"[Studio] External AI Server listening on port {self.port}...")

        # 3. Recorder & Replay
        self.recorder = EpisodeRecorder()
        self.replay_player = EpisodeReplayPlayer()

        # 4. Manual driving inputs
        self.manual_steer = 0.0
        self.manual_throttle = 0.0
        self.manual_brake = 0.0

        if not self.headless:
            # In interactive studio mode, allow free continuous driving without step/checkpoint timeouts
            self.env.termination_engine.config.max_episode_steps = 0
            self.env.termination_engine.config.max_seconds_without_checkpoint = 0.0
            self._init_graphics()

    def _init_graphics(self) -> None:
        pygame.init()
        pygame.display.set_caption("AI Simulation Studio - Autonomous Vehicle Platform")

        # Request OpenGL 3.3 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

        self.gl_ctx = moderngl.create_context()
        self.renderer = SimulationRenderer3D(self.gl_ctx)
        self.renderer.load_track(self.env.track)

        self.ui_renderer = UIOverlayRenderer(self.gl_ctx, self.width, self.height)
        self.hud = SimulationHUD()
        self.track_editor = VisualTrackEditor(self.env.road_def)
        self.inspector = EnvironmentInspector(self.env.road_def, self.env.vehicle.config, self.env.sensors)

        # Wire Inspector Callbacks
        def do_rebuild():
            self.env.set_road_definition(self.env.road_def)
            if not self.headless:
                self.renderer.load_track(self.env.track)
            self.obs, self.step_info = self.env.reset()
            print("[Studio] Rebuilt 3D track mesh and collision geometry.")
        self.inspector.on_rebuild_mesh = do_rebuild

        def do_save():
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "custom_environment.sim.json")
            self.project.road_def = self.env.road_def
            self.project.vehicle_config = self.env.vehicle.config
            self.project.save(save_path)
            print(f"[Studio] Project successfully saved to {save_path}")
        self.inspector.on_save_project = do_save

        def do_load():
            load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets", "custom_environment.sim.json")
            if os.path.exists(load_path):
                proj = EnvironmentProject.load(load_path)
                self.project = proj
                self.env.road_def = proj.road_def
                self.env.vehicle.config = proj.vehicle_config
                self.env.set_road_definition(proj.road_def)
                if not self.headless:
                    self.renderer.load_track(self.env.track)
                self.track_editor.road_def = self.env.road_def
                self.inspector.road_def = self.env.road_def
                self.inspector.vehicle_config = self.env.vehicle.config
                self.obs, self.step_info = self.env.reset()
                print(f"[Studio] Loaded project from {load_path}")
            else:
                print(f"[Studio] No saved project found at {load_path}")
        self.inspector.on_load_project = do_load

        def do_new():
            proj = create_oval_circuit()
            self.project = proj
            self.env.set_road_definition(proj.road_def)
            if not self.headless:
                self.renderer.load_track(self.env.track)
            self.track_editor.road_def = self.env.road_def
            self.inspector.road_def = self.env.road_def
            self.obs, self.step_info = self.env.reset()
            print("[Studio] Created new environment.")
        self.inspector.on_new_project = do_new

        def do_place_spawn():
            self.track_editor.is_placing_spawn = True
            print("[Studio] Click canvas to place spawn point.")
        self.inspector.on_start_place_spawn = do_place_spawn

        # Hook synthetic camera sensor to offscreen renderer
        cam_sensor = self.env.sensors.get_sensor("rgb_camera")
        if cam_sensor and hasattr(cam_sensor, 'set_offscreen_renderer'):
            self.offscreen_fbo = OffscreenFBO(self.gl_ctx, width=cam_sensor.width, height=cam_sensor.height)
            def offscreen_render_hook(sensor):
                return self._render_offscreen_camera(sensor)
            cam_sensor.set_offscreen_renderer(offscreen_render_hook)

        self.clock = pygame.time.Clock()

    def _render_offscreen_camera(self, camera_sensor: Any) -> Optional[np.ndarray]:
        """Renders scene from vehicle's onboard camera into offscreen FBO."""
        if not hasattr(self, 'offscreen_fbo') or self.offscreen_fbo is None:
            return None

        # Only execute offscreen render pass if camera observations are enabled
        if not self.env.observation_schema.include_camera_rgb:
            return None

        old_cam_pos = self.renderer.camera.pos
        old_cam_target = self.renderer.camera.target
        old_aspect = self.renderer.camera.aspect_ratio

        try:
            self.offscreen_fbo.bind()
            st = self.env.vehicle.state
            cos_y = math.cos(st.yaw)
            sin_y = math.sin(st.yaw)
            cam_x = st.pos.x + cos_y * camera_sensor.local_pos.x - sin_y * camera_sensor.local_pos.y
            cam_y = st.pos.y + sin_y * camera_sensor.local_pos.x + cos_y * camera_sensor.local_pos.y
            cam_z = st.pos.z + camera_sensor.local_pos.z

            self.renderer.camera.pos = Vec3(cam_x, cam_y, cam_z)
            self.renderer.camera.target = Vec3(cam_x + cos_y * 20.0, cam_y + sin_y * 20.0, cam_z + math.sin(camera_sensor.local_pitch) * 20.0)
            self.renderer.camera.aspect_ratio = float(camera_sensor.width) / float(camera_sensor.height)

            # Render 3D scene from vehicle camera
            self.renderer.render_frame(
                vehicle=self.env.vehicle,
                track=self.env.track,
                obstacles=self.env.obstacles,
                sensors=self.env.sensors,
                checkpoints=[],
                current_cp_idx=0,
                viewport_width=camera_sensor.width,
                viewport_height=camera_sensor.height,
                show_lidar_rays=False,
                show_trajectory=False,
                show_checkpoints=False
            )

            img = self.offscreen_fbo.read_rgb()
            return img
        finally:
            # ALWAYS restore main window default framebuffer and viewport!
            self.gl_ctx.screen.use()
            self.gl_ctx.viewport = (0, 0, self.width, self.height)
            self.renderer.camera.pos = old_cam_pos
            self.renderer.camera.target = old_cam_target
            self.renderer.camera.aspect_ratio = old_aspect

    def run(self) -> None:
        """Main simulation execution loop."""
        print("[Studio] AI Simulation Studio started. Press ESC to exit.")
        while self.is_running:
            dt = 1.0 / 60.0

            # 1. Process Network Messages from External AI
            external_stepped = self.server.poll_and_process()

            # 2. Process GUI Events (if visual mode)
            if not self.headless:
                self._handle_events()

                # If no external AI is controlling this tick, check manual keyboard inputs
                if not external_stepped and self.active_mode == "mode_sim":
                    self._handle_keyboard_drive()
                    act = [self.manual_steer, self.manual_throttle, self.manual_brake]
                    self.obs, reward, term, trunc, self.step_info = self.env.step(act)

                    # Record frame if recording is active
                    if self.recorder.is_recording:
                        st = self.env.vehicle.state
                        self.recorder.record_step(
                            step=self.env.current_step,
                            sim_time=self.env.clock.sim_time,
                            vehicle_pos=(st.pos.x, st.pos.y, st.pos.z),
                            vehicle_yaw=st.yaw,
                            speed=st.speed,
                            action=act,
                            reward=reward,
                            reward_breakdown=self.step_info.get('reward_breakdown', {}),
                            telemetry=self.step_info,
                            is_colliding=self.step_info.get('is_colliding', False)
                        )

                    if term or trunc:
                        self.obs, self.step_info = self.env.reset()

                # Update camera position
                st = self.env.vehicle.state
                self.renderer.camera.update(st.pos, st.yaw, dt)

                # Render Frame
                self._render()
                pygame.display.flip()
                self.clock.tick(60)
            else:
                # Headless loop
                time.sleep(0.005)

        self.cleanup()

    def _handle_keyboard_drive(self) -> None:
        """Smooth keyboard vehicle control for manual testing."""
        keys = pygame.key.get_pressed()

        # Steering: A/D or Left/Right
        steer_target = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer_target -= 0.8
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer_target += 0.8

        self.manual_steer += (steer_target - self.manual_steer) * 0.25

        # Throttle: W or Up
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.manual_throttle = min(1.0, self.manual_throttle + 0.1)
        else:
            self.manual_throttle = max(0.0, self.manual_throttle - 0.15)

        # Brake: S or Down or Space
        if keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_SPACE]:
            self.manual_brake = min(1.0, self.manual_brake + 0.2)
        else:
            self.manual_brake = max(0.0, self.manual_brake - 0.2)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.ui_renderer.resize(event.w, event.h)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                    return
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.obs, self.step_info = self.env.reset()
                elif event.key == pygame.K_c:
                    # Cycle camera modes
                    cams = [CameraMode.CHASE, CameraMode.HOOD, CameraMode.TOP_DOWN, CameraMode.ORBIT]
                    idx = (cams.index(self.renderer.camera.mode) + 1) % len(cams)
                    self.renderer.camera.mode = cams[idx]
                elif event.key == pygame.K_TAB:
                    self.hud.show_obs_inspector = not self.hud.show_obs_inspector

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos, event.button)

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.active_mode == "mode_editor":
                    self.track_editor.handle_mouse_up()

            elif event.type == pygame.MOUSEWHEEL:
                if self.active_mode == "mode_editor":
                    self.track_editor.handle_mouse_wheel(event.y)

            elif event.type == pygame.MOUSEMOTION:
                if self.active_mode == "mode_editor":
                    self.track_editor.handle_mouse_move(event.pos, (self.width * 0.5, self.height * 0.5))

    def _handle_mouse_click(self, mouse_pos: Tuple[int, int], button: int) -> None:
        # Check Top Bar Buttons
        fonts = {
            'title': self.ui_renderer.font_title,
            'bold': self.ui_renderer.font_bold,
            'small': self.ui_renderer.font_small,
            'mono': self.ui_renderer.font_mono
        }
        top_btns = self.hud.draw_top_bar(
            self.ui_renderer.ui_surface, self.width, self.active_mode,
            self.renderer.camera.mode, self.server.is_running,
            self.server.is_client_connected, self.server.total_steps_served, fonts
        )
        for rect, action_id in top_btns:
            if rect.collidepoint(mouse_pos):
                if action_id in ("mode_sim", "mode_editor", "mode_replay"):
                    self.active_mode = action_id
                    if action_id == "mode_editor":
                        self.renderer.camera.mode = CameraMode.TOP_DOWN
                    return
                elif action_id.startswith("cam_"):
                    cam_name = action_id.replace("cam_", "")
                    self.renderer.camera.mode = cam_name
                    return

        # Check Editor interactions if in editor mode
        if self.active_mode == "mode_editor":
            insp_w = 340
            insp_h = min(600, self.height - 110)
            insp_btns = self.inspector.draw(self.ui_renderer.ui_surface, 15, 55, insp_w, insp_h, fonts)
            for rect, action_id in insp_btns:
                if rect.collidepoint(mouse_pos):
                    if action_id.startswith("tab_"):
                        self.inspector.active_tab = action_id.replace("tab_", "")
                    elif action_id.startswith("prop_minus_"):
                        pid = action_id.replace("prop_minus_", "")
                        step = -1.0
                        for p in self.inspector.get_properties_for_active_tab():
                            if p.prop_id == pid:
                                step = -p.step
                                break
                        self.inspector.handle_property_change(pid, step)
                    elif action_id.startswith("prop_plus_"):
                        pid = action_id.replace("prop_plus_", "")
                        step = 1.0
                        for p in self.inspector.get_properties_for_active_tab():
                            if p.prop_id == pid:
                                step = p.step
                                break
                        self.inspector.handle_property_change(pid, step)
                    elif action_id.startswith("prop_toggle_"):
                        pid = action_id.replace("prop_toggle_", "")
                        self.inspector.handle_property_change(pid, None)
                    elif action_id.startswith("prop_enum_"):
                        pid = action_id.replace("prop_enum_", "")
                        self.inspector.handle_property_change(pid, None)
                    elif action_id.startswith("prop_act_"):
                        pid = action_id.replace("prop_act_", "")
                        self.inspector.handle_property_change(pid, None)
                    elif action_id == "action_rebuild_mesh":
                        if self.inspector.on_rebuild_mesh:
                            self.inspector.on_rebuild_mesh()
                    elif action_id == "action_save_project":
                        if self.inspector.on_save_project:
                            self.inspector.on_save_project()
                    elif action_id == "action_load_project":
                        if self.inspector.on_load_project:
                            self.inspector.on_load_project()
                    elif action_id == "action_new_project":
                        if self.inspector.on_new_project:
                            self.inspector.on_new_project()
                    return

            # Canvas click for control point selection/addition/dragging/spawn
            if mouse_pos[0] > (insp_w + 25):
                self.track_editor.handle_mouse_down(mouse_pos, button, (self.width * 0.5, self.height * 0.5))
                self.inspector.select_control_point(self.track_editor.selected_point_idx)

        # Check Bottom Bar buttons
        bot_btns = self.hud.draw_bottom_bar(
            self.ui_renderer.ui_surface, self.width, self.height,
            self.recorder.is_recording, len(self.recorder.frames),
            self.active_mode == "mode_replay", self.replay_player.current_frame_idx,
            self.replay_player.total_frames, fonts
        )
        for rect, action_id in bot_btns:
            if rect.collidepoint(mouse_pos):
                if action_id == "btn_record":
                    if self.recorder.is_recording:
                        self.recorder.stop_recording(self.step_info.get('termination_reason', 'manual_stop'))
                        # Auto-save episode to disk
                        ep_path = os.path.join(os.path.dirname(__file__), "..", "last_episode.json")
                        self.recorder.save_to_file(ep_path)
                        print(f"[Studio] Saved episode recording to {ep_path}")
                    else:
                        self.recorder.start_recording(self.env.road_def.name, self.env.clock.seed)
                        print("[Studio] Started recording episode frames...")
                    return

    def _render(self) -> None:
        """Renders 3D scene followed by 2D HUD overlay."""
        self.gl_ctx.screen.use()
        self.gl_ctx.viewport = (0, 0, self.width, self.height)

        # 1. Render 3D OpenGL viewport
        self.renderer.render_frame(
            vehicle=self.env.vehicle,
            track=self.env.track,
            obstacles=self.env.obstacles,
            sensors=self.env.sensors,
            checkpoints=self.env.track.checkpoints,
            current_cp_idx=self.env.checkpoint_tracker.current_index,
            viewport_width=self.width,
            viewport_height=self.height,
            ambient_light=self.env.scenario.ambient_light,
            show_lidar_rays=True,
            show_trajectory=True,
            show_checkpoints=True
        )

        # 2. Render 2D UI Overlay
        self.ui_renderer.clear()
        surf = self.ui_renderer.ui_surface
        fonts = {
            'title': self.ui_renderer.font_title,
            'bold': self.ui_renderer.font_bold,
            'small': self.ui_renderer.font_small,
            'mono': self.ui_renderer.font_mono
        }

        # Top Bar
        self.hud.draw_top_bar(
            surf, self.width, self.active_mode,
            self.renderer.camera.mode, self.server.is_running,
            self.server.is_client_connected, self.server.total_steps_served, fonts
        )

        if self.active_mode == "mode_sim":
            # Telemetry HUD on Left
            st = self.env.vehicle.state
            self.hud.draw_telemetry_hud(
                surf, 15, 55, self.step_info,
                st.speed, self.step_info.get('heading_error', 0.0),
                self.step_info.get('lateral_offset', 0.0),
                self.env.last_action, fonts
            )

            # Live Reward Inspector on Right
            self.hud.draw_reward_inspector(
                surf, self.width - 315, 55,
                self.step_info.get('reward_breakdown', {}),
                self.env.reward_engine.total_accumulated_reward, fonts
            )

            # Onboard Camera Sensor Picture-in-Picture
            cam_sensor = self.env.sensors.get_sensor("rgb_camera")
            if cam_sensor:
                self.hud.draw_camera_pip(surf, self.width - 315, 345, cam_sensor, fonts)

        elif self.active_mode == "mode_editor":
            # Render 2D top-down spline editor overlay
            canvas_rect = pygame.Rect(0, 42, self.width, self.height - 87)
            self.track_editor.draw_editor(
                surf, canvas_rect, fonts['small'],
                self.env.track.spline, self.env.track.checkpoints
            )

            # Unified Environment Inspector
            insp_w = 340
            insp_h = min(600, self.height - 110)
            self.inspector.draw(surf, 15, 55, insp_w, insp_h, fonts)

        # Bottom Bar
        self.hud.draw_bottom_bar(
            surf, self.width, self.height,
            self.recorder.is_recording, len(self.recorder.frames),
            self.active_mode == "mode_replay", self.replay_player.current_frame_idx,
            self.replay_player.total_frames, fonts
        )

        # Blit 2D surface over 3D context
        self.ui_renderer.render_to_screen()

    def cleanup(self) -> None:
        self.server.stop()
        if not self.headless:
            pygame.quit()
        print("[Studio] Simulator shut down cleanly.")
