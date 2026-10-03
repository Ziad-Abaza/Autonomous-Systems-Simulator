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
from sim_core.world.entity import WorldEntity
from sim_env.environment import SimulationEnvironment
from sim_net.server import SimulationServer
from sim_render.renderer import SimulationRenderer3D
from sim_render.camera import CameraMode, SimulationCamera
from sim_render.offscreen import OffscreenFBO
from sim_ui.ui_overlay import UIOverlayRenderer
from sim_ui.hud import SimulationHUD
from sim_ui.editor import VisualTrackEditor
from sim_ui.inspector import EnvironmentInspector
from sim_recorder.recorder import EpisodeRecorder
from sim_recorder.replay import EpisodeReplayPlayer
from sim_project.serializer import EnvironmentProject
from sim_env.templates import EnvironmentTemplateManager
from sim_env.export import TrainingExporter
from sim_experiment.manifest import ExperimentManifest, TrainingConfig, EvaluationConfig
from sim_experiment.manager import ExperimentManager
from sim_experiment.orchestrator import LocalTrainingOrchestrator
from sim_experiment.run import RunManager, RunStatus
from sim_experiment.metrics import MetricsReader
from sim_experiment.reproduce import check_reproducibility
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
        self.env = SimulationEnvironment(
            road_def=self.project.road_def,
            agent=self.project.agent,
            scenario_def=self.project.scenario_def
        )
        self.obs, self.step_info = self.env.reset()

        # 2. Start TCP Simulation Server for External AI
        self.server = SimulationServer(self.env, host="127.0.0.1", port=self.port)
        self.server_started = self.server.start()
        if self.server_started:
            print(f"[Studio] External AI Server listening on port {self.port}...")

        # 3. Recorder & Replay
        self.recorder = EpisodeRecorder()
        self.replay_player = EpisodeReplayPlayer()

        # 3b. Training & Experiment services (UI calls services; never touches internals)
        exp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
        self.exp_mgr = ExperimentManager(root_dir=exp_root)
        self.orch = LocalTrainingOrchestrator(experiments_root=exp_root)
        self.run_mgr = RunManager()
        self.train_state = {"selected_experiment": None, "selected_run_id": None, "last_poll": 0.0}

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

        # Isolated Cameras
        self.agent_camera = SimulationCamera()
        self.debug_camera = SimulationCamera()

        self.ui_renderer = UIOverlayRenderer(self.gl_ctx, self.width, self.height)
        self.hud = SimulationHUD()
        self.track_editor = VisualTrackEditor(self.env.road_def, self.env.entities)
        self.inspector = EnvironmentInspector(
            self.env.road_def,
            self.env.vehicle.config,
            self.env.sensors,
            self.env.reward_engine.config,
            self.env.observation_schema,
            self.env.entities,
            agent=self.project.agent,
            scenario_def=self.project.scenario_def
        )

        # Wire Inspector Callbacks
        def do_rebuild():
            self.env.set_road_definition(self.env.road_def)
            if not self.headless:
                self.renderer.load_track(self.env.track)
            self.obs, self.step_info = self.env.reset()
            self.inspector.run_validation()
            print("[Studio] Rebuilt 3D track mesh, broadphase, and collision geometry.")
        self.inspector.on_rebuild_mesh = do_rebuild

        def do_agent_modified():
            self.env.set_agent(self.inspector.agent)
            self.project.agent = self.inspector.agent
            print("[Studio] Recompiled agent action/obs/reward/term runtime pipelines.")
        self.inspector.on_agent_modified = do_agent_modified

        def do_save():
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "custom_environment.sim.json")
            self.project.road_def = self.env.road_def
            self.project.vehicle_config = self.env.vehicle.config
            self.project.entities = self.env.entities
            self.project.agent = self.inspector.agent
            self.project.scenario_def = self.inspector.scenario_def
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
                self.env.entities = proj.entities
                self.env.obstacles = [e for e in proj.entities if getattr(e, 'is_collidable', False) or getattr(e, 'entity_type', '') in ('obstacle', 'barrier', 'cone')]
                self.env.set_agent(proj.agent)
                self.env.scenario_def = proj.scenario_def
                self.env.set_road_definition(proj.road_def)
                if not self.headless:
                    self.renderer.load_track(self.env.track)
                self.track_editor.road_def = self.env.road_def
                self.track_editor.entities = self.env.entities
                self.inspector.road_def = self.env.road_def
                self.inspector.vehicle_config = self.env.vehicle.config
                self.inspector.entities = self.env.entities
                self.inspector.agent = proj.agent
                self.inspector.scenario_def = proj.scenario_def
                self.inspector.run_validation()
                self.obs, self.step_info = self.env.reset()
                print(f"[Studio] Loaded project from {load_path}")
            else:
                print(f"[Studio] No saved project found at {load_path}")
        self.inspector.on_load_project = do_load

        def do_new():
            proj = create_oval_circuit()
            self.project = proj
            self.env.clear_entities()
            self.env.set_agent(proj.agent)
            self.env.scenario_def = proj.scenario_def
            self.env.set_road_definition(proj.road_def)
            if not self.headless:
                self.renderer.load_track(self.env.track)
            self.track_editor.road_def = self.env.road_def
            self.track_editor.entities = self.env.entities
            self.inspector.road_def = self.env.road_def
            self.inspector.entities = self.env.entities
            self.inspector.agent = proj.agent
            self.inspector.scenario_def = proj.scenario_def
            self.inspector.run_validation()
            self.obs, self.step_info = self.env.reset()
            print("[Studio] Created new environment.")
        self.inspector.on_new_project = do_new

        def do_export_training():
            export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "exported_training")
            files = TrainingExporter.export_training_bundle(
                output_dir=export_dir,
                env_project=self.project,
                scenario=self.project.scenario_def,
                experiment=self.project.experiment_config
            )
            print(f"[Studio] Training bundle exported to {export_dir}: {list(files.keys())}")
        self.inspector.on_export_training = do_export_training

        def do_load_template(template_id: str):
            proj = EnvironmentTemplateManager.create_project_from_template(template_id)
            self.project = proj
            self.env.clear_entities()
            for ent in proj.entities:
                self.env.add_entity(ent)
            self.env.set_agent(proj.agent)
            self.env.scenario_def = proj.scenario_def
            self.env.set_road_definition(proj.road_def)
            if not self.headless:
                self.renderer.load_track(self.env.track)
            self.track_editor.road_def = self.env.road_def
            self.track_editor.entities = self.env.entities
            self.inspector.road_def = self.env.road_def
            self.inspector.vehicle_config = self.env.vehicle.config
            self.inspector.entities = self.env.entities
            self.inspector.agent = proj.agent
            self.inspector.scenario_def = proj.scenario_def
            self.inspector.run_validation()
            self.obs, self.step_info = self.env.reset()
            print(f"[Studio] Loaded template: {template_id}")
        self.inspector.on_load_template = do_load_template

        def do_place_tool(tool_type: str):
            self.track_editor.active_tool = tool_type
            print(f"[Studio] Tool activated: {tool_type}. Click canvas to place.")
        self.inspector.on_start_place_tool = do_place_tool

        # Training & Experiments panel
        self.inspector.train_provider = self._train_provider
        self.inspector.on_train_action = self._train_action

        # Hook synthetic camera sensor to offscreen renderer
        cam_sensor = self.env.sensors.get_sensor("rgb_camera")
        if cam_sensor and hasattr(cam_sensor, 'set_offscreen_renderer'):
            self.offscreen_fbo = OffscreenFBO(self.gl_ctx, width=cam_sensor.width, height=cam_sensor.height)
            def offscreen_render_hook(sensor):
                return self._render_offscreen_camera(sensor)
            cam_sensor.set_offscreen_renderer(offscreen_render_hook)

        self.clock = pygame.time.Clock()

    def _render_offscreen_camera(self, camera_sensor: Any) -> Optional[np.ndarray]:
        """Renders scene from vehicle's onboard camera into offscreen FBO using isolated agent camera."""
        if not hasattr(self, 'offscreen_fbo') or self.offscreen_fbo is None:
            return None

        # Only execute offscreen render pass if camera observations are enabled
        if not self.env.observation_schema.include_camera_rgb:
            return None

        try:
            self.offscreen_fbo.bind()
            st = self.env.vehicle.state
            cos_y = math.cos(st.yaw)
            sin_y = math.sin(st.yaw)
            cam_x = st.pos.x + cos_y * camera_sensor.local_pos.x - sin_y * camera_sensor.local_pos.y
            cam_y = st.pos.y + sin_y * camera_sensor.local_pos.x + cos_y * camera_sensor.local_pos.y
            cam_z = st.pos.z + camera_sensor.local_pos.z

            self.agent_camera.pos = Vec3(cam_x, cam_y, cam_z)
            self.agent_camera.target = Vec3(cam_x + cos_y * 20.0, cam_y + sin_y * 20.0, cam_z + math.sin(camera_sensor.local_pitch) * 20.0)
            self.agent_camera.fov_degrees = camera_sensor.fov_degrees

            # Render 3D scene from vehicle camera with ZERO debug overlays
            self.renderer.render_frame(
                vehicle=self.env.vehicle,
                track=self.env.track,
                obstacles=self.env.obstacles + [e for e in self.env.entities if e not in self.env.obstacles],
                sensors=self.env.sensors,
                checkpoints=[],
                current_cp_idx=0,
                viewport_width=camera_sensor.width,
                viewport_height=camera_sensor.height,
                ambient_light=self.env.scenario.ambient_light,
                show_lidar_rays=False,
                show_trajectory=False,
                show_checkpoints=False,
                camera=self.agent_camera
            )

            img = self.offscreen_fbo.read_rgb()
            return img
        finally:
            # ALWAYS restore main window default framebuffer and viewport!
            self.gl_ctx.screen.use()
            self.gl_ctx.viewport = (0, 0, self.width, self.height)

    # ------------------------------------------------- training & experiments

    def _train_provider(self) -> Dict[str, Any]:
        """Supplies the inspector TRAIN tab with service-layer state (throttled)."""
        exps = self.exp_mgr.list_experiments()
        sel_exp = self.train_state["selected_experiment"]
        if sel_exp is None and exps:
            sel_exp = exps[0]["experiment_id"]
            self.train_state["selected_experiment"] = sel_exp

        sel_run = None
        if sel_exp:
            try:
                exp_dir = self.exp_mgr.experiment_dir(sel_exp)
                runs = self.run_mgr.list_runs(exp_dir)
                run_id = self.train_state["selected_run_id"] or (runs[-1]["run_id"] if runs else None)
                if run_id:
                    run = self.run_mgr.load_run(exp_dir, run_id)
                    sel_run = run.to_dict()
                    mpath = os.path.join(exp_dir, "runs", run_id, "metrics.jsonl")
                    eps = MetricsReader(mpath).by_scope("episode")
                    sel_run["reward_series"] = [
                        r["metrics"].get("reward") for r in eps[-60:]
                        if isinstance(r.get("metrics", {}).get("reward"), (int, float))
                    ]
            except Exception:
                sel_run = None

        return {
            "experiments": exps,
            "selected_experiment": sel_exp,
            "selected_run": sel_run,
        }

    def _train_action(self, prop_id: str) -> None:
        """Dispatches TRAIN tab actions to the application services."""
        ts = self.train_state
        try:
            if prop_id == "trn_create":
                manifest = ExperimentManifest.from_project(
                    project=self.project,
                    scenario=self.project.scenario_def,
                    training=TrainingConfig(algorithm="ppo", total_timesteps=20000,
                                            checkpoint_frequency=5000),
                    evaluation=EvaluationConfig(eval_seeds=[0, 1], num_episodes=3),
                    name=f"{self.project.name} experiment",
                    random_seed=42,
                )
                exp_dir = self.exp_mgr.create(manifest)
                ts["selected_experiment"] = manifest.experiment_id
                print(f"[Studio] Experiment created: {manifest.experiment_id}")

            elif prop_id.startswith("trn_sel_"):
                idx = int(prop_id.replace("trn_sel_", ""))
                exps = self.exp_mgr.list_experiments()
                if 0 <= idx < len(exps):
                    ts["selected_experiment"] = exps[idx]["experiment_id"]
                    ts["selected_run_id"] = None

            elif prop_id == "trn_launch" and ts["selected_experiment"]:
                manifest = self.exp_mgr.load(ts["selected_experiment"])
                exp_dir = self.exp_mgr.experiment_dir(manifest.experiment_id)
                run_id = self.orch.launch(manifest, exp_dir, trainer="ppo")
                self.exp_mgr.mark_launched(manifest.experiment_id)
                ts["selected_run_id"] = run_id
                print(f"[Studio] Launched PPO run {run_id}")

            elif prop_id == "trn_cancel" and ts["selected_experiment"] and ts["selected_run_id"]:
                exp_dir = self.exp_mgr.experiment_dir(ts["selected_experiment"])
                self.orch.cancel(exp_dir, ts["selected_run_id"])
                print(f"[Studio] Cancelled run {ts['selected_run_id']}")

            elif prop_id == "trn_resume" and ts["selected_experiment"] and ts["selected_run_id"]:
                from sim_experiment.artifacts import ArtifactRegistry
                exp_dir = self.exp_mgr.experiment_dir(ts["selected_experiment"])
                rd = self.run_mgr.run_dir(exp_dir, ts["selected_run_id"])
                ckpt = ArtifactRegistry(rd).latest_of_kind("checkpoint")
                if ckpt:
                    manifest = self.exp_mgr.load(ts["selected_experiment"])
                    new_id = self.orch.launch(
                        manifest, exp_dir, trainer="ppo",
                        resume_from={"parent_run_id": ts["selected_run_id"],
                                     "checkpoint": os.path.join(rd, ckpt["path"])},
                    )
                    ts["selected_run_id"] = new_id
                    print(f"[Studio] Resumed as run {new_id}")

            elif prop_id == "trn_eval" and ts["selected_experiment"] and ts["selected_run_id"]:
                from sim_experiment.artifacts import ArtifactRegistry
                from sim_experiment.evaluation import evaluate_policy, make_policy_from_checkpoint
                from sim_experiment.headless import build_env_from_dicts
                exp_dir = self.exp_mgr.experiment_dir(ts["selected_experiment"])
                rd = self.run_mgr.run_dir(exp_dir, ts["selected_run_id"])
                ckpt = ArtifactRegistry(rd).latest_of_kind("checkpoint")
                if ckpt:
                    manifest = self.exp_mgr.load(ts["selected_experiment"])
                    env = build_env_from_dicts(manifest.environment,
                                               manifest.scenario_configuration,
                                               seed=manifest.random_seed)
                    policy = make_policy_from_checkpoint(os.path.join(rd, ckpt["path"]), "ppo")
                    result = evaluate_policy(
                        env, policy,
                        seeds=manifest.evaluation.eval_seeds,
                        num_episodes=manifest.evaluation.num_episodes,
                        result_kwargs={"checkpoint_path": ckpt["path"], "algorithm": "ppo",
                                       "env_fingerprint": manifest.environment_fingerprint,
                                       "scenario_id": manifest.scenario_id},
                    )
                    eval_path = os.path.join(rd, "evaluation", f"{result.eval_id}.json")
                    result.save(eval_path)
                    ArtifactRegistry(rd).register("evaluation",
                                                  os.path.relpath(eval_path, rd))
                    self.run_mgr.add_artifact_ref(exp_dir, ts["selected_run_id"],
                                                  "evaluation", os.path.relpath(eval_path, rd))
                    print(f"[Studio] Evaluation: mean_reward={result.aggregate['mean_reward']}")

            elif prop_id == "trn_repro" and ts["selected_experiment"]:
                report = check_reproducibility(
                    self.exp_mgr.experiment_dir(ts["selected_experiment"]))
                status = "REPRODUCIBLE" if report["reproducible"] else "NOT REPRODUCIBLE"
                print(f"[Studio] Reproducibility: {status} "
                      f"({len(report['checks'])} checks, failures: {report['fatal_failures']})")

            elif prop_id == "trn_export" and ts["selected_experiment"]:
                dest = os.path.join(self.exp_mgr.root_dir, "exported",
                                    ts["selected_experiment"])
                self.exp_mgr.export(ts["selected_experiment"], dest)
                print(f"[Studio] Exported experiment to {dest}")

        except Exception as e:
            print(f"[Studio] Training action '{prop_id}' failed: {e}")

    def _poll_training_runs(self) -> None:
        """Throttled run monitoring: refreshes status/progress ~1 Hz."""
        now = time.time()
        if now - self.train_state["last_poll"] < 1.0:
            return
        self.train_state["last_poll"] = now
        sel_exp = self.train_state["selected_experiment"]
        if not sel_exp:
            return
        try:
            exp_dir = self.exp_mgr.experiment_dir(sel_exp)
            for summary in self.run_mgr.list_runs(exp_dir):
                if summary["status"] in ("QUEUED", "STARTING", "RUNNING", "PAUSED"):
                    self.orch.poll(exp_dir, summary["run_id"])
        except Exception:
            pass

    def run(self) -> None:
        """Main simulation execution loop."""
        print("[Studio] AI Simulation Studio started. Press ESC to exit.")
        while self.is_running:
            dt = 1.0 / 60.0

            # 0. Throttled training-run monitoring
            self._poll_training_runs()

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
                    if self.active_mode == "mode_editor" and self.track_editor.active_tool:
                        self.track_editor.active_tool = None
                    else:
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
            insp_w = 350
            insp_h = min(620, self.height - 100)
            insp_btns = self.inspector.draw(self.ui_renderer.ui_surface, 15, 50, insp_w, insp_h, fonts)
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
                        if action_id == "prop_act_sc_add_obstacle":
                            self.track_editor.active_tool = "obstacle"
                        elif action_id == "prop_act_sc_add_barrier":
                            self.track_editor.active_tool = "barrier"
                        elif action_id == "prop_act_sc_add_cone":
                            self.track_editor.active_tool = "cone"
                        elif action_id == "prop_act_sc_add_sign":
                            self.track_editor.active_tool = "traffic_sign"
                        elif action_id == "prop_act_sc_add_light":
                            self.track_editor.active_tool = "traffic_light"
                        elif action_id.startswith("prop_act_sc_ent_sel_"):
                            eid = action_id.replace("prop_act_sc_ent_sel_", "")
                            self.track_editor.select_entity(eid)
                            self.inspector.select_entity(eid)
                        else:
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
                handled = self.track_editor.handle_mouse_down(mouse_pos, button, (self.width * 0.5, self.height * 0.5))
                if self.track_editor.selected_entity_id:
                    self.inspector.select_entity(self.track_editor.selected_entity_id)
                elif self.track_editor.selected_point_idx is not None:
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
                        ep_path = os.path.join(os.path.dirname(__file__), "..", "last_episode.json")
                        self.recorder.save_to_file(ep_path)
                        print(f"[Studio] Saved episode recording to {ep_path}")
                    else:
                        obs_schema = self.env.agent.observation_space.export_schema() if self.env.agent else self.env.observation_schema.to_dict()
                        act_schema = self.env.agent.action_space.export_schema() if self.env.agent else self.env.action_config.to_dict()
                        rew_cfg = self.env.agent.reward_function.to_dict() if self.env.agent else self.env.reward_engine.config.to_dict()
                        fp = self.project.compute_fingerprint() if hasattr(self.project, 'compute_fingerprint') else ""
                        self.recorder.start_recording(
                            track_name=self.env.road_def.name,
                            seed=self.env.clock.seed,
                            env_config=self.project.to_dict() if hasattr(self.project, 'to_dict') else {},
                            env_version=getattr(self.project, 'environment_version', '1.0.0'),
                            scenario_name=self.env.scenario_def.name if self.env.scenario_def else "Default",
                            observation_schema=obs_schema,
                            action_schema=act_schema,
                            reward_config=rew_cfg,
                            fingerprint=fp,
                            scenario_config=self.env.scenario_def.to_dict() if self.env.scenario_def else {}
                        )
                        print("[Studio] Started recording episode frames with complete Phase 3 metadata...")
                    return

    def _render(self) -> None:
        """Renders 3D scene followed by 2D HUD overlay."""
        self.gl_ctx.screen.use()
        self.gl_ctx.viewport = (0, 0, self.width, self.height)

        # 1. Render 3D OpenGL viewport
        all_scene_obs = self.env.obstacles + [e for e in self.env.entities if e not in self.env.obstacles]
        self.renderer.render_frame(
            vehicle=self.env.vehicle,
            track=self.env.track,
            obstacles=all_scene_obs,
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
            insp_w = 350
            insp_h = min(620, self.height - 100)
            self.inspector.draw(surf, 15, 50, insp_w, insp_h, fonts)

        # Observation inspector modal overlay (press TAB to toggle)
        if self.hud.show_obs_inspector:
            self.hud.draw_observation_inspector(
                surf,
                (self.width - 620) // 2,
                (self.height - 360) // 2,
                self.obs,
                self.env.observation_schema,
                self.step_info,
                fonts
            )

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
