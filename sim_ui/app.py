"""
Main Simulation Platform Application.
Combines 3D ModernGL rendering, Pygame SDL2 windowing, track editor,
RL simulation engine, TCP external AI server, and episode recorder.
"""

from __future__ import annotations
import sys
import os
import time
import json
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
from sim_project.settings import StudioSettings
from sim_project.library import TrackLibrary
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

        # Studio settings + asset libraries (Phase 7).
        # Settings live inside the organized data dir (<repo>/data/),
        # not loose at the repo root; a legacy root file is migrated.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        settings_path = os.path.join(repo_root, "data", "studio_settings.json")
        legacy_settings = os.path.join(repo_root, "studio_settings.json")
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        if (os.path.exists(legacy_settings)
                and not os.path.exists(settings_path)):
            try:
                os.replace(legacy_settings, settings_path)
            except OSError:
                pass
        self.settings = StudioSettings(settings_path)
        import sim_ui.theme as _theme
        _theme.set_theme(self.settings.theme)
        self.library = TrackLibrary(os.path.join(repo_root, "tracks"))
        self.presets_lib = TrackLibrary(os.path.join(repo_root, "presets"),
                                        readonly=True)
        self.record_dir = self.settings.recordings_dir()

        # Screen navigation (Phase 7 studio shell)
        self.studio_screen = "home"          # "home" | "workspace"
        self.ws_tab = "SIMULATE"      # "EDIT" | "SIMULATE" | "REPLAY" | "DATA"
        self.active_dialog: Optional[str] = None
        self.dialog_payload: Any = None
        self.dirty = False
        self.dataset_detail: Optional[str] = None
        self.recording_summary: Optional[Dict[str, Any]] = None
        self.replay_path: Optional[str] = None
        self.replay_accum = 0.0
        self._last_recorded_step = -1

        # Legacy mode state kept for internals (render branches)
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

        # Phase-7 UI layer: hit-registry context + studio screens
        from sim_ui.widgets import UIContext, Fonts
        self.ui_fonts = Fonts(scale=self.settings.ui_scale)
        self.ui_ctx = UIContext(self.ui_renderer.ui_surface, self.ui_fonts)
        from sim_ui.screens.home_screen import HomeScreen
        from sim_ui.screens.workspace_screen import WorkspaceScreen
        self.home_screen = HomeScreen(self)
        self.workspace = WorkspaceScreen(self)

        self.track_editor = VisualTrackEditor(self.env.road_def, self.env.entities)
        self.track_editor.on_change = self._editor_changed

        from sim_ui.edit_history import EditHistory
        self.edit_history = EditHistory()
        self.edit_history.clear(self._project_snapshot())
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

        self.inspector.on_save_project = self.save_project
        self.inspector.on_load_project = self.open_file_dialog
        self.inspector.on_new_project = self.request_new_track

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
            proj.file_path = getattr(self.project, "file_path", None)
            self._apply_project(proj, proj.file_path)
            self.mark_dirty()
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
        sel_algo = None
        if sel_exp:
            try:
                manifest = self.exp_mgr.load(sel_exp)
                sel_algo = manifest.training.algorithm
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
                    cpath = os.path.join(exp_dir, "runs", run_id, "curriculum_state.json")
                    if os.path.exists(cpath):
                        with open(cpath, "r", encoding="utf-8") as f:
                            sel_run["curriculum"] = json.load(f)
            except Exception:
                sel_run = None

        batch_status = None
        batch_rec = self.train_state.get("batch")
        workers = []
        if batch_rec:
            try:
                batch_status = batch_rec["scheduler"].status(batch_rec["batch_id"])
                from sim_ui.train_providers import worker_rows
                workers = worker_rows(scheduler=batch_rec["scheduler"])
            except Exception:
                batch_status = None
        if not workers:
            try:
                from sim_experiment.worker_registry import WorkerRegistry
                from sim_ui.train_providers import worker_rows
                workers = worker_rows(
                    registry=WorkerRegistry(self.exp_mgr.root_dir))
            except Exception:
                workers = []

        dataset_preview = None
        ds_dir = self.train_state.get("dataset_dir")
        if ds_dir:
            try:
                from sim_ui.train_providers import dataset_preview as _dp
                dataset_preview = _dp(ds_dir)
            except Exception:
                dataset_preview = None

        try:
            from sim_ui.train_providers import comparison_to_multichart
            comparison_chart = comparison_to_multichart(
                self.train_state.get("comparison"))
        except Exception:
            comparison_chart = []

        return {
            "experiments": exps,
            "selected_experiment": sel_exp,
            "selected_algorithm": sel_algo,
            "selected_run": sel_run,
            "batch_status": batch_status,
            "dataset_report": self.train_state.get("dataset_report"),
            "dataset_preview": dataset_preview,
            "comparison": self.train_state.get("comparison"),
            "comparison_chart": comparison_chart,
            "workers": workers,
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
                run_id = self.orch.launch(manifest, exp_dir,
                                          trainer=manifest.training.algorithm)
                self.exp_mgr.mark_launched(manifest.experiment_id)
                ts["selected_run_id"] = run_id
                print(f"[Studio] Launched {manifest.training.algorithm.upper()} run {run_id}")

            elif prop_id == "trn_batch" and ts["selected_experiment"]:
                from sim_experiment.scheduler import BatchScheduler, RetryPolicy
                manifest = self.exp_mgr.load(ts["selected_experiment"])
                exp_dir = self.exp_mgr.experiment_dir(manifest.experiment_id)
                base_seed = manifest.random_seed
                scheduler = BatchScheduler(experiments_root=self.exp_mgr.root_dir,
                                           max_workers=2)
                batch = scheduler.create_batch(
                    manifest, exp_dir,
                    specs=[{"seed": base_seed}, {"seed": base_seed + 1}],
                    trainer=manifest.training.algorithm,
                    retry_policy=RetryPolicy(max_retries=0))
                scheduler.tick()
                ts["batch"] = {"scheduler": scheduler,
                               "batch_id": batch["batch_id"],
                               "exp_dir": exp_dir}
                self.exp_mgr.mark_launched(manifest.experiment_id)
                print(f"[Studio] Batch {batch['batch_id']} started "
                      f"({batch['jobs']} jobs)")

            elif prop_id == "trn_batch_cancel" and ts.get("batch"):
                result = ts["batch"]["scheduler"].cancel_batch(ts["batch"]["batch_id"])
                print(f"[Studio] Batch cancelled: {result['status']}")

            elif prop_id == "trn_dataset" and ts["selected_experiment"] and ts["selected_run_id"]:
                from sim_experiment.dataset import export_dataset
                exp_dir = self.exp_mgr.experiment_dir(ts["selected_experiment"])
                rd = self.run_mgr.run_dir(exp_dir, ts["selected_run_id"])
                out = os.path.join(rd, "dataset_export")
                ts["dataset_report"] = export_dataset(rd, out)
                ts["dataset_dir"] = out
                print(f"[Studio] Dataset exported: "
                      f"{ts['dataset_report']['episodes']} episodes, "
                      f"{ts['dataset_report']['steps']} steps -> {out}")

            elif prop_id == "trn_compare" and ts["selected_experiment"]:
                from sim_experiment.analytics import compare_runs, list_run_dirs
                exp_dir = self.exp_mgr.experiment_dir(ts["selected_experiment"])
                ts["comparison"] = compare_runs(
                    list_run_dirs(exp_dir), metric="reward", scope="episode",
                    smooth_window=10)
                n = len(ts["comparison"]["series"])
                print(f"[Studio] Compared {n} runs on episode reward")

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
                        manifest, exp_dir, trainer=manifest.training.algorithm,
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
                    policy = make_policy_from_checkpoint(
                        os.path.join(rd, ckpt["path"]),
                        algorithm=manifest.training.algorithm)
                    result = evaluate_policy(
                        env, policy,
                        seeds=manifest.evaluation.eval_seeds,
                        num_episodes=manifest.evaluation.num_episodes,
                        result_kwargs={"checkpoint_path": ckpt["path"],
                                       "algorithm": manifest.training.algorithm,
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

        # Drive any active batch scheduler (dispatch + poll transitions)
        batch_rec = self.train_state.get("batch")
        if batch_rec:
            try:
                sched = batch_rec["scheduler"]
                st = sched.status(batch_rec["batch_id"])
                if st["finished"] < st["total"]:
                    sched.tick(batch_rec["batch_id"])
                elif not batch_rec.get("finalized"):
                    result = sched._finalize_batch(
                        sched._batches[batch_rec["batch_id"]], "completed")
                    batch_rec["finalized"] = True
                    print(f"[Studio] Batch {batch_rec['batch_id']} finished: "
                          f"{result['status']}")
            except Exception:
                pass

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

    # ------------------------------------------------- studio shell

    def _status(self, text: str, tone: str = "info") -> None:
        if hasattr(self, "ui_ctx"):
            self.ui_ctx.status(text, tone)
        else:
            print(f"[Studio] {text}")

    def _project_snapshot(self) -> Dict[str, Any]:
        """Full document snapshot: road + entities + agent + scenario +
        vehicle config. Covers both editor and inspector edits so undo
        applies to the whole environment document."""
        self.project.road_def = self.env.road_def
        self.project.vehicle_config = self.env.vehicle.config
        self.project.entities = self.env.entities
        if hasattr(self, "inspector"):
            self.project.agent = self.inspector.agent
            self.project.scenario_def = self.inspector.scenario_def
        return self.project.to_dict()

    def _restore_snapshot(self, snap) -> None:
        file_path = getattr(self.project, "file_path", None)
        proj = EnvironmentProject.from_dict(snap)
        self._apply_project(proj, file_path, reset_history=False)
        self.dirty = True

    def _editor_changed(self) -> None:
        """Committed editor mutation → dirty + history snapshot."""
        self.dirty = True
        if hasattr(self, "edit_history"):
            self.edit_history.push(self._project_snapshot())

    def mark_dirty(self) -> None:
        self.dirty = True
        if hasattr(self, "edit_history"):
            self.edit_history.push(self._project_snapshot())

    def undo(self) -> None:
        snap = self.edit_history.undo() if hasattr(self, "edit_history") else None
        if snap is None:
            self._status("Nothing to undo", "info")
            return
        self._restore_snapshot(snap)
        self.dirty = True
        self._status("Undo", "info")

    def redo(self) -> None:
        snap = self.edit_history.redo() if hasattr(self, "edit_history") else None
        if snap is None:
            self._status("Nothing to redo", "info")
            return
        self._restore_snapshot(snap)
        self.dirty = True
        self._status("Redo", "info")

    def _apply_project(self, proj: EnvironmentProject,
                       path: Optional[str] = None,
                       reset_history: bool = True) -> None:
        """Single project-load path: env + editor + inspector + history.
        reset_history=False when restoring via undo/redo (the stack
        manages itself)."""
        proj.file_path = path
        self.project = proj
        self.env.entities = proj.entities
        self.env.obstacles = [e for e in proj.entities
                              if getattr(e, "is_collidable", False)
                              or getattr(e, "entity_type", "")
                              in ("obstacle", "barrier", "cone")]
        self.env.set_agent(proj.agent)
        self.env.scenario_def = proj.scenario_def
        self.env.set_road_definition(proj.road_def)
        if not self.headless:
            self.renderer.load_track(self.env.track)
            self.track_editor.road_def = self.env.road_def
            self.track_editor.entities = self.env.entities
            self.track_editor.deselect_all()
            self.track_editor.tool = "select"
            self.track_editor.active_tool = None
            self.inspector.road_def = self.env.road_def
            self.inspector.vehicle_config = self.env.vehicle.config
            self.inspector.entities = self.env.entities
            self.inspector.agent = proj.agent
            self.inspector.scenario_def = proj.scenario_def
            self.inspector.active_tab = "OVERVIEW"
        self.inspector.run_validation()
        self.obs, self.step_info = self.env.reset()
        self.dirty = False
        if reset_history and hasattr(self, "edit_history"):
            self.edit_history.clear(self._project_snapshot())

    def open_project(self, proj: EnvironmentProject,
                     path: Optional[str] = None) -> None:
        """Open a project into the workspace EDIT tab."""
        self._apply_project(proj, path)
        if path:
            self.library.mark_opened(path)
            self.settings.push_recent(path)
            self.settings.save()
        self.studio_screen = "workspace"
        self.ws_tab = "EDIT"
        self.active_dialog = None
        if not self.headless:
            # frame once the real canvas rect is known (next draw)
            self.track_editor._needs_frame = True
        if hasattr(self, "home_screen"):
            self.home_screen.invalidate_thumbs()

    def open_project_path(self, path: str) -> None:
        try:
            proj = EnvironmentProject.load(path)
        except Exception as e:
            self._status(f"Cannot open track: {e}", "error")
            return
        self.open_project(proj, path)

    def new_track_from_template(self, template_id: str,
                                name: Optional[str] = None) -> None:
        proj = EnvironmentTemplateManager.create_project_from_template(
            template_id)
        if name:
            proj.name = name
            proj.road_def.name = name
        path = self.library.create(proj)
        self.open_project(proj, path)
        self._status(f"Track created: {proj.name}", "ok")

    def request_new_track(self) -> None:
        """Inspector NEW button — confirm discard, then open dialog."""
        if self.dirty:
            self.active_dialog = "confirm_new"
        else:
            self.active_dialog = "new_track"
            self.ui_ctx.inputs["nt_name"] = {
                "text": "Untitled Track", "caret": 13}

    def go_home(self) -> None:
        self.studio_screen = "home"
        self.active_dialog = None
        self.home_screen.invalidate_thumbs()

    def set_ws_tab(self, tab: str) -> None:
        self.ws_tab = tab
        if tab == "SIMULATE":
            self.renderer.camera.mode = CameraMode.CHASE

    def request_quit(self) -> None:
        if self.dirty:
            self.active_dialog = "confirm_quit"
        else:
            self.is_running = False

    # ------------------------------------------------- file dialogs

    def _tk(self, fn_name: str, **kwargs):
        """Run a native tkinter file dialog; returns "" if unavailable."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            fn = getattr(filedialog, fn_name)
            try:
                return fn(parent=root, **kwargs)
            finally:
                root.destroy()
        except Exception as e:
            print(f"[Studio] File dialog unavailable: {e}")
            return ""

    def save_project(self) -> None:
        """Save current track to its file, or Save-As if never saved."""
        if not self.project:
            return
        self.project.road_def = self.env.road_def
        self.project.vehicle_config = self.env.vehicle.config
        self.project.entities = self.env.entities
        self.project.agent = self.inspector.agent
        self.project.scenario_def = self.inspector.scenario_def
        path = getattr(self.project, "file_path", None)
        if not path:
            self.save_project_as()
            return
        self.project.save(path)
        self.dirty = False
        self.settings.push_recent(path)
        self.settings.save()
        self.home_screen.invalidate_thumbs(path)
        self._status(f"Saved {os.path.basename(path)}", "ok")

    def save_project_as(self) -> None:
        initial = os.path.join(self.library.root,
                               f"{self.project.name or 'track'}")
        path = self._tk("asksaveasfilename",
                        initialdir=self.library.root,
                        initialfile=os.path.basename(initial),
                        defaultextension=".sim.json",
                        filetypes=[("Simulation Track", "*.sim.json")])
        if not path:
            return
        self.project.file_path = path
        self.save_project()

    def open_file_dialog(self) -> None:
        path = self._tk("askopenfilename",
                        initialdir=self.library.root,
                        filetypes=[("Simulation Track", "*.sim.json")])
        if path:
            self.open_project_path(path)

    def choose_data_root(self) -> None:
        path = self._tk("askdirectory", initialdir=self.settings.data_root,
                        title="Choose data root")
        if not path:
            return
        err = self.settings.validate_data_root(path)
        if err:
            self._status(f"Invalid data root: {err}", "error")
            return
        self.settings.data_root = path
        self.settings.save()
        self.record_dir = self.settings.recordings_dir()
        self._status("Data root updated", "ok")

    def browse_recording_file(self) -> None:
        path = self._tk("askopenfilename",
                        initialdir=self.record_dir,
                        filetypes=[("Episode recording", "*.json *.gz")])
        if path:
            self.load_replay(path)

    def reveal_in_folder(self, path: str) -> None:
        try:
            target = path if os.path.isdir(path) else os.path.dirname(path)
            os.startfile(target)  # Windows
        except Exception:
            pass

    def rebuild_fonts(self) -> None:
        from sim_ui.widgets import Fonts
        self.ui_fonts = Fonts(scale=self.settings.ui_scale)
        self.ui_ctx.fonts = self.ui_fonts

    # ------------------------------------------------- recording

    def suggest_recording_name(self) -> str:
        base = (self.project.name or "episode").lower().replace(" ", "_")
        return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"

    def start_recording(self, name: str, dest_dir: str) -> None:
        obs_schema = (self.env.agent.observation_space.export_schema()
                      if self.env.agent else self.env.observation_schema.to_dict())
        act_schema = (self.env.agent.action_space.export_schema()
                      if self.env.agent else self.env.action_config.to_dict())
        rew_cfg = (self.env.agent.reward_function.to_dict()
                   if self.env.agent else self.env.reward_engine.config.to_dict())
        fp = (self.project.compute_fingerprint()
              if hasattr(self.project, "compute_fingerprint") else "")
        self.recorder.start_recording(
            track_name=self.env.road_def.name,
            seed=self.env.clock.seed,
            env_config=(self.project.to_dict()
                        if hasattr(self.project, "to_dict") else {}),
            env_version=getattr(self.project, "environment_version", "1.0.0"),
            scenario_name=(self.env.scenario_def.name
                           if self.env.scenario_def else "Default"),
            observation_schema=obs_schema,
            action_schema=act_schema,
            reward_config=rew_cfg,
            fingerprint=fp,
            scenario_config=(self.env.scenario_def.to_dict()
                             if self.env.scenario_def else {}))
        self._pending_rec = {"name": name, "dest": dest_dir}
        self._last_recorded_step = self.env.current_step
        self._status("Recording started", "ok")
        print(f"[Studio] Recording → {dest_dir}")

    def stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        self.recorder.stop_recording(
            self.step_info.get("termination_reason", "manual_stop"))
        pend = getattr(self, "_pending_rec", None) or {}
        name = pend.get("name") or self.suggest_recording_name()
        dest = pend.get("dest") or self.record_dir
        ep_dir = os.path.join(dest, name)
        os.makedirs(ep_dir, exist_ok=True)
        ep_path = os.path.join(ep_dir, "episode.json")
        self.recorder.save_to_file(ep_path)
        # lightweight directory manifest so browsers see it instantly
        manifest = {
            "kind": "episode_recording",
            "name": name,
            "steps": self.recorder.metadata.get("total_steps", 0),
            "track_name": self.recorder.metadata.get("track_name", ""),
            "simulator_version": self.recorder.metadata.get(
                "simulator_version", ""),
            "created": time.time(),
        }
        with open(os.path.join(ep_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self.recording_summary = {
            "steps": manifest["steps"],
            "duration_s": (self.recorder.frames[-1]["t"]
                           if self.recorder.frames else 0.0),
            "total_return": sum(f.get("reward", 0.0)
                                for f in self.recorder.frames),
            "termination": manifest.get("termination_reason", "manual_stop"),
            "path": ep_dir,
        }
        self.active_dialog = "rec_summary"
        print(f"[Studio] Saved episode recording to {ep_path}")

    def _record_step_if_needed(self) -> None:
        """Record the latest env step — works for manual AND
        externally-driven episodes."""
        if not self.recorder.is_recording:
            return
        if self.env.current_step <= self._last_recorded_step:
            return
        self._last_recorded_step = self.env.current_step
        st = self.env.vehicle.state
        info = getattr(self.env, "last_step_info", None) or self.step_info
        reward = getattr(self.env, "last_step_reward", 0.0)
        self.recorder.record_step(
            step=self.env.current_step,
            sim_time=self.env.clock.sim_time,
            vehicle_pos=(st.pos.x, st.pos.y, st.pos.z),
            vehicle_yaw=st.yaw,
            speed=st.speed,
            action=list(self.env.last_action),
            reward=reward,
            reward_breakdown=info.get("reward_breakdown", {}),
            telemetry=info,
            is_colliding=info.get("is_colliding", False))

    # ------------------------------------------------- replay

    def load_replay(self, path: str) -> None:
        try:
            data = EpisodeRecorder.load_from_file(path)
        except Exception as e:
            self._status(f"Cannot load recording: {e}", "error")
            return
        self.replay_player.load_recording(data)
        self.replay_path = path
        self._apply_replay_frame()
        self.ws_tab = "REPLAY"
        self.renderer.camera.mode = CameraMode.CHASE
        self._status(
            f"Loaded {self.replay_player.total_frames} frames", "ok")

    def _apply_replay_frame(self) -> None:
        frame = self.replay_player.get_current_frame()
        if not frame:
            return
        st = self.env.vehicle.state
        pos = frame.get("pos", [st.pos.x, st.pos.y, st.pos.z])
        st.pos.x, st.pos.y, st.pos.z = pos[0], pos[1], pos[2]
        st.yaw = frame.get("yaw", st.yaw)
        # speed is derived from vel_body — set longitudinal velocity directly
        spd = frame.get("speed", 0.0)
        st.vel_body = Vec2(spd, 0.0)
        fwd = Vec2(math.cos(st.yaw), math.sin(st.yaw))
        st.vel_world = fwd * spd

    def _replay_tick(self, dt: float) -> None:
        p = self.replay_player
        if self.ws_tab != "REPLAY" or not p.frames or not p.is_playing:
            return
        self.replay_accum += dt * p.playback_speed * 60.0
        while self.replay_accum >= 1.0:
            self.replay_accum -= 1.0
            p.step_forward()
        self._apply_replay_frame()

    def open_experiment_env(self, exp_id: str) -> None:
        """Open the environment snapshot of an experiment as a track."""
        try:
            manifest = self.exp_mgr.load(exp_id)
            proj = EnvironmentProject.from_dict(manifest.environment)
            proj.file_path = None  # snapshot is not a library file
            self.open_project(proj, None)
            self.mark_dirty()  # snapshot is a working copy, not saved
        except Exception as e:
            self._status(f"Cannot open experiment env: {e}", "error")

    # ------------------------------------------------- event routing

    def _dispatch_action(self, action: str, payload: Any) -> None:
        """Single dispatch point for every UI action."""
        if action.startswith("dlg_"):
            self._handle_dialog_action(action, payload)
            return
        if action.startswith(("prop_", "tab_", "action_")):
            self._route_inspector_action(action)
            return
        if self.studio_screen == "home":
            # home handles its own actions; workspace handles shared ones
            # (dataset/experiment actions also exist on the home screen)
            if not self.home_screen.on_action(action, payload):
                self.workspace.on_action(action, payload)
        else:
            if (not self.workspace.on_action(action, payload)
                    and self.ws_tab == "EDIT"):
                self.workspace.editor_ui.on_action(action, payload)

    def _route_inspector_action(self, action_id: str) -> None:
        insp = self.inspector
        if action_id.startswith("tab_"):
            insp.active_tab = action_id.replace("tab_", "")
            return
        if action_id.startswith("prop_nav_"):
            insp.handle_nav(action_id.replace("prop_nav_", ""))
            return
        mutated = True
        if action_id.startswith(("prop_minus_", "prop_plus_")):
            sign = -1.0 if action_id.startswith("prop_minus_") else 1.0
            pid = action_id.split("prop_minus_", 1)[-1] if sign < 0 \
                else action_id.split("prop_plus_", 1)[-1]
            step = sign
            for p in insp.get_properties_for_active_tab():
                if p.prop_id == pid:
                    step = sign * p.step
                    break
            insp.handle_property_change(pid, step)
        elif action_id.startswith("prop_toggle_"):
            insp.handle_property_change(
                action_id.replace("prop_toggle_", ""), None)
        elif action_id.startswith("prop_enum_"):
            insp.handle_property_change(
                action_id.replace("prop_enum_", ""), None)
        elif action_id.startswith("prop_act_"):
            place_map = {
                "prop_act_sc_add_obstacle": "obstacle",
                "prop_act_sc_add_barrier": "barrier",
                "prop_act_sc_add_cone": "cone",
                "prop_act_sc_add_sign": "traffic_sign",
                "prop_act_sc_add_light": "traffic_light",
            }
            if action_id in place_map:
                self.track_editor.active_tool = place_map[action_id]
            elif action_id.startswith("prop_act_sc_ent_sel_"):
                eid = action_id.replace("prop_act_sc_ent_sel_", "")
                self.track_editor.select_entity(eid)
                insp.select_entity(eid)
            else:
                insp.handle_property_change(
                    action_id.replace("prop_act_", ""), None)
        elif action_id == "action_rebuild_mesh":
            if insp.on_rebuild_mesh:
                insp.on_rebuild_mesh()
            self.mark_dirty()
            return
        elif action_id == "action_save_project":
            self.save_project()
            return
        elif action_id == "action_load_project":
            self.open_file_dialog()
            return
        elif action_id == "action_new_project":
            self.request_new_track()
            return
        else:
            mutated = False
        if mutated:
            self._editor_changed()

    def _handle_dialog_action(self, action: str, payload: Any) -> None:
        kind = action.split(":", 1)
        verb, dlg_id = kind[0], (kind[1] if len(kind) > 1 else "")
        if verb == "dlg_cancel":
            self.active_dialog = None
            return
        # dlg_confirm:<id>
        if dlg_id == "new":
            name = (payload or {}).get("name") or "Untitled Track"
            tid = (payload or {}).get("template_id") or "empty"
            self.active_dialog = None
            self.new_track_from_template(tid, name)
        elif dlg_id == "delete_track":
            path = (self.dialog_payload or {}).get("path")
            self.active_dialog = None
            if path:
                try:
                    self.library.delete(path)
                    self.home_screen.invalidate_thumbs()
                    self._status("Track deleted", "ok")
                except Exception as e:
                    self._status(f"Delete failed: {e}", "error")
        elif dlg_id == "rename":
            path = (self.dialog_payload or {}).get("path")
            name = (payload or {}).get("name", "").strip()
            self.active_dialog = None
            if path and name:
                try:
                    self.library.rename(path, name)
                    if getattr(self.project, "file_path", None) == path:
                        self.project.name = name
                        self.project.road_def.name = name
                    self.home_screen.invalidate_thumbs()
                    self._status(f"Renamed to {name}", "ok")
                except Exception as e:
                    self._status(f"Rename failed: {e}", "error")
        elif dlg_id == "delete_ds":
            path = (self.dialog_payload or {}).get("path")
            self.active_dialog = None
            if path and os.path.isdir(path):
                import shutil
                try:
                    shutil.rmtree(path)
                    if self.dataset_detail == path:
                        self.dataset_detail = None
                    self._status("Deleted", "ok")
                except Exception as e:
                    self._status(f"Delete failed: {e}", "error")
        elif dlg_id == "quit":
            self.is_running = False
        elif dlg_id == "discard_new":
            self.active_dialog = "new_track"
            self.ui_ctx.inputs["nt_name"] = {
                "text": "Untitled Track", "caret": 13}
        elif dlg_id == "record":
            self.active_dialog = None
            self.start_recording((payload or {}).get("name"),
                                 (payload or {}).get("dest"))
        elif dlg_id == "rec_open_folder":
            self.active_dialog = None
            if payload:
                self.reveal_in_folder(payload)
        elif dlg_id == "rec_view_replay":
            self.active_dialog = None
            if payload:
                ep = os.path.join(payload, "episode.json") \
                    if os.path.isdir(payload) else payload
                self.load_replay(ep)
        elif dlg_id == "rec_again":
            self.active_dialog = "record"
            self.ui_ctx.inputs["rec_name"] = {
                "text": self.suggest_recording_name(), "caret": 0}

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
                if (not external_stepped and self.studio_screen == "workspace"
                        and self.ws_tab == "SIMULATE"):
                    self._handle_keyboard_drive()
                    act = [self.manual_steer, self.manual_throttle, self.manual_brake]
                    self.obs, reward, term, trunc, self.step_info = self.env.step(act)

                    if term or trunc:
                        self.obs, self.step_info = self.env.reset()

                # Recording captures both manual and externally-driven steps
                self._record_step_if_needed()

                # Replay playback advancement
                self._replay_tick(dt)

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
        ctx = self.ui_ctx
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.request_quit()
                continue

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.ui_renderer.resize(event.w, event.h)
                continue

            elif event.type == pygame.MOUSEMOTION:
                ctx.mouse_pos = event.pos
                if (self.studio_screen == "workspace" and self.ws_tab == "EDIT"):
                    self.workspace.editor_ui.canvas_mouse_move(event.pos)
                continue

            elif event.type == pygame.TEXTINPUT:
                ctx.text_input_event(event)
                continue

            elif event.type == pygame.KEYDOWN:
                # 1. focused text input consumes keys
                if ctx.key_down(event):
                    continue
                # 2. dialogs
                if self.active_dialog:
                    if event.key == pygame.K_ESCAPE:
                        self.active_dialog = None
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if self.active_dialog == "new_track":
                            from sim_ui.screens import dialogs as _dlg
                            self._handle_dialog_action(
                                "dlg_confirm:new", {
                                    "name": ctx.inputs.get(
                                        "nt_name", {}).get("text", ""),
                                    "template_id": getattr(
                                        _dlg.draw_new_track, "selected",
                                        "empty")})
                    continue
                # 3. screen shortcuts
                ctrl = bool(event.mod & pygame.KMOD_CTRL)
                if ctrl and event.key == pygame.K_s and self.studio_screen == "workspace":
                    self.save_project()
                    continue
                if ctrl and event.key == pygame.K_o and self.studio_screen == "workspace":
                    self.open_file_dialog()
                    continue
                if self.studio_screen == "workspace" and self.ws_tab == "EDIT":
                    if self.workspace.editor_ui.handle_key(event):
                        continue
                elif self.studio_screen == "workspace" and self.ws_tab == "SIMULATE":
                    if event.key == pygame.K_r:
                        self.obs, self.step_info = self.env.reset()
                    elif event.key == pygame.K_c:
                        cams = [CameraMode.CHASE, CameraMode.HOOD,
                                CameraMode.TOP_DOWN, CameraMode.ORBIT]
                        idx = (cams.index(self.renderer.camera.mode) + 1) % len(cams)
                        self.renderer.camera.mode = cams[idx]
                    elif event.key == pygame.K_TAB:
                        self.hud.show_obs_inspector = \
                            not self.hud.show_obs_inspector
                    continue
                elif self.studio_screen == "workspace" and self.ws_tab == "REPLAY":
                    if event.key == pygame.K_SPACE:
                        self.replay_player.toggle_play()
                    elif event.key == pygame.K_LEFT:
                        self.replay_player.step_backward()
                        self._apply_replay_frame()
                    elif event.key == pygame.K_RIGHT:
                        self.replay_player.step_forward()
                        self._apply_replay_frame()
                    continue
                # 4. escape hierarchy: dialog > tool > selection > home > quit
                if event.key == pygame.K_ESCAPE:
                    if self.studio_screen == "workspace":
                        self.go_home()
                    else:
                        self.request_quit()
                continue

            elif event.type == pygame.MOUSEBUTTONDOWN:
                r = ctx.mouse_down(event.pos, event.button)
                if r is not None:
                    kind = r.get("kind")
                    if r.get("action"):
                        self._dispatch_action(r["action"], r.get("payload"))
                    elif (kind == "canvas" and self.studio_screen == "workspace"
                          and self.ws_tab == "EDIT"):
                        ed = self.track_editor
                        ed.handle_mouse_down(event.pos, event.button)
                        # selection sync → inspector
                        if ed.selected_entity_id:
                            self.inspector.select_entity(ed.selected_entity_id)
                        elif ed.selected_point_idx is not None:
                            self.inspector.select_control_point(
                                ed.selected_point_idx)
                continue

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.studio_screen == "workspace" and self.ws_tab == "EDIT":
                    self.workspace.editor_ui.canvas_mouse_up()
                continue

            elif event.type == pygame.MOUSEWHEEL:
                if ctx.mouse_wheel(ctx.mouse_pos, event.y):
                    continue
                if (self.studio_screen == "workspace" and self.ws_tab == "EDIT"):
                    # inspector scroll if cursor over panel, else canvas zoom
                    pa = self.inspector.props_area
                    if pa and pa.collidepoint(ctx.mouse_pos):
                        self.inspector.scroll(event.y)
                    else:
                        self.workspace.editor_ui.canvas_wheel(
                            ctx.mouse_pos, event.y)
                continue

    def _render(self) -> None:
        """Renders the active screen: 3D viewport (when needed) + UI overlay."""
        self.gl_ctx.screen.use()
        self.gl_ctx.viewport = (0, 0, self.width, self.height)

        # 1. 3D scene — only in the tabs that show a 3D viewport.
        #    The EDIT tab uses a dedicated 2D canvas; drawing 3D behind it
        #    was the source of the old double-visualization problem.
        show_3d = (self.studio_screen == "workspace"
                   and (self.ws_tab == "SIMULATE"
                        or (self.ws_tab == "REPLAY"
                            and self.replay_player.total_frames > 0)))
        if show_3d:
            all_scene_obs = self.env.obstacles + [
                e for e in self.env.entities if e not in self.env.obstacles]
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
        else:
            import sim_ui.theme as _t
            c = _t.C.bg
            self.gl_ctx.clear(c[0] / 255, c[1] / 255, c[2] / 255)

        # 2. UI overlay via the Phase-7 widget context
        self.ui_renderer.clear()
        self.ui_ctx.surface = self.ui_renderer.ui_surface
        self.ui_ctx.begin_frame()
        if self.studio_screen == "home":
            self.home_screen.draw(self.ui_ctx)
        else:
            self.workspace.draw(self.ui_ctx)
        from sim_ui.widgets import tooltip, status_toast
        tooltip(self.ui_ctx)
        status_toast(self.ui_ctx, self.width, self.height)
        self.ui_renderer.render_to_screen()

    def cleanup(self) -> None:
        self.server.stop()
        if not self.headless:
            pygame.quit()
        print("[Studio] Simulator shut down cleanly.")
