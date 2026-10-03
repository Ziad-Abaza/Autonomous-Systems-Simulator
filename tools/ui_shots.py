"""
UI screenshot capture tool — drives the real app offscreen and saves PNGs
of every major screen for manual UX validation.

    python tools/ui_shots.py [width height]

Requires a GPU/GL context (osmesa/angle works). Output:
    docs/phase7_shots/<WxH>_<name>.png
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
from sim_ui.app import SimulationStudioApp
from sim_render.camera import CameraMode
from sim_project.presets import (create_oval_circuit,
                                 create_serpentine_track)

W = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
H = int(sys.argv[2]) if len(sys.argv) > 2 else 720
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "phase7_shots")
os.makedirs(OUT, exist_ok=True)

app = SimulationStudioApp(width=W, height=H, headless=False, port=8799)

# Force dark theme for the evidence set — settings persist across runs
import sim_ui.theme as _T
_T.set_theme("dark")


def shot(name):
    app._render()
    data = app.gl_ctx.screen.read(components=3)  # before flip!
    pygame.display.flip()
    img = pygame.image.frombuffer(data, (W, H), "BGR").convert()
    img = pygame.transform.flip(img, False, True)
    pygame.image.save(img, os.path.join(OUT, f"{W}x{H}_{name}.png"))
    print("saved", name)


# settle frames with real camera update
for _ in range(30):
    st = app.env.vehicle.state
    app.renderer.camera.update(st.pos, st.yaw, 1 / 60.0)
    app.env.step([0.0, 0.5, 0.0])
    app._render()
    pygame.display.flip()

# ---- HOME ----
app.studio_screen = "home"
app.home_screen.section = "TRACKS"
shot("h1_home_tracks")
app.home_screen.section = "TEMPLATES"
shot("h2_home_templates")
app.home_screen.section = "DATASETS"
shot("h3_home_datasets")
app.home_screen.section = "EXPERIMENTS"
shot("h4_home_experiments")
app.home_screen.section = "SETTINGS"
shot("h5_home_settings")

# new-track dialog
app.active_dialog = "new_track"
shot("h6_new_track_dialog")
app.active_dialog = None

# confirm dialog
app.active_dialog = "confirm_delete"
app.dialog_payload = {"path": r"D:\tracks\demo.sim.json", "name": "Demo"}
shot("h7_confirm_delete")
app.active_dialog = None
app.dialog_payload = None

# ---- WORKSPACE: EDIT ----
app.open_project(create_oval_circuit(), None)
app.ws_tab = "EDIT"
shot("e1_edit_oval")

for tab in ["OVERVIEW", "TRACK", "POINT", "AGENT", "OBS", "REWARD",
            "VALIDATE", "TRAIN"]:
    app.inspector.active_tab = tab
    shot(f"e2_insp_{tab.lower()}")

# serpentine — framing stress test
app.open_project(create_serpentine_track(), None)
shot("e3_edit_serpentine")
# unfitted view to compare
app.track_editor.zoom = 4.0
app.track_editor.view_offset_x = 0
app.track_editor.view_offset_y = 0
shot("e4_serpentine_unfitted")
app.track_editor.frame_all()

# ---- WORKSPACE: SIMULATE ----
app.open_project(create_oval_circuit(), None)
app.ws_tab = "SIMULATE"
app.renderer.camera.mode = CameraMode.CHASE
for _ in range(30):
    st = app.env.vehicle.state
    app.renderer.camera.update(st.pos, st.yaw, 1 / 60.0)
    app.env.step([0.0, 0.5, 0.0])
shot("s1_simulate")

app.hud.show_obs_inspector = True
shot("s2_obs_inspector")
app.hud.show_obs_inspector = False

# recording dialog
app.active_dialog = "record"
shot("s3_record_dialog")
app.active_dialog = None
# recording in progress state
app.recorder.is_recording = True
app.recorder.frames = [{}] * 42
shot("s4_recording_active")
app.recorder.is_recording = False
app.recorder.frames = []

# ---- WORKSPACE: REPLAY ----
app.ws_tab = "REPLAY"
shot("r1_replay_empty")
# record a few frames then load into player
app.recorder.start_recording(track_name="demo", seed=1)
for i in range(60):
    app.env.step([0.1, 0.4, 0.0])
    st = app.env.vehicle.state
    app.recorder.record_step(i, i / 60.0, (st.pos.x, st.pos.y, st.pos.z),
                             st.yaw, st.speed, [0.1, 0.4, 0.0], 0.0,
                             {}, {}, False)
app.recorder.stop_recording("demo")
app.replay_player.load_recording({"metadata": app.recorder.metadata,
                                  "frames": app.recorder.frames})
app.replay_player.seek(30)
app._apply_replay_frame()
for _ in range(10):
    st = app.env.vehicle.state
    app.renderer.camera.update(st.pos, st.yaw, 1 / 60.0)
    app._render()
shot("r2_replay_loaded")

# ---- WORKSPACE: DATA ----
app.ws_tab = "DATA"
shot("d1_data_datasets")

# ---- light theme ----
import sim_ui.theme as T
T.set_theme("light")
app.studio_screen = "home"
app.home_screen.section = "TRACKS"
shot("l1_home_light")
app.studio_screen = "workspace"
app.ws_tab = "EDIT"
shot("l2_edit_light")
app.ws_tab = "SIMULATE"
shot("l3_sim_light")
T.set_theme("dark")

app.cleanup()
print("done")
