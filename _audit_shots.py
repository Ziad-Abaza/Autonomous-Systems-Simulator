"""Temporary Phase-7 audit tool: launch the real app, render each major
UI state, and capture the GL framebuffer to docs/phase7_shots/.
Not part of the product - deleted after audit."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("docs/phase7_shots", exist_ok=True)

import numpy as np
from PIL import Image
import pygame

from sim_ui.app import SimulationStudioApp
from sim_render.camera import CameraMode

W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1600, 900)
tag = f"{W}x{H}"

app = SimulationStudioApp(width=W, height=H, headless=False, port=8799)

def shot(name):
    app._render()
    data = app.gl_ctx.screen.read(components=3)  # read back buffer BEFORE flip
    pygame.display.flip()
    img = Image.frombytes("RGB", (W, H), data).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(f"docs/phase7_shots/{tag}_{name}.png")
    print("saved", name)

# settle a few frames with real camera update + env step like run() does
for _ in range(30):
    st = app.env.vehicle.state
    app.renderer.camera.update(st.pos, st.yaw, 1/60.0)
    app.env.step([0.0, 0.5, 0.0])
    app._render(); pygame.display.flip()

shot("01_sim")
app.inspector  # ensure built

# Editor mode
app.active_mode = "mode_editor"
app.renderer.camera.mode = CameraMode.TOP_DOWN
shot("02_editor_overview")

# Inspector tabs worth seeing
for tab in ["SCENE", "TRACK", "AGENT", "OBS", "REWARD", "VALIDATE", "TRAIN"]:
    app.inspector.active_tab = tab
    shot(f"03_tab_{tab.lower()}")

# Replay mode
app.active_mode = "mode_replay"
shot("04_replay")

# Observation inspector overlay in sim mode
app.active_mode = "mode_sim"
app.hud.show_obs_inspector = True
shot("05_obs_inspector")
app.hud.show_obs_inspector = False

# Serpentine track in editor (framing problem demonstration)
from sim_project.presets import create_serpentine_track
proj = create_serpentine_track()
app.env.set_road_definition(proj.road_def)
app.renderer.load_track(app.env.track)
app.track_editor.road_def = app.env.road_def
app.inspector.road_def = app.env.road_def
app.active_mode = "mode_editor"
shot("06_editor_serpentine")

# Large custom track framing check
from sim_project.serializer import EnvironmentProject
p = EnvironmentProject.load("presets/custom_environment.sim.json")
app.env.set_road_definition(p.road_def)
app.renderer.load_track(app.env.track)
app.track_editor.road_def = app.env.road_def
shot("07_editor_custom")

app.cleanup()
print("done")
