"""Capture additional docs screenshots: inspector ACTION/TERM/SCENARIO/SCENE/ENTITY
tabs and the DYNAMICS workspace tab. Based on tools/ui_shots.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
from sim_ui.app import SimulationStudioApp
from sim_render.camera import CameraMode
from sim_project.presets import create_oval_circuit

W, H = 1600, 900
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)

app = SimulationStudioApp(width=W, height=H, headless=False, port=8799)
import sim_ui.theme as _T
_T.set_theme("dark")


def shot(name):
    app._render()
    data = app.gl_ctx.screen.read(components=3)
    pygame.display.flip()
    img = pygame.image.frombuffer(data, (W, H), "BGR").convert()
    img = pygame.transform.flip(img, False, True)
    pygame.image.save(img, os.path.join(OUT, f"{name}.png"))
    print("saved", name)


for _ in range(30):
    st = app.env.vehicle.state
    app.renderer.camera.update(st.pos, st.yaw, 1 / 60.0)
    app.env.step([0.0, 0.5, 0.0])
    app._render()
    pygame.display.flip()

app.open_project(create_oval_circuit(), None)
app.ws_tab = "EDIT"

for tab, name in [("ACTION", "inspector-action"),
                  ("TERM", "inspector-termination"),
                  ("SCENARIO", "inspector-scenario"),
                  ("SCENE", "inspector-scene"),
                  ("ENTITY", "inspector-entity")]:
    app.inspector.active_tab = tab
    shot(name)

# DYNAMICS workspace tab
app.ws_tab = "DYNAMICS"
shot("dynamics.png".replace(".png", ""))

app.cleanup()
print("done")
