"""
End-to-end interaction smoke test — renders real frames so hit regions
register, then posts synthetic pygame events through _handle_events and
asserts the app state transitions.

    python tools/smoke_interactions.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
from sim_ui.app import SimulationStudioApp

app = SimulationStudioApp(width=1280, height=720, headless=False, port=8797)
checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def frame():
    app._render()
    pygame.display.flip()


def click_action(action_id, contains=False):
    """Render one frame, find a region with this action id, click it."""
    frame()
    for r in app.ui_ctx.regions:
        a = r.get("action")
        if a and (a == action_id or (contains and action_id in a)):
            cx, cy = r["rect"].center
            app.ui_ctx.mouse_pos = (cx, cy)
            ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                    pos=(cx, cy), button=1)
            pygame.event.post(ev)
            app._handle_events()
            return True
    print(f"  (no region for {action_id})")
    return False


def key(k, mod=0):
    ev = pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod)
    pygame.event.post(ev)
    app._handle_events()


# ---- HOME → new track flow
frame()
check("boots to home", app.studio_screen == "home")
click_action("nav", False) or check("nav region exists", False)
# click New Track button
assert click_action("new_track"), "no New Track button"
check("new-track dialog opens", app.active_dialog == "new_track")
frame()
# pick template list row (nt_template) then create
click_action("nt_template", contains=True)
assert click_action("dlg_confirm:new", contains=True)
check("workspace opens after create", app.studio_screen == "workspace")
check("lands on EDIT tab", app.ws_tab == "EDIT")
check("project saved to library",
      bool(getattr(app.project, "file_path", None))
      and os.path.exists(app.project.file_path))

# ---- workspace tabs
assert click_action("ws_tab", contains=False) or True
frame()
for r in app.ui_ctx.regions:
    if r.get("action") == "ws_tab" and r.get("payload") == "SIMULATE":
        cx, cy = r["rect"].center
        app.ui_ctx.mouse_pos = (cx, cy)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(cx, cy), button=1))
        app._handle_events()
check("SIMULATE tab", app.ws_tab == "SIMULATE")

frame()
for r in app.ui_ctx.regions:
    if r.get("action") == "ws_tab" and r.get("payload") == "EDIT":
        cx, cy = r["rect"].center
        app.ui_ctx.mouse_pos = (cx, cy)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(cx, cy), button=1))
        app._handle_events()
check("back on EDIT", app.ws_tab == "EDIT")

# ---- canvas click selects something / deselect
frame()
cv = app.workspace.editor_ui.canvas_rect
app.ui_ctx.mouse_pos = cv.center
pygame.event.post(pygame.event.Event(
    pygame.MOUSEBUTTONDOWN, pos=cv.center, button=1))
app._handle_events()
check("canvas click handled without crash", True)

# ---- undo stack works after a mutation
import copy
before = copy.deepcopy(app.track_editor.road_def.to_dict())
cp = app.track_editor.road_def.control_points[0]
cp.x += 5.0
app._editor_changed()
check("dirty after mutation", app.dirty)
key(pygame.K_z, pygame.KMOD_CTRL)
after = app.track_editor.road_def.control_points[0].x
check("ctrl+z undoes point move",
      abs(after - (cp.x - 5.0)) < 0.01 or abs(after - before["control_points"][0]["x"]) < 0.01)

# ---- save via Ctrl+S
key(pygame.K_s, pygame.KMOD_CTRL)
check("save clears dirty", not app.dirty)

# ---- back to library
frame()
click_action("ws_home")
check("back to library", app.studio_screen == "home")

app.cleanup()
failed = [n for n, ok in checks if not ok]
print(f"\n{len(checks)-len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
