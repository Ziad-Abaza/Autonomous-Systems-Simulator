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

def mdown(pos, btn=1):
    app.ui_ctx.mouse_pos = pos
    pygame.event.post(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, pos=pos, button=btn))
    app._handle_events()


def mmove(pos):
    app.ui_ctx.mouse_pos = pos
    pygame.event.post(pygame.event.Event(
        pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))
    app._handle_events()


def mup():
    pygame.event.post(pygame.event.Event(
        pygame.MOUSEBUTTONUP, pos=app.ui_ctx.mouse_pos, button=1))
    app._handle_events()


def wheel(pos, y):
    app.ui_ctx.mouse_pos = pos
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=y))
    app._handle_events()


ed = app.track_editor
frame()
cv = app.workspace.editor_ui.canvas_rect
empty = (cv.x + 24, cv.y + 24)   # in-canvas spot far from the track

# ---- select + drag a control point (cp[0] coincides with the spawn
#      marker — spawn wins the hit test; use cp[2] instead)
cp = ed.road_def.control_points[2]
ox, oy = cp.x, cp.y
sx, sy = ed.world_to_screen(cp.x, cp.y)
mdown((int(sx), int(sy)))
check("point selected on click", ed.selected_point_idx == 2)
check("drag starts on click", ed.is_dragging_point)
mmove((int(sx) + 40, int(sy) + 25))
mup()
check("drag moved control point",
      abs(cp.x - ox) > 0.1 or abs(cp.y - oy) > 0.1)
check("dirty after drag", app.dirty)

# ---- deselect via empty-space click (inside the canvas)
mdown(empty)
check("empty click deselects",
      ed.selected_point_idx is None and not ed.selected_entity_id)

# ---- wheel zoom anchored on canvas
z0 = ed.zoom
wheel(cv.center, 1)
check("wheel zooms in", ed.zoom > z0)
wheel(cv.center, -1)
check("wheel zooms out", abs(ed.zoom - z0) < 0.01)

# ---- RMB pan on empty space inside the canvas
ed.view_offset_x = 0.0
ed.view_offset_y = 0.0
mdown(empty, btn=3)
mmove((empty[0] + 60, empty[1] - 30))
mup()
check("RMB pans view", abs(ed.view_offset_x) > 0.01 or
      abs(ed.view_offset_y) > 0.01)

# ---- frame all: every control point projects inside the canvas
ed.frame_all()
inside = all(cv.collidepoint(ed.world_to_screen(p.x, p.y))
             for p in ed.road_def.control_points)
check("frame_all keeps all points visible", inside)

# ---- draw tool inserts a point on an edge
ed.tool = "draw"
n0 = len(ed.road_def.control_points)
p1 = ed.road_def.control_points[0]
p2 = ed.road_def.control_points[1]
s1 = ed.world_to_screen(p1.x, p1.y)
s2 = ed.world_to_screen(p2.x, p2.y)
mid = ((s1[0] + s2[0]) // 2, (s1[1] + s2[1]) // 2)
mdown(mid)
mup()
check("draw tool inserts point on edge",
      len(ed.road_def.control_points) == n0 + 1)
ed.tool = "select"

# ---- delete via Del key
key(pygame.K_DELETE)
check("Del deletes selected point",
      len(ed.road_def.control_points) == n0)

# ---- undo restores the deleted point
key(pygame.K_z, pygame.KMOD_CTRL)
check("undo restores point",
      len(ed.road_def.control_points) == n0 + 1)

# ---- place entity via placement tool
ed.active_tool = "obstacle"
c0 = len(ed.entities)
mdown(cv.center)
check("entity placed on canvas", len(ed.entities) == c0 + 1)
check("entity selected after placement", bool(ed.selected_entity_id))

# ---- snap enabled → entity position snapped to integer
ed.snap_enabled = True
ed.active_tool = "cone"
mdown((cv.centerx + 37, cv.centery + 19))
ent = ed.entities[-1]
check("snap rounds position",
      abs(ent.pos.x - round(ent.pos.x)) < 1e-6
      and abs(ent.pos.y - round(ent.pos.y)) < 1e-6)
ed.snap_enabled = False

# ---- drag entity
es = ed.world_to_screen(ent.pos.x, ent.pos.y)
ox, oy = ent.pos.x, ent.pos.y
ed.select_entity(ent.entity_id)
mdown((int(es[0]), int(es[1])))
mmove((int(es[0]) + 25, int(es[1]) + 15))
mup()
check("entity drag moves it",
      abs(ent.pos.x - ox) > 0.1 or abs(ent.pos.y - oy) > 0.1)

# ---- Phase 7.5: Loop toggle closes an open track
was_closed = ed.road_def.is_closed
click_action("ed_loop")
check("Loop toggle flips topology",
      ed.road_def.is_closed != was_closed)

# ---- Phase 7.5: draw tool closes loop at first point
if ed.road_def.is_closed:
    click_action("ed_loop")  # back to open
ed.tool = "draw"
cp0 = ed.road_def.control_points[0]
s0 = ed.world_to_screen(cp0.x, cp0.y)
mdown((int(s0[0]) + 2, int(s0[1]) + 1))
mup()
check("draw click on first point closes track",
      ed.road_def.is_closed is True)
ed.tool = "select"
click_action("ed_loop")  # reopen for remaining checks
ed.tool = "select"

# ---- Phase 7.5: entity editing via inspector properties
insp = app.inspector
ent0 = ed.entities[0] if ed.entities else None
if ent0 is not None:
    insp.select_entity(ent0.entity_id)
    ex0 = ent0.pos.x
    insp.handle_property_change("ent_pos_x", 5.0)
    check("inspector entity edit mutates position",
          abs(ent0.pos.x - (ex0 + 5.0)) < 1e-6)
else:
    check("inspector entity edit mutates position", False)

# ---- Phase 7.5: SENSORS tab — add second camera
frame()
if click_action("tab_SENSORS"):
    frame()
    n_cams = len([c for c in insp.agent.sensor_configs
                  if c.sensor_type == "camera_rgb"])
    click_action("prop_act_sen_add_camera")
    check("SENSORS add camera",
          len([c for c in insp.agent.sensor_configs
               if c.sensor_type == "camera_rgb"]) == n_cams + 1)
    # camera registered as an image channel in the obs contract
    names = [s["name"] for s in
             insp.agent.observation_space.image_channel_specs()]
    new_cam = [c for c in insp.agent.sensor_configs
               if c.sensor_type == "camera_rgb"][-1]
    check("camera in observation contract", new_cam.name in names)
    # runtime suite was rebuilt with the new camera
    check("runtime sensor rebuilt",
          new_cam.name in app.env.sensors.sensors)
else:
    check("SENSORS tab exists", False)
    check("SENSORS add camera", False)
    check("camera in observation contract", False)
    check("runtime sensor rebuilt", False)

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
