---
noteId: "690ea000bf4911f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — UX Audit

Audit of the application **as it actually runs**, not just as the code reads.
Evidence: source inspection (`sim_ui/`, `sim_render/`, `sim_recorder/`,
`sim_project/`, `sim_experiment/`) plus live screenshots captured from the
real application at 1280×720, 1600×900 and 1920×1080 in
`docs/phase7_shots/`.

Platform reality: the app is a **pygame + ModernGL desktop application**.
There is no DOM, no browser, no widget toolkit. All UI is immediate-mode
pygame drawing on an alpha surface that is blitted over the GL framebuffer
each frame (`UIOverlayRenderer`). This constrains and shapes every UX
decision in Phase 7.

---

## 1. Current screens

One window, three top-level modes switched by top-bar tabs
(`SIMULATION`, `TRACK EDITOR`, `REPLAY` — `sim_ui/hud.py:52`):

| Screen | Contents | Evidence |
|---|---|---|
| Simulation | 3D viewport + left telemetry panel + right reward-decomposition panel + camera-sensor PiP + top bar + bottom bar | `1600x900_01_sim.png` |
| Track Editor | 3D viewport + 2D spline overlay drawn over it + floating inspector panel (350×620) + top/bottom bars | `1600x900_02_editor_overview.png` |
| Replay | Bare 3D viewport + top bar + bottom bar with play/scrub controls | `1280x720_04_replay.png` |
| Observation inspector | Modal overlay toggled with TAB (any mode) | `1600x900_05_obs_inspector.png` |

There is **no home screen, no library, no project picker**. The app boots
directly into the simulation of a hardcoded oval circuit
(`sim_ui/app.py:69`).

## 2. Current navigation model

- Top bar: 3 mode tabs + 4 camera-mode buttons + server status badge.
- Inspector: two rows of tabs — `OVERVIEW SCENE TRACK POINT ENTITY` /
  `AGENT OBS ACTION REWARD TERM SCENARIO VALIDATE TRAIN`
  (`sim_ui/inspector.py:66-67`).
- Mode switching is only reachable by mouse click on the top bar.
- `ESC` quits the whole application instantly (unless a placement tool is
  active), `sim_ui/app.py:674-679`.

There is no concept of "where am I" beyond the highlighted mode tab.
No breadcrumbs, no back stack, no workspace switcher.

## 3. Current user journeys

- **Launch** → simulation view of the default oval. Server starts on port
  8765 immediately.
- **Drive** → WASD/arrows while an external AI is not connected.
- **Author** → click TRACK EDITOR → draw/drag on the 2D canvas → tune
  values in inspector tabs → click REBUILD 3D ENVIRONMENT → SAVE.
- **Train** → editor mode → TRAIN tab → CREATE FROM ENV → LAUNCH.
- **Record** → bottom-bar RECORD → stop → file silently written to
  `last_episode.json` in the repo root (`app.py:814`).
- **Replay** → click REPLAY → dead end (see §14).
- **Datasets** → TRAIN tab → EXPORT DATASET on a selected run → written
  under the run's directory; previewable in the same tab.

## 4. Existing reusable UI components

| Component | Role | Location |
|---|---|---|
| `UIOverlayRenderer` | pygame surface → GL texture blit, fonts | `sim_ui/ui_overlay.py` |
| `SimulationHUD` | top bar, telemetry, reward panel, camera PiP, bottom bar, obs inspector | `sim_ui/hud.py` |
| `VisualTrackEditor` | 2D spline canvas: world↔screen transform, tools, gizmos | `sim_ui/editor.py` |
| `EnvironmentInspector` | tabbed property panel, validation, TRAIN surface | `sim_ui/inspector.py` |
| `PropertyRow` | data-driven property descriptor (float/int/bool/enum/action/label) | `sim_ui/inspector.py:36` |
| `train_providers` | headless-testable data providers for TRAIN tab | `sim_ui/train_providers.py` |

**There is no widget toolkit.** Every panel hand-codes `pygame.draw.rect`
+ `blit`. Hit-testing works by *re-invoking the draw method inside the
click handler* to obtain `(rect, action_id)` lists
(`app.py:714-719, 725-727, 803-809`) — layout logic is duplicated between
render and input paths and the two can drift.

## 5. Existing editor functionality

`VisualTrackEditor` (`sim_ui/editor.py`):

- Control points: click-select, left-drag move, click-on-segment insert,
  right-click delete (keeps ≥3), left-click empty space appends.
- Entities: obstacle/barrier/cone/sign/light placement tools (armed from
  the inspector SCENE tab), left-drag move, right-click delete.
- Spawn point: place tool + select; yaw editable in inspector.
- Gizmos: curvature-colored centerline, direction chevrons, road-boundary
  outlines, checkpoint gates with badges, spawn footprint + heading arrow,
  width circle on selected point, per-point metadata label.
- Pan: middle-drag or right-drag on empty space. Zoom: wheel, 0.5–35 px/m.

Dead affordances already in the code: `is_dragging_width` and
`is_rotating_entity` flags exist but are **never set to True**
(`editor.py:33-37`) — width handles and entity rotation were drawn but
the input path was never wired.

## 6. Existing track-authoring capabilities

- `RoadDefinition`: Catmull-Rom control points `(x, y, z, width, banking)`,
  `is_closed`, `num_checkpoints`, `spawn_point(x, y, z, yaw, initial_speed)`.
- Presets: `oval_circuit`, `serpentine_track`, `obstacle_challenge`,
  `custom_environment` in `presets/*.sim.json`.
- `EnvironmentTemplateManager`: 4 templates — `empty`, `basic_driving`,
  `lane_following`, `obstacle_avoidance` (`sim_env/templates.py`).

## 7. Current camera implementation

`SimulationCamera` (`sim_render/camera.py`): perspective-only, 60° FOV,
near 0.5 / far 1000. Modes: CHASE, HOOD, TOP_DOWN (fixed `height=90`,
vehicle-locked + `pan_offset`), ORBIT. No framing/fit logic anywhere; the
top-down mode centers on the *vehicle*, not the track.

The editor does **not** use this camera — it uses its own 2D transform
(`view_offset_*`, `zoom`) while the 3D top-down scene continues to render
behind it, producing the double-visualization confusion visible in
`1600x900_02_editor_overview.png`.

## 8. Current world/viewport scaling — the framing bug

Root causes, all verified in code:

1. `world_to_screen` maps world origin to `screen_center` which callers
   pass as `(width*0.5, height*0.5)` — the **whole window** center
   (`app.py:396`). The inspector (~375 px), top bar (42 px) and bottom
   bar (45 px) are ignored, so the usable canvas center is not the
   transform center.
2. `zoom` is a fixed default of 4.0 px/m regardless of track size. A
   ~150 m serpentine needs ~600 px vertically *plus* gizmos — it is
   clipped top and bottom at 720p/900p
   (`1600x900_06_editor_serpentine.png`, `1280x720_06_editor_serpentine.png`).
3. Wheel zoom does not anchor to the cursor — it just rescales around the
   view center (`editor.py:213-215`); the docs claim zoom-to-cursor.
4. No bounds computation exists (`RoadDefinition` bounds, entity bounds,
   spawn bounds are never aggregated for the view).
5. The 3D top-down camera has a fixed height of 90 m and follows the
   vehicle — it cannot frame a track either.
6. `canvas_rect` for the editor spans `(0, 42, width, height-87)` — the
   full window — so content renders *underneath* the inspector panel.
   Clicks under the inspector are discarded (`mouse_pos[0] > insp_w+25`)
   creating an invisible dead zone over real geometry.

## 9. Current coordinate systems

World: X = east/right, Y = north/forward, Z = up (`camera.py:23`), meters.
Editor screen transform: `sx = cx + (wx + ox)·zoom`,
`sy = cy − (wy + oy)·zoom` — Y-flip is consistent.
Angles: radians internally, degrees in labels (spawn badge shows °).
Units are displayed inconsistently — most property rows have no `unit`.

## 10. Current save/load behavior

- Save = a single hardcoded slot: `presets/custom_environment.sim.json`
  (`app.py:159-170`). Load = the same slot (`app.py:172-198`).
- `EnvironmentProject.save()` is schema-versioned (2.0/3.0, migration on
  load) and bumps `environment_version` on change — solid format.
- **No** Save As, no file picker, no recents, no dirty/modified flag, no
  unsaved-changes warning. `Ctrl+S`/`Ctrl+O` are documented in
  `AUTHORING.md` but **not implemented** — the KEYDOWN handler only knows
  ESC/R/C/TAB (`app.py:673-689`).
- NEW discards all work instantly with no confirmation.

## 11. Current recording behavior

- Bottom-bar RECORD toggles `EpisodeRecorder`; stop writes
  `last_episode.json` in the repo root — a **silent overwrite every
  session** (`app.py:803-816`).
- Metadata captured is genuinely good: sim + protocol versions, env
  fingerprint, obs/action schemas, reward config, scenario.
- Defects: frames capped at 5000 and **old frames are silently dropped**
  (`recorder.py:78-80`); recording only happens in the manual-drive
  branch, so **externally-driven AI episodes are never recorded**
  (`app.py:600-619`); there is no destination choice, no dataset naming,
  no progress summary after stop.
- `EpisodeReplayPlayer` supports load/play/pause/seek (`replay.py`) but
  nothing in the app ever calls `load_recording` — see §14.

## 12. Existing data formats

| Asset | Format | Location |
|---|---|---|
| Environment | `*.sim.json` — EnvironmentProject schema 2.0/3.0 | `presets/` |
| Episode recording | `{metadata, frames}` JSON or .gz | `last_episode.json` (root) |
| Dataset | `transitions_v1`: `manifest.json` + `episodes.jsonl` + optional `splits.json` | `<run_dir>/dataset_export/` |
| Experiment | `experiment.json` + `environment.json` + `scenario.json` + `runs/<run_id>/` | `experiments/` |
| Metrics | `metrics.jsonl` per run | run dir |

## 13. Existing dataset/trajectory infrastructure

Rich and service-layer clean (`sim_experiment/`): `export_dataset`,
`validate_dataset`, `split_dataset`, `dataset_statistics`,
`inspect_dataset`, trajectory sampling (uniform/best/mixed), BC training.
The UI touches only a fraction: TRAIN tab → EXPORT DATASET writes into
the run dir and `dataset_preview` shows validation + first 8 episodes.
There is **no dataset browser, no dataset listing, no user-chosen
location**, and no link from a dataset back to its environment.

## 14. UI/UX defects (evidence-backed)

Critical:

1. **Replay mode is dead** — `load_recording` is never called; the
   `btn_replay_play` rect is emitted but never handled; scrubber drawn
   but non-functional (`1280x720_04_replay.png`, `app.py:806-809`).
2. **Inspector tab labels overlap** — second row renders
   `ACTIONREWARE TERMSCENARIVALIDATITRAIN` (all editor screenshots).
3. **Track cannot be framed** — see §8; serpentine/custom tracks clip
   (screenshots).
4. **2D editor + 3D top-down render on top of each other** — the editor
   draws over the live 3D scene; the vehicle mesh occludes the canvas
   center (`1280x720_06_editor_serpentine.png`).
5. **Documented shortcuts don't exist** — `Ctrl+S`, `Ctrl+O`, `Delete`,
   `C` (close loop), `Tab` (cycle) from `AUTHORING.md` are absent from
   the input handler; Delete/Backspace do nothing.
6. **Recording silently overwrites `last_episode.json`** and silently
   drops frames past 5000; external-AI episodes can't be recorded at all.
7. **NEW discards all work; ESC quits** — no confirmation, no dirty flag.

Serious:

8. Hit-testing duplicates layout math (re-draw inside click handler) —
   the click path and render path can diverge; e.g., inspector width is
   hardcoded to 350 in *two* places.
9. Properties edit via +/- buttons only — no text entry, no sliders;
   enums cycle blindly; no reset; no tooltips; no units on most rows.
10. Inspector is fixed 350×620 — at 720p it leaves 50 px slack; at
    smaller windows it clips; content taller than the panel is
    unreachable (no scroll).
11. Left ~375 px of the editor canvas is an invisible dead zone; the
    track still renders under the panel.
12. Bottom hint bar always shows driving hints ("Drive: WASD…") even in
    the editor and replay modes.
13. "REWARD DECOMPOSITION" title collides with the Total value
    (`*_01_sim.png` top-right).
14. Wheel zoom doesn't anchor to cursor; pan is middle/right-drag only —
    undiscoverable.
15. Click-on-empty-canvas **inserts a control point** — accidental
    geometry creation is one stray click away.
16. No file dialogs anywhere (open/save/record destination/replay load).
17. No scrollbars, no text input, no menus, no dialogs, no tooltips —
    required primitives for this phase do not exist yet.
18. No theme system — every color is a hardcoded RGB literal scattered
    across `hud.py`/`editor.py`/`inspector.py`.

## 15. Accessibility / readability

- Fixed 13–14 px body text (Segoe UI/Consolas); no scaling.
- No keyboard focus model, no focus ring, no keyboard-operable panels.
- Status mostly uses icon+text (good) but several gizmos are color-only
  (curvature green/yellow/red has a text badge only on the selected point).
- Contrast: generally adequate on dark panels; grid lines at ~120 alpha
  are borderline over the bright green 3D ground.
- No tooltips for icon-like controls; abbreviations (OBS, TERM) are
  jargon-y.

## 16. Responsive / layout

- `VIDEORESIZE` is handled: GL viewport + UI surface/texture recreated.
- Everything else is absolute: panels don't collapse, don't reflow, don't
  clip content to themselves. At 1280×720 the inspector barely fits; at
  laptop sizes < 900 px wide the editor canvas is mostly dead space.
- No handling of display DPI scaling (pygame default = system-scaled;
  fonts are fixed px so they shrink on hi-DPI relative to screen).

## 17. Missing workflows

Home/library, track cards/thumbnails, new-track flow (blank/template/
duplicate), project metadata (description, modified time), multi-file
save management, recents, dataset browser, recording configuration +
destination picker, replay file loading, experiment↔environment linking,
validation→setting navigation, undo/redo, dirty tracking, autosave,
quit confirmation, keyboard shortcut map, editor view commands
(frame-all/selected).

## 18. Technical limitations affecting UX

1. **Immediate-mode UI**: no widget tree, no layout engine, no input
   focus, no scroll regions, no text editing. Any Phase-7 surface
   (library cards, dialogs, browsers) needs a minimal retained widget
   layer on top of `UIOverlayRenderer`.
2. Hit-testing via re-draw is O(panels) per click and fragile — needs a
   unified "collect interactive rects during the single render pass"
   model.
3. Native file dialogs don't exist in pygame — destination selection
   needs either `tkinter.filedialog` (stdlib, works on Windows, opens a
   native dialog on the same thread) or a custom in-app browser.
4. `EpisodeRecorder` is memory-only with a 5 000-frame cap — long
   recordings need chunked/flush-to-disk or a documented limit.
5. Colors are literals — a design-token pass must precede any theming.
6. The editor's 2D transform and the 3D camera are separate systems that
   must be unified or deliberately split (recommended: dedicated editor
   viewport that suppresses the 3D pass, or a properly-integrated
   orthographic editor camera).
7. The recorder hook lives inside the manual-drive branch — recording
   external-agent steps requires moving the hook outside that branch
   (small fix, architectural rule intact).
8. No dirty flag / modification tracking on `EnvironmentProject` — needed
   for save-state UX and unsaved-changes warnings.
