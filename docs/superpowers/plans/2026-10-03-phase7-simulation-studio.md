---
noteId: "a932c2b0bf4911f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — Simulation Studio UX: Implementation Plan

> **For agentic workers:** implements `PHASE_7` brief. Steps use checkbox syntax for tracking.

**Goal:** Turn the existing simulation platform into a coherent studio:
home/track library → track workspace (editor/sim/replay/settings) →
recording & datasets → experiments, with a real widget layer, design
tokens, and a correctly-framed editor viewport.

**Architecture:** everything stays pygame+ModernGL. A minimal retained
UI layer (`sim_ui/widgets.py`) gives buttons/panels/scroll/text-input/
dialogs/menus with single-pass render+hit-test. A token file
(`sim_ui/theme.py`) owns color/type. A library service
(`sim_project/library.py`) treats `*.sim.json` as the document type.
The editor viewport gets a real canvas rect + bounds framing. All
existing services (env, recorder, dataset, experiments) are consumed —
no domain logic moves into UI.

**Spec sources:** Phase-7 brief, `docs/PHASE_7_UX_AUDIT.md`,
`docs/PHASE_7_UX_REFERENCES.md`.

## Global Constraints

- Do not touch physics/sensors/reward/termination/trainer/dataset schema.
- Editor visualizations never enter observations (observation-contract).
- Keep app runnable every commit; headless (`--headless`) and CLI flows
  unchanged.
- Pure logic lives in non-pygame-touching functions where possible so
  tests run headless (pygame surfaces work without a display; fonts need
  `pygame.font.init()` only).
- File dialogs via `tkinter.filedialog` (stdlib) — native on Windows,
  no new dependency.
- Recording destination configurable; persisted in
  `studio_settings.json` at repo root (gitignored).

## Review Focus

- Immediate-mode input: clicks must not leak through overlays/panels
  into the editor canvas (topmost-first dispatch).
- Frame-all math must handle: empty track, 1-point track, huge and tiny
  tracks, non-square viewport rects.
- Dirty tracking must catch all mutation paths (editor drags, inspector
  +/-, tool placement, template load).
- Recording hook must capture external-AI steps too, not just manual.
- Library scan must skip malformed `.sim.json` without crashing.

---

## Task 1 — UI primitives + theme tokens

**Files:**
- Create: `sim_ui/theme.py` (semantic color tokens, dark + light; fonts)
- Create: `sim_ui/widgets.py` (UIContext hit-registry, Button, Panel,
  TabBar-ish NavList, ScrollArea, TextInput, Menu/Dropdown, Modal,
  Toast/status message)
- Test: `tests/test_ui_widgets.py` (headless: registry dispatch order,
  scroll clamping, text input editing ops; pure geometry, no GL)

Key decisions:
- `UIContext` collects `(rect, action_id, z)` during the single render
  pass; `dispatch_click(pos)` picks topmost z, returns action_id —
  replaces re-draw-in-click-handler for new surfaces.
- Modal stack: topmost modal swallows all clicks beneath it.
- TextInput: focus, caret, backspace/delete/arrows/Home/End, select-all;
  committed on Enter/blur.
- Theme: `theme.py` exposes `C` namespace (bg, panel, border, text,
  text_dim, accent, warn, error, ok, rec, selection…) + `set_theme()`.

- [ ] write tests → implement → commit

## Task 2 — Track library service + thumbnails

**Files:**
- Create: `sim_project/library.py`
  `TrackLibrary(root)` → `scan() -> List[TrackAsset]`
  `TrackAsset{path,name,description,env_version,schema_version,
  entity_count,point_count,is_closed,length_m,modified,file_size,
  favorite,broken}`
  ops: `create_from_template`, `duplicate`, `rename`, `delete`,
  `set_favorite`, `touch`.
  Metadata extras (description/favorite) persist in a per-library
  `.library.json` sidecar — `.sim.json` stays untouched.
- Create: `sim_ui/thumbnails.py`
  `render_track_thumbnail(road_def, entities, size)->pygame.Surface`
  (dark bg, boundary fill, centerline, spawn arrow; cached PNG in
  `<root>/.thumbs/<hash>.png`)
- Test: `tests/test_track_library.py` (scan, create, duplicate, rename,
  delete, favorites, broken-file tolerance, thumbnail determinism)

- [ ] tests → implement → commit

## Task 3 — Studio shell: HOME + workspace navigation

**Files:**
- Create: `sim_ui/screens/home_screen.py` — Recent Tracks, Templates,
  New Track, Datasets, Experiments, Settings sections; card grid +
  list; search field.
- Create: `sim_ui/screens/workspace.py` — mode header (track name •
  dirty dot • path), nav: LIBRARY | EDIT | SIMULATE | SETTINGS |
  DATA | EXPERIMENTS.
- Modify: `sim_ui/app.py` — start on HOME; `_load_project(project,
  path)` unified loader (replaces 3 duplicated load paths); mode ↔
  screen routing; ESC = back one level, not quit; quit confirms dirty.
- Modify: `main.py` — `--track` behavior preserved (opens workspace
  directly, skips home) for CLI compat.
- Test: `tests/test_studio_shell.py` (navigation state machine pure
  parts: mode transitions, dirty-flag rules)

- [ ] tests → implement → commit

## Task 4 — Editor viewport rework + frame-all

**Files:**
- Modify: `sim_ui/editor.py` — canvas is an explicit `viewport_rect`
  (region between rails), transform centers on it; wheel zoom anchors
  at cursor; `frame_all(track_bounds)`, `frame_point(x,y)`,
  `frame_bounds(b)` with padding; grid follows viewport.
- Create: `sim_ui/track_bounds.py` (or in editor module) —
  `compute_track_bounds(road_def, entities)->Bounds` pure function.
- Modify: `sim_ui/app.py` — editor mode suppresses the 3D render pass
  and draws a dedicated editor background (kills double-visualization);
  3D stays for SIMULATE. Editor auto-frame-all on mode entry + on track
  load.
- Test: `tests/test_editor_view.py` (bounds→transform roundtrip,
  frame-all fits tiny/huge tracks at 4 viewport sizes, zoom anchor)

- [ ] tests → implement → commit

## Task 5 — Editor toolbar + outline + status bar + shortcuts

**Files:**
- Modify: `sim_ui/editor.py` + new `sim_ui/editor_ui.py` — left rail:
  tool list + outline tree (Control Points/Entities/Spawn/Gates);
  right rail: existing inspector (moved to right edge, height =
  viewport); top toolbar strip (Select/Draw/Place/Snap/Frame/View/
  Save); status bar (mode, tool, cursor world coords, zoom, dirty).
- Selection model stays in VisualTrackEditor; outline ⇄ canvas synced.
- Keys (editor mode): F frame-selected, A frame-all, Delete delete
  selected, Ctrl+S save, Ctrl+Z/Y undo/redo, G grid, S snap toggle,
  Esc deselect/ cancel tool.
- Tools become explicit: Select (no mutation) vs Draw Point (mutation)
  — kills stray-click geometry creation.
- Test: extend `test_editor_view.py` + pure dispatch tests.

- [ ] tests → implement → commit

## Task 6 — Save/Save-As/dirty + undo/redo + dialogs

**Files:**
- Modify: `sim_project/serializer.py` — `file_path` attr on project
  (non-serialized); `is_dirty` via fingerprint compare.
- Create: `sim_ui/edit_history.py` — snapshot-based undo stack for
  editor mutations (control_points+entities+spawn deep-copies; merge
  continuous drags).
- Modify: `sim_ui/app.py` — real Ctrl+S (current path or Save-As
  dialog), Save-As via `tkinter.filedialog.asksaveasfilename`,
  unsaved-changes confirm on quit/new/load (custom modal, not tkinter).
- Test: `tests/test_edit_history.py`, save/dirty unit tests.

- [ ] tests → implement → commit

## Task 7 — Inspector polish + validation navigation

**Files:**
- Modify: `sim_ui/inspector.py` — fix tab overflow (two-tier nav: group
  labels over icon-tabs OR collapsible section list — pick section list
  matching Godot pattern); per-tab validation badge; issue click →
  switch to owning tab; units on rows; section collapsing.
- Keep PropertyRow model and all property definitions unchanged.
- Test: `test_ui_services.py`-style provider tests where pure.

- [ ] implement → smoke → commit

## Task 8 — Recording workflow + destination + replay

**Files:**
- Create: `sim_project/settings.py` — `StudioSettings` (data_root,
  theme, recent_tracks) persisted to `studio_settings.json`.
- Modify: `sim_ui/app.py` — RECORD start opens Recording dialog
  (destination dir via tkinter `askdirectory`, episode name);
  writes `<dest>/<name>/episode.json` + manifest; stop → summary
  overlay; move `record_step` outside manual-only branch so external-AI
  steps record; bump frame cap to 50k with warning instead of silent drop.
- REPLAY mode: file picker lists `recordings/` + `*.json` episodes →
  `replay_player.load_recording` → per-frame vehicle pose application +
  scrub/play/pause/speed → vehicle drawn from frame pos/yaw.
- Test: `tests/test_recording_workflow.py` (dest validation, manifest,
  replay frame application pure parts).

- [ ] tests → implement → commit

## Task 9 — Dataset browser

**Files:**
- Create: `sim_ui/screens/datasets_screen.py` — scans
  `experiments/**/dataset_export` + `<data_root>/datasets`; rows:
  name, episodes, steps, size, format, source run/env; actions:
  inspect panel (uses `inspect_dataset`), open folder, delete
  (confirm modal), open-source-experiment jump.
- Test: `tests/test_dataset_browser.py` (scan pure function over a
  fixture dir tree).

- [ ] tests → implement → commit

## Task 10 — Experiments screen + context carry-over

**Files:**
- Create: `sim_ui/screens/experiments_screen.py` — experiment list
  (from `exp_mgr.list_experiments`), selected detail (manifest fields,
  runs, status), actions reusing `_train_action` backend; "New
  experiment from this track" carries current project context.
- Keep TRAIN tab backend (`_train_provider`/`_train_action`) — the new
  screen is a better front-end on the same services.

- [ ] implement → smoke → commit

## Task 11 — Theming pass + responsive + polish

- Tokenize new surfaces fully; migrate hud/editor literals where cheap.
- Panels collapse under ~1100 px width; inspector becomes overlay at
  small widths; status bar truncates gracefully.
- Fix reward-panel title collision; contextual bottom hints per mode.
- Screenshot pass at 1280×720 / 1600×900 / 1920×1080 / ~1366×768.

## Task 12 — Validation + docs + report

- Generalize `_audit_shots.py` → `tools/ui_shots.py` (kept as a real
  tool) for screenshot-driven validation.
- Docs: PHASE_7_OVERVIEW, TRACK_EDITOR, TRACK_LIBRARY,
  DATA_RECORDING_WORKFLOW, DATASET_BROWSER, EDITOR_CAMERA_AND_SCALE,
  PHASE_7_UI_VALIDATION, PHASE_7_FINAL_REPORT (status matrix).
- Full test suite run.
- Delete `_audit_shots.py` (superseded by tools/ui_shots.py).
