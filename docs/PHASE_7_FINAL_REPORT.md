---
noteId: "0f9b3a80bf5611f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — Final Hardening Report

Scope: finish the remaining UX gaps from `PHASE_7_STATUS.md`, verify the
real application end-to-end, and bring the studio to product quality.
Audit: `docs/PHASE_7_FINAL_QA_AUDIT.md` (evidence-backed findings only).

## Fixes implemented

### Theme & visual consistency
- **Inspector fully themed** — all 43 hardcoded RGB literals replaced with
  `theme.C` tokens. Previously rendered a dark illegible panel under the
  light theme; now coherent in both themes.
- **Editor canvas viz colors → tokens** — selection ring → `C.selection`,
  point/entity labels → `C.text*`, spawn → `vx_spawn`, curvature →
  `vx_curv_*`, outlines → `vx_boundary`. White labels were invisible on
  the light canvas; now readable in both themes.
- **Inspector action buttons** — `accent_soft` bg + white text was
  unreadable in light theme (white on pale blue); now `accent`.
- **HUD** intentionally keeps dark translucent panels — viewport overlays
  on the 3D scene need dark for contrast in *both* themes (documented
  choice, not a gap).

### Hit-testing
- **Fixed region dispatch off-by-one** — `begin_frame` snapshotted the
  *previous* frame's regions into `_dispatch_regions`, so for one frame
  after every render, clicks were tested against stale regions. Dispatch
  now uses the current frame's regions directly. Latent product bug
  exposed by the interaction harness.

### Microcopy & labels
- Inspector title `RL ENVIRONMENT DESIGNER` → `ENVIRONMENT`.
- Tabs: row 1 full words (`Overview Scene Track Point Entity`); row 2
  readable abbreviations (`Agent Obs Act Rwd Trm Scn Val Trn`) with
  descriptive tooltips on hover (`TAB_TOOLTIPS`).
- `RL Readiness Gate`/`Gate Status` → `Training Readiness`.
- `EXPORT TRAINING`/`LOAD BASIC`/`DELETE POINT`… → sentence-case actions
  (`Export Training`, `Load Basic`, `Delete Point`, `Validate Now`).
- Bottom row `NEW/SAVE/LOAD` → `New / Save / Open...`;
  `REBUILD 3D ENVIRONMENT` → `Rebuild 3D`.
- Status bar hides the `saved` chip on the DATA tab (meaningless there).

### Track library
- **Rename** — new `draw_rename` dialog + card menu item + app handler
  (`library.rename` already existed; UI was missing).
- **Open-document indicator** — the card for the currently loaded track
  shows `open` / `● unsaved`.

### Replay
- Picker rows now show **name · track · steps · date** (reads each
  episode dir's `manifest.json`).
- Player info chip shows **recording name · track · time/duration ·
  frame · speed**.
- 3D scene only renders behind REPLAY when a recording is loaded —
  the empty picker no longer floats over the live world.

### Recording
- Summary dialog gained next-step actions: **View Replay / Record Again /
  Show in Folder / Done**.
- `stop_recording` writes per-episode `manifest.json` — picked up by the
  picker and DATA browser.

### DATA tab
- Split list into **EPISODE RECORDINGS** and **TRAINING DATASETS** with
  human kind labels (`Recording`/`Dataset`, not `REC`/`DS`).
- Fixed misclassification: episode dirs (manifest `kind=episode_recording`)
  were listed as datasets.
- Added **View Replay** for recordings in the detail pane.

### Undo/redo coverage
- Snapshots upgraded from `(road, entities)` tuples to full
  `EnvironmentProject` documents — undo/redo now covers agent, reward,
  scenario and other inspector-only edits, not just canvas edits.
- `EditHistory` made value-agnostic (deep-copies any snapshot).

### Tooling
- `tools/ui_shots.py`: fixed BGR→RGB channel swap (all prior screenshots
  had swapped colors; product colors were correct all along), forced
  dark theme for the evidence set, added record-dialog / confirm-dialog /
  recording-active / light-editor / light-sim states.
- `tools/smoke_interactions.py`: **25 checks** — clicks drive real hit
  regions through `_handle_events`; covers point select/drag, deselect,
  wheel zoom, RMB pan, frame_all visibility, edge insert, Del, undo,
  entity place/drag, snap, save, navigation.
- `tools/tcp_recording_e2e.py`: **15 checks** — real SimGymEnv client
  over TCP drives 25 steps while recording → verifies `episode.json` +
  `manifest.json` on disk, frame contents, summary dialog, and that the
  saved file loads back into the replay player and applies poses.

## Test results
- Full suite: **392 passed** (baseline re-run after all changes below)
- `smoke_interactions.py`: **25/25**
- `tcp_recording_e2e.py`: **15/15**

## Screenshot validation
`docs/phase7_shots/` — captured at 1280×720, 1600×900, 1920×1080
(dark) + light-theme home/editor/sim at 1280×720:
home, library sections, new-track dialog, confirm dialog, editor
(oval + serpentine, all inspector tabs), simulate HUD, record dialog,
recording-active state, replay picker + player, DATA tab, settings,
light theme. All inspected for clipping/overlap/glyphs/hierarchy.

## Remaining limitations (honest)
- Keyboard navigation inside dropdown menus — not implemented
  (accessibility debt).
- Track-card hover preview — deferred (thumbnails already communicate).
- `ui_scale` applies to fonts but not widget metrics — functional,
  not pixel-tuned.
- HUD microcopy still uses terse field labels (`Center Dev:`,
  `Heading Err:`) — consistent with telemetry density, acceptable.
- `SPAWN (90°)` label can overlap its arrow at extreme zooms — cosmetic.

## Files changed this pass
`sim_ui/inspector.py` (theming + labels + microcopy),
`sim_ui/editor.py` (viz tokens + spawn label offset + theme import),
`sim_ui/app.py` (project snapshots, replay_path, rec actions, show_3d
gate, unused import cleanup),
`sim_ui/edit_history.py` (value-agnostic snapshots),
`sim_ui/editor_ui.py` (tab tooltips),
`sim_ui/widgets.py` (dispatch fix),
`sim_ui/screens/dialogs.py` (rename dialog + summary actions),
`sim_ui/screens/home_screen.py` (rename item, dirty/open badge),
`sim_ui/screens/workspace_screen.py` (replay metadata, exp/ds actions,
status bar),
`sim_ui/screens/datasets_panel.py` (grouping, kind fix, View Replay),
`tests/test_ui_widgets.py` (dispatch semantics),
`tools/ui_shots.py`, `tools/smoke_interactions.py`,
`tools/tcp_recording_e2e.py` (new),
`docs/PHASE_7_FINAL_QA_AUDIT.md`, `docs/PHASE_7_FINAL_REPORT.md`.
