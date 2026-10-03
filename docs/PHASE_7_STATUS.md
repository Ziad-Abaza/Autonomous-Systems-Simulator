---
noteId: "0de004b0bf5011f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — Simulation Studio: Status & Verification

Verification legend:
- ✅ **VERIFIED** — exercised end-to-end with evidence (screenshot / interaction check / test).
- ⚠️ **PARTIAL** — implemented and code-verified; real interaction not fully exercised.
- ❌ **NOT DONE** — intentionally deferred or stubbed.
- 🚫 **N/A** — out of scope for this pass.

Evidence: `docs/phase7_shots/` (1280×720, 1600×900, 1920×1080),
`tools/ui_shots.py` (render capture), `tools/smoke_interactions.py`
(12/12 end-to-end checks: home → new track → create → edit → dirty →
undo → save → library).

## Studio shell

| Item | Status | Notes |
|------|--------|-------|
| Home screen (library-first boot) | ✅ | `python main.py` boots to Tracks grid; `--track` opens workspace |
| Workspace header (back, name+dirty, tabs, save, cam) | ✅ | Screenshots e1/s1/r2/d1 |
| Workspace tabs EDIT/SIMULATE/REPLAY/DATA | ✅ | Tab switching via real click dispatch |
| Unsaved-changes guards (quit, new) | ⚠️ | Dialogs render + routes wired; ESC/quit path exercised only in smoke test |
| Theme (dark/light) | ⚠️ | Tokens applied app-wide; light theme screenshotted on home only |
| UI scale | ⚠️ | `Fonts(scale)` + settings UI; non-1.0 scale not screenshot-verified |

## Track library & persistence

| Item | Status | Notes |
|------|--------|-------|
| TrackLibrary service (scan/save/delete/favorite/recents) | ✅ | `test_track_library.py` — 14 tests |
| Track cards + generated thumbnails | ✅ | h1 screenshot with real assets |
| New-track dialog (name + template) | ✅ | h6 + smoke test (create → file on disk → EDIT) |
| Save / Save-As / dirty dot | ✅ | Ctrl+S smoke-tested; Save-As uses native dialog (manual only) |
| Read-only bundled presets | ✅ | presets lib marked preset; delete guarded |

## Editor workspace

| Item | Status | Notes |
|------|--------|-------|
| Canvas-rect world↔screen transforms | ✅ | Serpentine now fully framed (was clipped) |
| frame_all / frame_selected (A/F) | ✅ | Auto-fit on open; toolbar buttons |
| Dedicated 2D canvas (no 3D underlay) | ✅ | 3D only renders on SIMULATE/REPLAY |
| Toolbar (tools, place menu, snap/grid, view, zoom) | ✅ | e1/e3 screenshots |
| Outline rail w/ selection sync | ✅ | Points/entities/spawn/gates listed + click-select |
| Status bar (cursor coords, zoom, dirty, hints) | ✅ | |
| Undo/redo (Ctrl+Z / Ctrl+Y) | ✅ | Snapshot history; smoke-tested; `test_edit_history.py` |
| Inspector docked right + tab shortcuts | ✅ | Abbreviated labels — collision fixed |
| Canvas drag/pan/zoom/insert interactions | ⚠️ | Code paths rewritten to canvas coords; click verified, drag paths need manual QA |
| Undo coverage for agent/reward edits | ⚠️ | Snapshots cover road_def + entities only |

## Simulation / recording / replay

| Item | Status | Notes |
|------|--------|-------|
| SIMULATE HUD (telemetry/reward/PiP) | ✅ | s1/s2; reward title overlap fixed |
| Recording dialog (name + destination) | ⚠️ | Dialog + flow coded; start/stop paths smoke-verified partially |
| Recording writes per-episode dir + manifest | ✅ | `stop_recording` writes `<dir>/<name>/episode.json` + `manifest.json` |
| External-AI step recording | ⚠️ | `env.last_step_info` exposes step data; exercised only via manual steps |
| Replay picker (recordings list + browse) | ✅ | r1 |
| Replay player (scrub, step, speed, pose apply) | ✅ | r2 shows recorded pose in chase cam |

## Data & experiments

| Item | Status | Notes |
|------|--------|-------|
| DATA tab browser (recordings + datasets unified) | ✅ | d1; detail panel + folder reveal + delete |
| Home datasets/experiments sections | ⚠️ | Render verified; row actions wired but no real datasets/experiments present to exercise |
| Open-experiment-env | ⚠️ | `open_experiment_env` implemented; untested with real experiment |

## Known gaps / honest deferrals

- **Drag interactions** (point drag, edge insert, spawn drag) rewritten to
  canvas coordinates but only click-level smoke-verified — needs manual QA.
- **Hover preview** on track cards — not implemented.
- **Undo** does not cover agent/scenario/inspector-only changes (road +
  entities only).
- **Batch/training panel** unchanged inside inspector TRAIN tab (existing
  Phase 6 UI still works behind its own tab).
- `_audit_shots.py` replaced by `tools/ui_shots.py`; removed.
- `hud.draw_top_bar` / `draw_bottom_bar` retained but no longer called.
