# Phase 7 — Simulation Studio: Status & Verification

Verification legend:
- ✅ **VERIFIED** — exercised end-to-end with evidence (screenshot /
  interaction check / test).
- ⚠️ **PARTIAL** — implemented and code-verified; real interaction not
  fully exercised.
- ❌ **NOT DONE** — intentionally deferred or stubbed.
- 🚫 **N/A** — out of scope for this pass.

Evidence: `docs/phase7_shots/` (1280×720, 1600×900, 1920×1080, dark +
light), `tools/ui_shots.py` (render capture),
`tools/smoke_interactions.py` (**25/25** real click/drag checks),
`tools/tcp_recording_e2e.py` (**15/15** TCP recording checks),
`docs/PHASE_7_FINAL_QA_AUDIT.md`, `docs/PHASE_7_FINAL_REPORT.md`.

## Studio shell

| Item | Status | Notes |
|------|--------|-------|
| Home screen (library-first boot) | ✅ | `python main.py` boots to Tracks grid |
| Workspace header + tabs | ✅ | EDIT/SIMULATE/REPLAY/DATA verified by clicks |
| Unsaved-changes guards | ⚠️ | Dialogs + routes wired; quit path exercised in smoke |
| Theme (dark/light) | ✅ | Both themes screenshotted on home, editor, sim |
| UI scale | ⚠️ | Fonts scale; widget metrics fixed |

## Track library & persistence

| Item | Status | Notes |
|------|--------|-------|
| TrackLibrary service | ✅ | `test_track_library.py` |
| Cards + thumbnails + menu | ✅ | Open/Dup/Rename/Fav/Reveal/Delete |
| Rename dialog | ✅ | `draw_rename` + handler |
| Dirty indicator on open card | ✅ | `open` / `● unsaved` chip |
| New-track dialog | ✅ | Smoke-tested end-to-end |
| Save / dirty / recents | ✅ | Ctrl+S smoke-tested |

## Editor workspace

| Item | Status | Notes |
|------|--------|-------|
| Canvas transforms + fit-all | ✅ | All resolutions; serpentine framed |
| Toolbar + outline rail + status bar | ✅ | |
| Inspector tabs + tooltips + badges | ✅ | Themed; readable labels |
| Undo/redo | ✅ | Full project snapshot — covers road, entities, agent, reward, scenario |
| Drag interactions | ✅ | smoke: point drag, entity drag, edge insert, RMB pan, wheel zoom, snap |
| Hit dispatch | ✅ | off-by-one stale-region bug found & fixed |

## Simulation / recording / replay

| Item | Status | Notes |
|------|--------|-------|
| SIMULATE HUD | ✅ | Reward/telemetry panels legible |
| Record dialog + summary | ✅ | Destination + name + View Replay/Record Again actions |
| Manual-step recording | ✅ | via `_record_step_if_needed` |
| External TCP-step recording | ✅ | 15/15 — real client, frames, manifest, replay load |
| Replay picker + player | ✅ | Metadata rows; name/track/time/duration chip |
| View Replay from DATA tab | ✅ | |

## Data & experiments

| Item | Status | Notes |
|------|--------|-------|
| DATA tab: grouped recordings/datasets | ✅ | Fixed episode-recording misclassification |
| Detail pane + folder reveal + delete | ✅ | |
| Home datasets/experiments sections | ⚠️ | Render + row actions wired |

## Remaining limitations (documented, not silently worked around)

- Keyboard navigation inside menus — accessibility debt.
- Track-card hover preview — deferred.
- HUD keeps dark translucent panels in both themes by design.
- `ui_scale` resizes fonts only, not all widget metrics.
- HUD field labels stay terse (`Center Dev:`) — telemetry convention.
