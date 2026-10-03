---
noteId: "63b31420bf5211f1a29f1fbaabbd87c8"
tags: []

---

# Phase 7 — Final QA Audit

Evidence basis: code review of `sim_ui/` + `sim_project/`, screenshots in
`docs/phase7_shots/` (1280×720, 1600×900, 1920×1080, dark + light),
`tools/smoke_interactions.py` (12/12), `tools/ui_shots.py` render captures.
No findings are speculative — each cites a file or screenshot.

## A. Verified-complete (no action needed)

- Studio shell: home screen, workspace header, EDIT/SIMULATE/REPLAY/DATA tabs.
- Editor canvas transforms: `canvas_rect`-based world↔screen; serpentine
  auto-frames correctly at all three resolutions (e3 shots).
- Theme tokens (`sim_ui/theme.py`): complete semantic set, two intentional
  palettes. All *new* screens consume tokens.
- Widget layer: hit registry, tooltips, menus with screen-clamping,
  scroll regions, text inputs with caret.
- Track library: cards + cached thumbnails + context menu + delete confirm.
- New-track dialog, recording dialog, recording summary, confirm dialogs:
  consistent pattern, named actions (no bare "OK").
- Save/dirty/undo-redo for road + entities.
- Replay loads recordings and applies poses; transport bar works.
- Recording writes per-episode dir + manifest.
- 392 tests pass; multi-res shots verified.

## B. Confirmed defects (fix this pass)

| # | Finding | Evidence | Severity |
|---|---------|----------|----------|
| 1 | **Inspector ignores the theme system** — 43 hardcoded RGB literals; renders dark panel under light theme | `sim_ui/inspector.py` (draw path), `1280x720_l2_edit_light.png` | High — illegible text in light theme |
| 2 | **Inspector title is jargon** — "RL ENVIRONMENT DESIGNER"; section headers use `--- X ---` ASCII rules | `inspector.py` draw, e1 shot | Medium |
| 3 | **Tab labels unexplained** — `AGT OBS ACT RWD TRM SCN VAL TRN` with no tooltips | `inspector.py` `TAB_LABELS` | Medium |
| 4 | **No rename action** for tracks | `home_screen.py` menu items | Medium |
| 5 | **No dirty indicator on the open track's card** | `home_screen.py` TrackCard | Low |
| 6 | **Replay picker shows bare filenames** — no track name, steps, duration, date | `workspace_screen.py` `_draw_replay_picker` | Medium |
| 7 | **Replay player info chip minimal** — only frame idx + speed | `_draw_replay_player` | Medium |
| 8 | **Recording summary lacks next-step actions** — only "Open Folder"/"Close"; spec asks View Replay / Record Again | `dialogs.py` `draw_recording_summary` | Medium |
| 9 | **Undo does not cover agent/reward/scenario inspector edits** — snapshots are `(road_def, entities)` only | `app.py` `_editor_snapshot` | Medium |
| 10 | **DATA tab status bar shows "saved"** — meaningless there | `_draw_statusbar` | Low |
| 11 | **Editor drag interactions unverified** — code-verified only | needs scripted QA | Medium |
| 12 | **TCP external-step recording unverified E2E** | needs scripted QA | Medium |
| 13 | **Microcopy inconsistencies** — `EXPORT TRAINING`, `LOAD BASIC`, `Gate Status`, `NEW/SAVE/LOAD` button row | `inspector.py` props | Low |
| 14 | **ui_shots channel order was wrong** — BGR read as RGB; all prior screenshots had swapped colors | `tools/ui_shots.py` (fixed) | Tooling only — product colors were correct |

## C. Theme & readability findings

- Inspector, editor internals, HUD use literal colors instead of tokens.
  HUD panels overlay a 3D viewport — intentional dark translucency,
  documented as viewport-overlays (not a bug); inspector/editor are
  document surfaces and must follow the theme (bug).
- `SPAWN (90°)` label and selection colors are viewport viz tokens — fine.
- Editor grid/boundary colors already on `vx_*` tokens — fine.
- Confirm-dialog danger styling is correct (red) — earlier misread was
  the BGR capture bug, not the product.

## D. Interaction-risk areas (need scripted/manual QA)

- Point drag, edge-insert, spawn drag, entity drag, RMB pan, wheel zoom:
  rewritten to canvas coords; click-path verified but drag paths not
  exercised against the real event loop.
- `menu_draw` z-ordering vs canvas clicks — menus register at Z_MODAL;
  verified implicitly by smoke clicks but worth a deliberate case.
- `ui_scale` ≠ 1.0 font rebuild — implemented, never rendered.

## E. Deliberate deferrals (document, don't fix this pass)

- Track-card hover *preview* (larger thumbnail on hover) — low value vs
  cost; thumbnails already communicate.
- Keyboard navigation inside menus — documented as accessibility debt.
- Undo coverage extension to agent/scenario edits — now planned as a
  project-level snapshot restore (fix in this pass, see §B-9).
- 3D HUD panels keep dark translucent styling in both themes by design.
