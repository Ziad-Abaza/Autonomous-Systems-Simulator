# Keyboard & Mouse Reference

All bindings verified against `sim_ui/editor_ui.py` and `sim_ui/app.py`.

## Global / workspace

| Key | Context | Action |
|---|---|---|
| `Ctrl+S` | workspace | Save project |
| `Ctrl+O` | workspace | Open project |
| `Enter` | dialogs | Confirm (`new_track`) |
| `Esc` | everywhere | Close dialog → cancel tool → deselect → home → quit |

## EDIT tab — track editor

| Key | Action |
|---|---|
| `Ctrl+S` | Save |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `A` | Fit all |
| `F` | Fit selection |
| `Del` / `Backspace` | Delete selection |
| `G` | Toggle grid |
| `S` | Toggle snap (1 m) |
| `V` | Select tool |
| `P` | Draw-point tool |
| `Esc` | Cancel tool → exit draw → deselect |

### Mouse — editor

| Input | Action |
|---|---|
| LMB drag point/entity | Move (snap-applied) |
| LMB (draw tool, empty) | Append control point |
| LMB within 10 px of segment | Insert interpolated control point |
| LMB first CP, open track (≥3 pts) | Close the loop |
| RMB on point/entity | Delete (min 3 points kept) |
| RMB-drag / MMB-drag | Pan |
| Mouse wheel | Zoom ×1.15, cursor-anchored (0.05–60 px/m) |
| Spawn tool LMB | Place spawn marker |

## SIMULATE tab

| Key | Action |
|---|---|
| `W` / `↑` | Throttle |
| `S` / `↓` / `Space` | Brake |
| `A` / `D` or `←` / `→` | Steer |
| `R` | Reset |
| `C` | Cycle camera: CHASE → HOOD → TOP_DOWN → ORBIT |
| `Tab` | Observation inspector overlay |

## REPLAY tab

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Step backward / forward |

## See also

- [Simulation tab](simulation.md) · [Track Editor](../track-editor/track-editor.md)
