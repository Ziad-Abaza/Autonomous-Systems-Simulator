# Your First Simulation

A guided pass through the whole studio loop: **create a track → configure →
drive → record → replay**. About ten minutes.

## 1. Create a track

`python main.py` → home screen → **+ New Track**. Give it a name and pick a
template:

![New track dialog](../screenshots/new-track-dialog.png)

| Template | Produces |
|---|---|
| `empty` | Open 100 m straight — blank sandbox |
| `basic_driving` | Closed oval, 16 checkpoints (default) |
| `straight_sprint` | Open 300 m sprint, finish gate |
| `hairpin` | Closed circuit with a hairpin apex |
| `slalom` | Open 200 m slalom, 5 cone gates |
| `lane_following` | Serpentine + centering-tuned reward |
| `obstacle_avoidance` | Oval + cones/barrier, collision-weighted reward |

The workspace opens on the **EDIT** tab — a 2D authoring canvas over your
spline:

![Track editor](../screenshots/track-editor.png)

Drag a control point, or press `P` for the draw tool and click to add
points. `Ctrl+S` saves. Everything you change lands in one `.sim.json`
document. Details: [Track Editor](../track-editor/track-editor.md).

## 2. Check the environment

The right-side **Environment Inspector** shows the whole RL contract.
Flip through **AGENT → SENSORS → OBS → ACT → RWD → TRM → SCN**, then open
**VAL** — the validator should report readiness `PASS` (the bundled
templates are all RL-valid). If it fails, click an issue row to jump to the
tab that owns it. See [Environment Inspector](../user-guide/environment-inspector.md).

## 3. Drive it

Switch to **SIMULATE**:

![Live simulation](../screenshots/simulation.png)

- `W`/`↑` throttle · `S`/`↓`/`Space` brake · `A`/`D` or `←`/`→` steer
- `R` reset · `C` camera (chase → hood → top-down → orbit)
- `Tab` — observation inspector: what the AI actually sees vs ground truth

Left panel: live telemetry. Right panel: the reward decomposition — every
term of the reward function scored per step. The status bar shows
`AI listening on :8765` — an external agent could connect and drive this
very window right now.

> Interactive driving never terminates — step/checkpoint timeouts are
> disabled in the studio so you can explore freely. `R` resets manually.

## 4. Record an episode

Click **● Record** in the status bar → name the episode → confirm:

![Record dialog](../screenshots/record-dialog.png)

Drive for a while (every physics step is captured: pose, action, reward,
reward breakdown, collisions). Press **■ Stop** — a summary dialog shows
steps, duration, return, and termination reason.

## 5. Replay it

Switch to **REPLAY** → pick your recording from the list:

![Replay player](../screenshots/replay.png)

Scrub the timeline, `Space` to play/pause, `←`/`→` to step frame by frame,
`0.5–4×` playback speed, `C` still cycles the camera. The same replay
format is produced by training evaluations, so you can watch what a policy
actually did.

## Where next

- [Simulation tab reference](../user-guide/simulation.md)
- [Recording & Replay](../recording-replay/recording-and-replay.md)
- [Connect an external agent](../agents/external-agents.md) — or let the AI drive
- [Train a policy](../reinforcement-learning/training.md)
