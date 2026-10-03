# FAQ

## General

**What does the simulator provide vs what do I write?**
The simulator is strictly the environment: world, physics, sensors,
observations, actions, rewards, termination, scenarios, recording,
evaluation. Your agent — the policy — stays outside and connects over TCP
or the gymnasium adapter.

**What Python version?**
Developed and tested on **3.13.7**. Requires a modern Python with
`from __future__ import annotations` support (3.10+ recommended).

**Is there a requirements.txt?**
No — install: `pip install numpy pygame moderngl opencv-python gymnasium`
(plus `torch` for training, `pytest` for tests, `pyinstaller` for the
build). See [Installation](../getting-started/installation.md).

**Is there a license?**
No `LICENSE` file exists in the repository — all rights reserved unless
the author adds one.

## Environment

**How do I connect my own AI?**
`python main.py` (studio) or `python main.py --headless` — then
`SimGymEnv(host, port)` or `SimulationClient`. Any language works via the
NDJSON TCP protocol. See [External Agents](../agents/external-agents.md).

**What does the agent observe by default?**
Flat `float32` vector of 23 values — speed, body velocities, yaw rate,
steering, lateral offset, heading error, checkpoint distance, 15 LiDAR
ranges. Optional per-channel dict or RGB camera image.

**What's the action space?**
`Box([-1,0,0],[1,1,1])` = `[steer, throttle, brake]` continuous by
default, or 5 discrete options when configured.

**Can multiple agents connect?**
Single-env mode: one client. `--headless --num-envs N`: N env slots on
one port, assigned FIFO — connect N clients.

**Can the env run on its own?**
No — it's lockstep. The environment advances exactly once per `STEP`
message; there is no free-running tick.

**How do I change the scenario mid-training?**
`client.set_scenario("wet_adverse_weather", reset=True)` (protocol ≥ 2.1)
or the experiment curriculum system. See
[TCP Protocol](../agents/tcp-protocol.md) and
[Training](../reinforcement-learning/training.md#curriculum).

**Which features are declared but not functional?**
`reverse_max_speed`, per-point `friction`/`banking`, `spawn_mode
"random_cp"`, `episode_config.auto_reset_on_done`/`reset_behavior`,
`include_position`/`include_laps`/`include_progress` obs flags (always-on
fields or unimplemented), `normalization "standardized"`,
`TerminationConditionType.CUSTOM_THRESHOLD`. See
[Compatibility](compatibility.md).

## Training & experiments

**Which trainers are included?**
PPO, SAC, DQN (torch, CPU) + behavioral cloning (BC). A dummy trainer
exists for smoke tests. The `ppo_train.py` under `sim_client` is a
rollout demo, not a trainer.

**How do I run an experiment?**
`python -m sim_experiment.cli create --project <.sim.json> --algorithm
ppo --timesteps 50000 --seed 42` then `launch <id>`. See
[Experiments](../experiments/experiments.md).

**Are experiments reproducible?**
Manifest + environment fingerprint + seed → same environment every time;
`reproduce` re-verifies fingerprints. Training itself is not guaranteed
bit-identical across machines (torch).

**Can I export data for offline RL / imitation learning?**
Yes — `dataset-export` writes `transitions_v1` (episodes + manifest);
`train-bc` consumes it. Validation: `dataset-validate`. See
[Datasets](../datasets/datasets.md).

## Studio & files

**Where do tracks live?**
`tracks/` — your library (gitignored), `presets/` — bundled read-only.
Each track is a `.sim.json`.

**Where do recordings/datasets go?**
`<data_root>/recordings/`, `<data_root>/datasets/` (default `data/`),
plus `experiments/` for run artifacts.

**Are old project files supported?**
Yes — schema 1.x documents are migrated on load. Unknown keys are
ignored (tolerant loading).

## Troubleshooting quick hits

| Symptom | First check |
|---|---|
| Agent can't connect | Server running? `python main.py --headless --port 8765` |
| `SimGymEnv` fails at construction | It connects in `__init__` — no server = immediate error |
| Black/empty camera image | `include_camera_rgb` or camera `observable` enabled? |
| `SET_SCENARIO` rejected | Protocol < 2.1, or mid-episode without `reset:true` |
| Pygame pkg_resources warning | Harmless — known pygame deprecation warning |
| Bundle won't run training | Trainers excluded from PyInstaller build — run from source |

More: [Troubleshooting](../troubleshooting/troubleshooting.md).

## See also

- [Glossary](glossary.md) · [Compatibility](compatibility.md)
- [Getting Started](../getting-started/quickstart.md)
