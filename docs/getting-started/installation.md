# Installation

Install Simulation Studio (the Autonomous Systems Simulator) from source and verify it launches.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python | Developed and tested on **3.13.7**. There is no strict version pin — any recent Python 3 with the dependencies below should work. |
| OS | **Windows is the primary target** — the standalone build pipeline (`build/package_windows.py`) produces a Windows executable. Running from source works wherever the dependencies install. |
| GPU / drivers | The interactive studio requests an **OpenGL 3.3 Core** context via pygame + moderngl (`sim_ui/app.py`). Headless mode (`--headless`) needs no display or GL. |

## Clone

```bash
git clone https://github.com/Ziad-Abaza/Autonomous-Systems-Simulator.git
cd Autonomous-Systems-Simulator
```

## Install dependencies

> **There is no `requirements.txt`, `setup.py`, or `pyproject.toml` in the repository.**
> The package list below is derived from what the source actually imports.

```bash
pip install numpy pygame moderngl opencv-python gymnasium pyinstaller
```

| Package | Used by | Required? |
|---|---|---|
| `numpy` | Everywhere — physics, observations, wire protocol | Yes |
| `pygame` | Window, input, UI overlay (`sim_ui/`) | Yes, for the interactive studio |
| `moderngl` | OpenGL 3.3 renderer (`sim_ui/app.py`) | Yes, for the interactive studio |
| `opencv-python` (`cv2`) | Camera sensor rendering (`sim_core/sensors/camera_sensor.py`) | Yes |
| `gymnasium` | `SimGymEnv` adapter (`sim_client/gym_env.py`) | Recommended — falls back to legacy `gym` if absent |
| `pyinstaller` | Standalone build (`build/package_windows.py`) | Build only |

Optional extras:

```bash
pip install torch    # training: sim_experiment trainers, PPO/DQN/SAC baselines, agentRL
pip install pytest   # test suite
```

- **`torch`** is only needed for RL training: the experiment trainers (`sim_experiment/trainers/`), the baseline agents (`sim_client/agents/ppo_baseline.py`, `dqn_baseline.py`, `sac_baseline.py`), and `agentRL/`. Driving, recording, the TCP server, and the client SDK work without it.
- **`pytest`** is needed to run the test suite (see below).

## Verify the install

```bash
python main.py --help
```

Expected output — the argument list:

```text
usage: main.py [-h] [--headless] [--port PORT] [--num-envs NUM_ENVS]
               [--track TRACK] [--width WIDTH] [--height HEIGHT]
```

Then launch the studio:

```bash
python main.py
```

The Studio home screen should open. If presets are missing, first launch generates them into `presets/` automatically.

Run the test suite:

```bash
python -m pytest tests/ -q
```

Expected: **542 tests** pass (takes ~8 minutes).

## Optional: standalone executable

Instead of running from source, you can package a single Windows executable:

```bash
python build/package_windows.py
```

See [Standalone build](../development/standalone-build.md) for the full pipeline and output layout.

## Where data lives

| Path | Contents |
|---|---|
| `tracks/` | Your saved `*.sim.json` environments (writable library root) |
| `presets/` | Built-in preset environments (read-only in the library; regenerated only if the whole directory is missing) |
| `data/` | Default `data_root` — `recordings/`, `datasets/`, `studio_settings.json` |
| `experiments/` | Training experiments, runs, checkpoints |

## Troubleshooting

- **`SimGymEnv` construction fails** — the gym adapter connects to the simulator in `__init__`; start the sim first (`python main.py` or `python main.py --headless`).
- **Blank window / GL errors** — your GPU/driver must provide OpenGL 3.3 Core; remote or virtualized environments may not. Use `--headless` there.
- More fixes: [Troubleshooting](../troubleshooting/troubleshooting.md).

## See also

- [Quickstart](quickstart.md) — launch modes and a 60-second first drive
- [First simulation](first-simulation.md) — guided track → drive → record → replay
- [Standalone build](../development/standalone-build.md)
- [Compatibility notes](../reference/compatibility.md)
- [Troubleshooting](../troubleshooting/troubleshooting.md)
