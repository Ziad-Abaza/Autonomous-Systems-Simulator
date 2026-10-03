---
noteId: "42ac4f50bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Headless Environments

**Files:** `sim_experiment/headless.py`, `main.py --headless`

The simulator runs fully without rendering/UI — required for CI, CLI
training, and parallel workers.

## `main.py --headless`

- Skips `_init_graphics()` entirely — no pygame display, camera, or fonts.
- Starts the TCP server (`--port`, default 5555) and runs a fixed 60 Hz
  tick loop stepping whenever external control is active.
- `SimulationApplication` works in headless mode: recorder, server,
  inspector, and the experiment services (`exp_mgr`, `orch`,
  `_train_provider`) are all constructed — only drawing is skipped.

## In-process headless pool

```python
from sim_experiment.headless import build_env_from_dicts, HeadlessEnvPool

env = build_env_from_dicts(env_dict, scenario_dict, seed=42)  # single
pool = HeadlessEnvPool(env_dict, scenario_dict,
                       num_envs=4, base_seed=1000)             # N independent envs
```

- Each env gets `seed + index` — independent RNG, episodes, and agent
  state. No shared mutable state.
- `env.step()` is pure CPU + sensors; no render dependencies.

## Subprocess pool (OS-level isolation)

```python
from sim_experiment.headless import HeadlessSimProcessPool
pool = HeadlessSimProcessPool(num_envs=4)
ports = pool.start()   # spawns main.py --headless --port N per env
# ... trainers connect via SimGymEnv(127.0.0.1, port) ...
pool.stop()
```

Used when `env_mode="tcp"` — real process isolation and TCP boundary.
