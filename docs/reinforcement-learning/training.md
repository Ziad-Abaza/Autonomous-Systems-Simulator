# Training

How to train agents on Simulation Studio — from one-command experiment
runs to driving the environment yourself over TCP.

Four paths exist, in decreasing order of batteries-included:

| Path | What it is | When to use it |
|------|-----------|----------------|
| **Experiment platform** (`sim_experiment`) | Built-in PPO / SAC / DQN / BC trainers launched from the Studio TRAIN inspector tab or `python -m sim_experiment.cli` — manifests, run dirs, metrics, checkpoints, eval | You want tracked, reproducible runs with zero agent code |
| **External training over TCP** | Your own framework (Stable-Baselines3, CleanRL, raw PyTorch) steps the env through `SimulationClient` / `SimGymEnv` | You want full control of the algorithm — see [External Agents](../agents/external-agents.md) |
| **Behavior cloning** (`train-bc`) | Trains a `BCPolicy` (tanh MLP) on a `transitions_v1` dataset exported from a run | Imitation warm-starts |
| **agentRL library** | Early-stage continual-RL toolkit: obs/action adapters, reward & termination presets, track registry, mutators, replay buffers | Research scaffolding — see [below](#agentrl-library-experimental) |

## Experiment platform

The `sim_experiment` package owns the whole run lifecycle: an immutable
`ExperimentManifest` (fingerprinted from the env + scenario + seed +
training config), per-run directories under
`experiments/<experiment_id>/runs/<run_id>/` with `logs/`,
`checkpoints/`, `evaluation/`, `replays/`, `trajectories/`, and
`artifacts/`, a `metrics.jsonl` stream, and an orchestrator that spawns
trainers as `python -m sim_experiment.trainers.<name> --run-dir <dir>`
subprocesses.

### From the Studio — TRAIN inspector tab

Open a project, go to **EDIT**, and switch the inspector's RL tier to
**TRAIN**. The tab lists experiments and offers
create / launch / batch / cancel / resume / evaluate / reproduce /
export-dataset / export / compare actions plus a run monitor (status,
timesteps, episodes, curriculum stage, metrics, reward chart).

![TRAIN inspector tab showing experiment list and run monitor](../screenshots/inspector-training.png)

![TRAIN tab selected in the inspector tab strip](../screenshots/tab-training.png)

Studio-created experiments are opinionated: `trn_create` always builds a
**PPO** manifest with `total_timesteps=20000`,
`checkpoint_frequency=5000`, `eval_seeds=[0, 1]`, `num_episodes=3`, and
`random_seed=42`. For full control use the CLI.

### From the CLI — end-to-end worked example

All commands run from the repo root. `python -m sim_experiment.cli`
accepts a global `--root` (default `<repo>/experiments`).

```powershell
# 1. Gate the project for RL (validator report: errors, warnings)
python -m sim_experiment.cli validate-env --project presets/oval_circuit.sim.json

# 2. Create the experiment (writes experiments/<experiment_id>/)
python -m sim_experiment.cli create --project presets/oval_circuit.sim.json `
    --scenario basic_lane_following --name exp1 --algorithm ppo `
    --timesteps 50000 --num-envs 4 --eval-seeds 0,1 --seed 42

# 3. Find the experiment_id (exp_<16 hex> fingerprint)
python -m sim_experiment.cli list

# 4. Launch a run (trainer subprocess; --wait polls up to N seconds)
python -m sim_experiment.cli launch <experiment_id> --wait 60

# 5. Watch it
python -m sim_experiment.cli status <experiment_id> <run_id> --watch

# 6. List all runs for the experiment
python -m sim_experiment.cli runs <experiment_id>

# 7. Evaluate the latest registered checkpoint
python -m sim_experiment.cli evaluate <experiment_id> <run_id>

# 8. Export transitions_v1 dataset from the run's trajectories
python -m sim_experiment.cli dataset-export <experiment_id> <run_id> --dest out_ds

# 9. Behavior-clone from that dataset (a new BC run on the same manifest)
python -m sim_experiment.cli train-bc <experiment_id> --dataset out_ds
```

`launch` flags: `--trainer ppo|sac|dqn|bc|dummy` (defaults to the
manifest's algorithm), `--env-mode inprocess|process|tcp|tcp_multi`,
`--wait <seconds>` (default 0 = return immediately after STARTING).

### Environment modes (`env_mode`)

How the N training envs execute relative to the trainer process:

| `env_mode` | Mechanics | Notes |
|------------|-----------|-------|
| `inprocess` (default) | `SyncVectorEnv` over N `SimulationEnvironment` objects in the trainer process; seeds `base+i` | Fastest — no IPC; a crash kills the run |
| `process` | `ProcessVectorEnv` — one `spawn` `mp.Process` per env, pipelined pipe commands (`reset`, `step`, `set_scenario`, `ping`, `close`); `startup_timeout=120 s` | A dead worker raises `EnvWorkerCrash(worker_index)` — **retryable** by the batch scheduler |
| `tcp` | `HeadlessSimProcessPool` spawns N `main.py --headless --port P` processes on N ports; logs to `logs/sim_<port>.log` | Trainer talks NDJSON protocol 2.1; startup timeout 30 s/port |
| `tcp_multi` | One `main.py --headless --num-envs N --port P` process — one port, N env slots via `SimServerMulti` | Single shared process; env slots are FIFO on the wire |

`tcp`/`tcp_multi` runs embed `tcp{host, ports}` in the trainer's
`contract.json`; `process` adds process isolation without sockets.

## Training configuration

`TrainingConfig` (serialized in the manifest under `training`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `algorithm` | str | `"ppo"` | `ppo`, `sac`, `dqn`, or `custom:<name>` |
| `total_timesteps` | int | 50000 | Env steps budget (`--timesteps`) |
| `rollout_length` | int | 1024 | Steps per on-policy rollout segment (`--rollout`) |
| `batch_size` | int | 256 | Minibatch size (`--batch-size`) |
| `epochs` | int | 4 | Optimization epochs per update (`--epochs`) |
| `learning_rate` | float | 3e-4 | `--lr` |
| `discount_factor` | float | 0.99 | γ (`--gamma`) |
| `gae_lambda` | float | 0.95 | GAE λ (`--gae-lambda`) |
| `eval_frequency` | int | 5000 | Periodic-eval cadence in steps; **0 = off** (`--eval-freq`) |
| `checkpoint_frequency` | int | 10000 | Checkpoint cadence; **0 = off** (`--ckpt-freq`) |
| `logging_frequency` | int | 1 | Metric-write cadence |
| `num_envs` | int | 1 | Vector width (`--num-envs`) |
| `max_wall_seconds` | int | 0 | Wall-clock cap; **0 = unlimited** — expiry marks the run `INTERRUPTED` ("timeout") |
| `algorithm_config` | dict | `{}` | Per-algorithm hyperparameters (below) |

`EvaluationConfig` (`evaluation`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_seeds` | list[int] | `[0]` | Seeds cycled across eval episodes (`--eval-seeds 0,1`) |
| `num_episodes` | int | 5 | Episodes per evaluation (`--eval-episodes`) |
| `deterministic_policy` | bool | `True` | Greedy/Mean action selection during eval |
| `scenario_id` | str \| None | `None` | Override scenario for eval |

### `algorithm_config` keys

Pass extra hyperparameters as JSON:
`--alg-config '{"clip_coef":0.1,"ent_coef":0.0}'`.

**ppo** (`ppo_trainer`, continuous + discrete):

| Key | Default |
|-----|---------|
| `clip_coef` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `trajectory_episodes` | 0 (off; int = `first_n` trajectory-recording shorthand) |

**sac** (`sac_trainer`, continuous only):

| Key | Default |
|-----|---------|
| `buffer_size` | 100000 |
| `warmup_steps` | 1000 |
| `tau` | 0.005 |
| `alpha` | 0.2 |
| `auto_entropy_tuning` | on |
| `target_entropy` | −act_dim |
| `updates_per_step` | 1 |
| `target_update_interval` | 1 |
| `update_interval` | 500 |

**dqn** (`dqn_trainer`, discrete only):

| Key | Default |
|-----|---------|
| `buffer_size` | 50000 |
| `warmup_steps` | 1000 |
| `train_freq` | 4 |
| `target_update_interval` | 500 |
| `eps_start` / `eps_end` | 1.0 / 0.05 |
| `eps_decay_steps` | 10000 |
| `update_interval` | 500 |

**bc** (`bc_trainer`, continuous + discrete — driven by `train-bc`,
reads the dataset dir from the contract):

| Key | Default |
|-----|---------|
| `bc_epochs` | 20 |
| `hidden_sizes` | `(64, 64)` |
| `val_ratio` | 0.2 |
| `batch_size`, `bc_lr` | shared fields / trainer default |

PPO episode metrics include `reward`, `length`, `termination_reason`,
`mean_lateral_error`, `mean_speed`, `checkpoints_passed`; run metrics
include `sps`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`,
`explained_variance`, `episodes_completed`.

## Curriculum

A curriculum is an ordered `CurriculumStage` list serialized under the
`"curriculum"` key of the `.sim.json` project. Each stage:

| Field | Default | Description |
|-------|---------|-------------|
| `stage_id`, `name`, `description` | — | Identity/labels |
| `scenario_id` | — | One of the 6 standard scenarios |
| `target_metric` | `mean_return` | Also `lap_completion_rate`, `collision_rate`; aliases apply (`mean_return`→`mean_reward`) |
| `advancement_threshold` | 100.0 | Metric value needed to advance |
| `min_episodes` | 50 | Episodes required in-stage before advancing |
| `environment_overrides` | `{}` | Per-stage env tweaks; `target_speed`→`target_speed_override`, `time_limit`→`time_limit_override`, `surface_friction_mult`, `sensor_noise_mult`, `ambient_light`, `weather`, `time_of_day` |

Default 5-stage curriculum:

| # | Stage | Scenario | Threshold | Min eps |
|---|-------|----------|-----------|---------|
| S1 | Lane Keeping | `basic_lane_following` | mean_return ≥ 50 (target_speed 12) | 20 |
| S2 | High Speed | `high_speed_racing` | mean_return ≥ 120 (speed 22) | 30 |
| S3 | Obstacles | `obstacle_evasion` | lap_completion_rate ≥ 0.85 | 40 |
| S4 | Adverse | `wet_adverse_weather` | ≥ 0.80 | 40 |
| S5 | Full DR | `full_domain_randomization` | ≥ 0.90 | 50 |

At train time `CurriculumController`
(`sim_experiment/curriculum_runtime.py`, `CURRICULUM_STATE_VERSION="1.0"`)
— **not** `SimulationEnvironment` — applies the stage's scenario +
overrides and evaluates advancement once `episodes_in_stage ≥
min_episodes` and the threshold passes (lower-is-better for
`collision_rate`, `off_road_rate`, `timeout_rate`,
`mean_lateral_error`, `mean_heading_error`). Per-env stage seeds follow
`stage_seed = base_seed + env_index + stage_index × 10000`. State and
history persist to `runs/<run_id>/curriculum_state.json`.

```powershell
# Inspect a run's curriculum state
python -m sim_experiment.cli curriculum <experiment_id> <run_id>
```

## Checkpoints, artifacts, resume

- Trainers write `policy_*.pt` checkpoints at `checkpoint_frequency` and
  a final `policy_final.pt` into `runs/<run_id>/checkpoints/`.
- Every produced file is registered in
  `runs/<run_id>/artifacts/registry.jsonl` — `{kind:
  checkpoint|evaluation|replay|trajectory|other, path, created_at,
  step, metadata}` — so `evaluate` can default to the *latest registered
  checkpoint*.
- `python -m sim_experiment.cli resume <experiment_id> <run_id>` creates
  a **new run** whose `resume_from={parent_run_id, checkpoint}` points
  back at the parent — the original run is never mutated.
- Cancel with `python -m sim_experiment.cli cancel <experiment_id>
  <run_id>` (graceful terminate → kill after ~5 s).

## Batch and multi-seed runs

```powershell
# One job per seed (cross product with --scenarios if given)
python -m sim_experiment.cli batch <experiment_id> --seeds 1 2 3

# Actually run the batch — 2 workers, envs in spawned processes
python -m sim_experiment.cli batch-run <experiment_id> --seeds 1 2 3 `
    --workers 2 --env-mode process --max-retries 1 --timeout 600
```

`batch` creates `batches/<batch_id>/batch.json`; `batch-run` drives a
`BatchScheduler` (`max_workers`, heartbeats every 10 s, 30 s lease TTL)
and writes `batch_result.json`. Retries are type-gated by `RetryPolicy`:

- **Retryable**: `trainer_crash`, `trainer_exception`, `launch_failure`,
  `timeout`, `worker_crash`
- **Non-retryable**: `invalid_contract`, `invalid_curriculum`,
  `invalid_experiment`, `unsupported_algorithm`, `curriculum_mismatch`,
  `curriculum_state_missing`, `protocol_version_mismatch`,
  `launch_rejected`, `incompatible_environment`; unknown error types are
  not retried

Remote workers (`worker-serve`, `worker-status`, `worker-register`,
`worker-list`, `--worker host:port`) let a batch offload launches to
other machines over the NDJSON worker protocol 1.0 — see the
[CLI reference](../experiments/cli-reference.md).

## External training over TCP

Everything above is the built-in platform. To train with your own stack,
run the simulator headless (or interactively) and step it from any
process via `SimulationClient` / `SimGymEnv` — SB3, CleanRL, raw
PyTorch, whatever emits `[steer, throttle, brake]`:

```powershell
python main.py --headless --num-envs 4 --port 8765 --track oval
```

```python
import gymnasium as gym
from sim_client.gym_env import SimGymEnv   # gym.Env, 5-tuple API

env = SimGymEnv(host="127.0.0.1", port=8765)   # connects in __init__
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step([0.0, 0.6, 0.0])
```

The protocol is lockstep — one `STEP` message = one 60 Hz physics step —
and `SimGymEnv` exposes `Box`/`Discrete` spaces matching the project.
Full contract, message types, and included baseline agents:
[External Agents](../agents/external-agents.md) and
[TCP Protocol](../agents/tcp-protocol.md).

## agentRL library (experimental)

`agentRL` (`AGENT_VERSION="0.1.0"`) is a separate, **early-stage**
continual multi-track RL toolkit living alongside `sim_experiment`. The
stable surface below is numpy-only building blocks; since the initial
audit the package also gained torch-based `algos/` (PPO, SAC),
in-process `train/` trainers, an `eval/` matrix, `checkpoints/` IO, and
a config-driven runner (`python -m agentRL.experiments.matrix --exp
E001`) — all still experimental and **not** integrated into
`sim_experiment.cli`.

**Observation** — `ObservationSpec{channel_names, frame_stack=1,
prev_action=False, image=False}` names a subset/order of the env's
observation channels. Presets (`agentRL/obs/spec.py`):

| Preset | Channels | Flat dims |
|--------|----------|-----------|
| `state8` | speed, velocity_body(2), yaw_rate, steering_angle, distance_from_center, heading_error, distance_to_checkpoint | 8 |
| `full23` | `state8` + lidar_ranges | 23 |
| `lidar15` | lidar_ranges only | 15 |

`input_dim = vector_dim × frame_stack + (3 if prev_action)`;
`ObsEncoder` keeps a frame deque and can append the previous action.
`image=True` is reserved — the encoder raises `NotImplementedError`.

**Action** — `ActionAdapter(low, high, pos_only=())`
(`agentRL/act/adapter.py`): policies emit `[-1,1]`; `to_env` clips then
affinely maps to per-channel env bounds (steer [−1,1], throttle [0,1],
brake [0,1]), `from_env` inverts. `pos_only` channels (e.g. brake) map
raw `<0` to "released" — needed because env brake ≥ 0.1 dominates
throttle and would park a zero-mean policy forever. Tracks `prev_action`
for `prev_action=True` specs; `reset()` per episode.

**Reward preset `drive_v1`** — the anti-exploit preset. The stock racing
reward pays a *stationary* centered vehicle ≈ +0.8/step (centering 0.5 +
heading 0.3); `drive_v1` down-weights dense shaping and adds a heavy
time penalty so **idling nets ≈ −0.02/step** while 15 m/s earns
≈ +0.98/step:

| Component | Weight | Params |
|-----------|--------|--------|
| progress | 2.0 | `max_step_delta_m: 5.0` |
| speed | 0.5 | `target_speed_ms: 15.0` |
| centering | 0.05 | `max_distance_m: 6.0`, linear |
| heading | 0.05 | — |
| smooth_steer | −0.02 | — |
| checkpoint | 5.0 | — |
| completion | 100.0 | — |
| collision | −50.0 | — |
| off_road | −25.0 | — |
| reverse | −1.0 | `heading_threshold_deg: 100.0` |
| time_penalty | −0.12 | — |

**Termination preset `term_v1`** — terminates on collision, off_road,
wrong_direction (>120°), and course_completion (1 lap — success is a
termination here, unlike the stock config where it is disabled);
truncates on `max_steps: 1500` and `checkpoint_timeout` after 20 s.

**Tracks & envs**:

- `TrackRegistry.default()` — file tracks `oval`, `serpentine`, `smoke`
  → `tracks/{basic_driving_proving_ground,
  lane_following_serpentine_circuit, smoke_test}.sim.json` (tags `train`,
  `smoke`→`smoke`) plus generated loops `gen_loop_0..5` (parametric
  closed loops, seeds `10000+i`; `gen_loop_0..3` tagged `train`,
  `gen_loop_4..5` tagged `holdout` — guaranteed-unseen eval geometry).
- `EnvFactory(reward="drive_v1", termination="term_v1",
  obs_spec=PRESETS["full23"], sensor_names=None, max_duration_s=120)`
  clones the track project, injects the obs channel subset, presets, a
  minimal sensor suite, and the episode duration;
  `.build(track, seed)` → `SimulationEnvironment` via
  `sim_experiment.headless.build_env_from_dicts` (the canonical factory).
- `ScenarioMutator(seed, n_obstacles=(0,3), entity_types=(cone, barrier,
  obstacle), lateral_frac=(−0.4,0.4), s_range=(0.15,0.95),
  min_gap_m=15, spawn_jitter=(−1,1), spawn_yaw_jitter_deg=15,
  friction=(0.85,1.1), noise=(0.8,1.5))` — seeded per-episode
  `ScenarioDefinition` draws: obstacles sampled along the track's own
  centerline spline, spawn jitter, friction/sensor-noise multipliers.
  `draw_and_apply(env)` installs via `env.set_scenario`; deterministic
  per seed.

**Memory** — `ReplayBuffer(capacity, obs_dim, act_dim, seed)` stores
`done = terminated AND NOT truncated` (truncation bootstraps — horizon
cuts are not absorbing; the env's invalid step-after-done reply counts
as truncation too). `TrackRehearsalBuffer` keeps per-track buffers and
mixes `rehearsal_fraction` of each batch from previous tracks for
continual learning.

```python
from agentRL.envs.track_registry import TrackRegistry
from agentRL.envs.factory import EnvFactory

reg = TrackRegistry.default()
track = reg.load("oval")                 # file track
# track = reg.load("gen_loop_4")         # generated holdout
env = EnvFactory().build(track, seed=42) # SimulationEnvironment
obs, info = env.reset(seed=42)
```

> **Caveat:** `tracks/` is gitignored — the file tracks
> (`oval`/`serpentine`/`smoke`) only exist if the Studio has generated
> them; otherwise `TrackRegistry.default()` raises `KeyError` on load.
> Generated tracks need no files.

## Requirements & honest notes

- **torch is required** for all `sim_experiment` trainers (PPO/SAC/DQN/
  BC). The dev environment runs torch CPU — expect CPU-throughput
  training, not GPU.
- **Trainers are not in the standalone exe**: `build/package_windows.py`
  excludes torch and does not bundle `sim_experiment`, so the packaged
  app cannot launch training runs.
- `sim_client/agents/ppo_train.py` is only a **rollout demo** — it
  samples actions over `SimGymEnv` with no update code. Real PPO
  implementations live in `sim_client/ppo_baseline.py` (library) and
  `sim_experiment/trainers/ppo_trainer.py` (CLI-driven).
- `checkpoint_frequency=0` and `eval_frequency=0` disable those outputs;
  `max_wall_seconds=0` means unlimited wall time (expiry → `INTERRUPTED`
  "timeout", a retryable error).

## See also

- [RL Overview](rl-overview.md) — the env contract: obs, actions, reward, termination, seeds
- [Experiments](../experiments/experiments.md) — manifests, runs, metrics, artifacts, datasets
- [CLI Reference](../experiments/cli-reference.md) — every `sim_experiment.cli` subcommand
- [External Agents](../agents/external-agents.md) · [TCP Protocol](../agents/tcp-protocol.md)
- [Datasets](../datasets/datasets.md) — `transitions_v1` export for BC
- [Quickstart](../getting-started/quickstart.md) · [First Simulation](../getting-started/first-simulation.md) · [Simulating](../user-guide/simulation.md)
