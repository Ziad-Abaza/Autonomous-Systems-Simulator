# CLI Reference

Complete command-line reference for Simulation Studio: the experiment platform
(`python -m sim_experiment.cli`), the simulator launcher (`python main.py`), and
the bundled tools/agent entry points.

## `python -m sim_experiment.cli`

Training & experiment platform CLI — a thin wrapper over the `sim_experiment`
domain services (manifests, runs, metrics, datasets, batches, remote workers).

### Global flag

| Flag | Type | Default | Required |
|---|---|---|---|
| `--root` | path | `<repo>/experiments` | no |

All experiment state lives under `--root` as
`experiments/<experiment_id>/{experiment.json, environment.json, scenario.json, runs/}`.

### Experiments

#### `validate-env`

Validate a `.sim.json` project file for RL use (exit 0 if valid, 1 otherwise).

| Flag | Type | Default | Required |
|---|---|---|---|
| `--project` | path | — | yes |

```bash
python -m sim_experiment.cli validate-env --project presets/oval_circuit.sim.json
```

#### `create`

Create an experiment manifest from a project + scenario. Fails if the
environment fails validation unless `--force` is given.

| Flag | Type | Default | Required |
|---|---|---|---|
| `--project` | path | — | yes |
| `--scenario` | str | `basic_lane_following` | no |
| `--name` | str | `experiment` | no |
| `--algorithm` | str (`ppo`\|`sac`\|`dqn`\|`custom:<name>`) | `ppo` | no |
| `--timesteps` | int | `50000` | no |
| `--rollout` | int | `1024` | no |
| `--batch-size` | int | `256` | no |
| `--epochs` | int | `4` | no |
| `--lr` | float | `3e-4` | no |
| `--gamma` | float | `0.99` | no |
| `--gae-lambda` | float | `0.95` | no |
| `--eval-freq` | int | `0` (off) | no |
| `--ckpt-freq` | int | `10000` (0 = off) | no |
| `--num-envs` | int | `1` | no |
| `--max-wall-seconds` | float | `0` (unlimited) | no |
| `--seed` | int | `42` | no |
| `--eval-seeds` | csv ints | `""` (falls back to `[--seed]`) | no |
| `--eval-episodes` | int | `5` | no |
| `--alg-config` | JSON str | `""` | no |
| `--force` | flag | off | no |

```bash
python -m sim_experiment.cli create --project presets/oval_circuit.sim.json \
    --scenario basic_lane_following --name exp1 --algorithm ppo \
    --timesteps 50000 --seed 42
```

#### `list`

List all experiments under `--root` (skips legacy `experiment.json` files
without `manifest_version`).

| Flag | Type | Default | Required |
|---|---|---|---|
| — | — | — | — |

```bash
python -m sim_experiment.cli list
```

#### `show`

Print the full experiment manifest as JSON.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli show exp_556fe0488d4db907
```

#### `launch`

Launch a training run for an experiment. Marks the manifest launched (manifests
are immutable post-launch). Status flow: `CREATED → QUEUED → STARTING → RUNNING`.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--trainer` | str | `ppo` | no |
| `--env-mode` | `inprocess`\|`process`\|`tcp`\|`tcp_multi` | `inprocess` | no |
| `--wait` | float (seconds) | `0` (don't wait) | no |

```bash
python -m sim_experiment.cli launch exp_556fe0488d4db907 --trainer ppo --wait 600
```

#### `archive`

Archive an experiment directory.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli archive exp_556fe0488d4db907
```

#### `export`

Copy the entire experiment directory to a destination.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--dest` | path | — | yes |

```bash
python -m sim_experiment.cli export exp_556fe0488d4db907 --dest exported/exp_556fe0488d4db907
```

#### `reproduce`

Check whether an experiment is reproducible (manifest version, environment
fingerprint, scenario, schemas, simulator/protocol versions). Exit 0 if
reproducible, 1 otherwise. Note: exact reproduction is never guaranteed.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli reproduce exp_556fe0488d4db907
```

### Runs

#### `runs`

List all runs of an experiment.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli runs exp_556fe0488d4db907
```

#### `status`

Print a one-line status summary (status, timestep, episode count, latest
metrics). `--watch` polls every 2 s until the run reaches a terminal state
(`COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`).

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |
| `--watch` | flag | off | no |

```bash
python -m sim_experiment.cli status exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4 --watch
```

#### `cancel`

Cancel a running run (graceful terminate, then kill after a 5 s grace period).

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli cancel exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4
```

#### `resume`

Resume from the latest registered checkpoint of a run. Creates a **new** run
with `resume_from={parent_run_id, checkpoint}` — it does not restart the old run.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |
| `--trainer` | str | `ppo` | no |

```bash
python -m sim_experiment.cli resume exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4
```

#### `evaluate`

Evaluate a checkpoint against the manifest's evaluation config (seeds, episodes,
deterministic policy). Defaults to the latest registered checkpoint; writes the
result to the run's `evaluation/` dir and registers it as an artifact.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |
| `--checkpoint` | path | latest registered checkpoint | no |

```bash
python -m sim_experiment.cli evaluate exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4 \
    --checkpoint checkpoints/policy_final.pt
```

#### `trajectories`

List recorded trajectory episodes (`runs/<run_id>/trajectories/*.jsonl`) with
step count, return, and termination reason.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |
| `--json` | flag | off | no |

```bash
python -m sim_experiment.cli trajectories exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4 --json
```

#### `curriculum`

Print the run's `curriculum_state.json`. Reports "no curriculum" if the run has
none.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |

```bash
python -m sim_experiment.cli curriculum exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4
```

### Batches

#### `batch`

Expand a seeds × scenarios cross product into queued runs (created but not
launched).

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--seeds` | int list | all (None) | no |
| `--scenarios` | str list | all (None) | no |

```bash
python -m sim_experiment.cli batch exp_556fe0488d4db907 --seeds 1 2 3
```

#### `batch-run`

Create **and** execute a batch through the `BatchScheduler`, either with local
worker slots or a single remote worker (`--worker` requires `--token`). Exit 1
if any job failed or was cancelled. Retryable failures (`trainer_crash`,
`timeout`, `worker_crash`, …) are retried up to `--max-retries`.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--seeds` | int list | all (None) | no |
| `--scenarios` | str list | all (None) | no |
| `--trainer` | str | `ppo` | no |
| `--env-mode` | `inprocess`\|`process`\|`tcp`\|`tcp_multi` | `inprocess` | no |
| `--workers` | int | `2` | no |
| `--worker` | `host:port` | — | no |
| `--token` | str | — | no (required with `--worker`) |
| `--max-retries` | int | `0` | no |
| `--timeout` | float (seconds) | `600` | no |

```bash
python -m sim_experiment.cli batch-run exp_556fe0488d4db907 --seeds 1 2 3 \
    --workers 2 --max-retries 1 --timeout 1200
```

### Trainers & analysis

#### `trainers`

Dump `TRAINER_CAPABILITIES` (ppo, sac, dqn, bc, dummy) as JSON — supported
algorithms, action types, observation types, multi-env and eval support.

| Flag | Type | Default | Required |
|---|---|---|---|
| — | — | — | — |

```bash
python -m sim_experiment.cli trainers
```

#### `compare`

Compare run metric series within an experiment (raw + smoothed, max/min/final/
mean). `--runs` selects a comma-separated subset; default is all runs.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--metric` | str | `reward` | no |
| `--scope` | str | `episode` | no |
| `--smooth` | int | `10` | no |
| `--runs` | csv str | `""` (all runs) | no |

```bash
python -m sim_experiment.cli compare exp_556fe0488d4db907 --metric reward --smooth 10
```

### Datasets

#### `dataset-export`

Export run trajectories to a `transitions_v1` dataset (`episodes.jsonl` +
`manifest.json`). Optional filters: minimum return, environment fingerprint,
termination reasons. Diagnostic fields are excluded; only whitelisted agent
fields are written.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `run_id` | str (positional) | — | yes |
| `--dest` | path | — | yes |
| `--min-return` | float | — | no |
| `--env-fingerprint` | str | — | no |
| `--termination-reasons` | csv str | — | no |

```bash
python -m sim_experiment.cli dataset-export exp_556fe0488d4db907 run_20261003_101500_a1b2c3d4 \
    --dest datasets/run1 --min-return 100
```

#### `dataset-validate`

Validate a `transitions_v1` dataset directory (manifest format, unique episode
ids, required step fields, length consistency). Exit 0 if valid, 1 otherwise.

| Flag | Type | Default | Required |
|---|---|---|---|
| `dataset_dir` | path (positional) | — | yes |

```bash
python -m sim_experiment.cli dataset-validate datasets/run1
```

#### `dataset-split`

Write a deterministic `splits.json` (train/val/test) into the dataset. The
train fraction is `1 - val_frac - test_frac` (0.8 by default); assignment is a
deterministic hash of `{seed}:{episode_id}`.

| Flag | Type | Default | Required |
|---|---|---|---|
| `dataset_dir` | path (positional) | — | yes |
| `--val-frac` | float | `0.1` | no |
| `--test-frac` | float | `0.1` | no |
| `--seed` | int | `42` | no |

```bash
python -m sim_experiment.cli dataset-split datasets/run1 --val-frac 0.1 --test-frac 0.1
```

#### `dataset-stats`

Inspect a dataset: episode/step counts, return and length statistics,
termination-reason histogram, observation/action dims, fingerprints, splits,
and validation status.

| Flag | Type | Default | Required |
|---|---|---|---|
| `dataset_dir` | path (positional) | — | yes |

```bash
python -m sim_experiment.cli dataset-stats datasets/run1
```

#### `train-bc`

Launch a behavioral-cloning run on a `transitions_v1` dataset. Forces the
manifest's algorithm to `bc` for this run (the dataset carries the learned
behavior; the experiment pins env/scenario fingerprints). Dataset fingerprint
must match the run contract.

| Flag | Type | Default | Required |
|---|---|---|---|
| `experiment_id` | str (positional) | — | yes |
| `--dataset` | path | — | yes |
| `--epochs` | int | `20` | no |
| `--val-frac` | float | `0.2` | no |
| `--seed` | int | `42` | no |
| `--wait` | float (seconds) | `0` | no |

```bash
python -m sim_experiment.cli train-bc exp_556fe0488d4db907 --dataset datasets/run1 --wait 600
```

### Remote workers

Worker protocol `1.0` — TCP NDJSON request/response with a shared `--token`
(HMAC-compared). The worker service enforces experiment-dir containment for
LAUNCH/POLL/CANCEL.

#### `worker-serve`

Run a worker service on this machine (blocks until Ctrl+C).

| Flag | Type | Default | Required |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | no |
| `--port` | int | `9100` | no |
| `--token` | str | — | yes |
| `--worker-root` | path | `<repo>/experiments` | no |

```bash
python -m sim_experiment.cli worker-serve --port 9100 --token s3cret
```

#### `worker-status`

Handshake with a remote worker and print its capabilities.

| Flag | Type | Default | Required |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | no |
| `--port` | int | — | yes |
| `--token` | str | — | yes |

```bash
python -m sim_experiment.cli worker-status --port 9100 --token s3cret
```

#### `worker-register`

Register this CLI with a remote worker (returns `REGISTER_ACK` with the assigned
`worker_id`).

| Flag | Type | Default | Required |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | no |
| `--port` | int | — | yes |
| `--token` | str | — | yes |

```bash
python -m sim_experiment.cli worker-register --port 9100 --token s3cret
```

#### `worker-list`

Handshake with a worker and print its capability list.

| Flag | Type | Default | Required |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | no |
| `--port` | int | — | yes |
| `--token` | str | — | yes |

```bash
python -m sim_experiment.cli worker-list --port 9100 --token s3cret
```

### Benchmarks

#### `benchmark`

Measure headless environment throughput (env-steps/s) for a template project at
several vector sizes, stepping a fixed action `[0.0, 0.5, 0.0]`.

| Flag | Type | Default | Required |
|---|---|---|---|
| `--envs` | csv ints | `1,2,4` | no |
| `--steps` | int | `2000` | no |
| `--template` | str | `lane_following` | no |

```bash
python -m sim_experiment.cli benchmark --envs 1,2,4 --steps 2000
```

#### `learn-bench`

Run the phase-6 learning benchmark (`benchmarks.phase6.benchmark_runner`) on a
JSON config, then print a `convergence_report` verdict
(`insufficient_data`\|`regressed`\|`unstable`\|`improved`\|`no_improvement`).
Exit 1 if the verdict is `regressed`.

| Flag | Type | Default | Required |
|---|---|---|---|
| `--config` | path | — | yes |
| `--work-dir` | path | — | no |

```bash
python -m sim_experiment.cli learn-bench --config benchmarks/phase6/config.json
```

## `python main.py` — simulator launcher

Launches Simulation Studio (GUI) or a headless TCP server for external agents
(protocol 2.1).

| Flag | Type | Default | Description |
|---|---|---|---|
| `--headless` | flag | off | Run without GUI |
| `--port` | int | `8765` | TCP port for external AI model connection |
| `--num-envs` | int | `1` | Headless only: host N independent envs on `--port` via one multi-client server |
| `--track` | str | studio home | Open `oval`, `serpentine`, `obstacle`, or a path to a `.sim.json` directly into the workspace |
| `--width` | int | `1280` | Window width |
| `--height` | int | `720` | Window height |

```bash
python main.py --headless --port 8765 --num-envs 4
python main.py --track oval --width 1600 --height 900
```

## Tools & sample agents

| Command | Purpose |
|---|---|
| `python tools/ui_shots.py [W H]` | Capture ~35 QA screenshots to `assets/screenshots/qa/<W>x<H>_*.png` |
| `python tools/smoke_interactions.py` | ~37 interaction checks through real app event handling |
| `python tools/tcp_recording_e2e.py` | 15-check TCP-driven recording end-to-end test |
| `python -m sim_client.agents.pid_driver [host] [port]` | PID driver agent (**positional** host/port args) |
| `python -m sim_client.agents.random_agent` | Random-action agent (no args) |
| `python -m sim_client.agents.ppo_train` | PPO rollout **demo** — not a real trainer (no args) |

```bash
python -m sim_client.agents.pid_driver 127.0.0.1 8765
python -m sim_client.agents.random_agent
```

## See also

- [Experiments overview](../experiments/experiments.md)
- [Training](../reinforcement-learning/training.md)
- [External agents](../agents/external-agents.md)
- [Development](../development/development.md)
