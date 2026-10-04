---
noteId: "6648e720bf8711f1a29f1fbaabbd87c8"
tags: []

---

# agentRL — Continual RL Driving Platform

Production-grade reinforcement-learning layer for this simulator:
real agents (SAC / PPO), multi-track + continual training with
rehearsal, obstacle learning, seeded reproducibility, and an
evaluation/experiment infrastructure that produces *measured* evidence.

Everything below is operational documentation — commands, parameters,
defaults, and the tuning recipe that actually produced driving policies.
For measured results see `AGENT_RL_FINAL_REPORT.md`; for design see
`AGENT_RL_ARCHITECTURE.md`.

---

## 1. Requirements

- Python 3.13, PyTorch (CPU build works; ~30–48 env-steps/s per run)
- Run everything from the **repo root** (`Simulation/`), not inside `agentRL/`
- Tests: `pytest tests/agent -q` (69 tests)

## 2. Package map

```
agentRL/
  core/        versioning, seed_tree, AgentConfig/TrainConfig, BaseRLAgent
  obs/         ObservationSpec (channel subsets, frame_stack, prev_action), ObsEncoder
  act/         ActionAdapter — [-1,1]^3 <-> env bounds, positive-half brake
  rewards/     DRIVE_V1 reward preset + TERM_V1 termination preset
  envs/        TrackRegistry, track_gen (parametric loops), EnvFactory, ScenarioMutator
  memory/      ReplayBuffer + TrackRehearsalBuffer (per-track buffers)
  algos/       SACAgent (primary), PPOAgent (rollouts + GAE)
  train/       OffPolicyTrainer, OnPolicyTrainer, MixedTrackTrainer, ContinualTrainer
  eval/        evaluate_policy, failure classifier, EvalMatrix
  baselines/   PD lane-follower + collect_demos (buffer warmstart)
  experiments/ matrix runner + configs e001..e010 + runs/ artifacts
tests/agent/   lab scripts (run_policy, evaluate_policy, inspect_episode,
               watch_agent, compare) + the test suite
```

## 3. Quickstart

### Train (the proven recipe)

```powershell
# SAC on the oval: 60k steps, 15k PD-demo warmstart, 8 m/s spawn
python -m agentRL.experiments.matrix --exp E001c --algos sac
```

Artifacts land in `agentRL/experiments/runs/E001c/sac/`:
`metrics.jsonl`, `checkpoints/latest.pt` + `step_*.pt`,
`run_state.json`, `experiment_result.json`.

### Evaluate a checkpoint

```powershell
python tests/agent/evaluate_policy.py `
  --ckpt agentRL/experiments/runs/E003/sac/checkpoints/latest.pt `
  --track oval --out eval.json
# prints: mean_return / completion / collision_rate / mean_speed
```

### Watch it drive in the 3D studio

```powershell
# terminal 1 — opens the rendered studio + external-AI TCP server
python main.py --track oval --port 8765

# terminal 2 — the checkpoint drives; you watch
python tests/agent/watch_agent.py `
  --ckpt agentRL/experiments/runs/E003/sac/checkpoints/latest.pt `
  --episodes 5 --stochastic
```

`--stochastic` runs the trained policy's sampler (the behavior that
produced the best episodes); omit it for the deterministic mean policy,
which is more cautious and slower mid-training.

Headless TCP variant (no window): `python main.py --headless --port 8765`
then the same `watch_agent.py` command — this is the path exercised by
the TCP end-to-end test.

### One-shot inline (no files)

```powershell
python tests/agent/run_policy.py `
  --ckpt <ckpt.pt> --track oval --episodes 3   # in-process, prints metrics
```

## 4. Experiment matrix

Configs live in `agentRL/experiments/configs/eNNN.json`. Run one:

```powershell
python -m agentRL.experiments.matrix --exp E003 --algos sac
python -m agentRL.experiments.matrix --exp E003 --steps-override 2000   # smoke
```

| Kind | Config keys | What it does |
|---|---|---|
| `single` | `track`, `steps` | one env, one track |
| `mixed` | `tracks[]`, `steps` | env pool, random track per episode, per-track rehearsal buffers |
| `continual` | `phases[{track_id,steps,demos}]`, `holdouts[]`, `eval_seeds`, `eval_max_steps` | sequential A→B→C, eval matrix every phase, `continual_report.json` with retention/forgetting/transfer |
| `eval` | `ckpt`, `tracks[]`, `mutator?`, `eval_seeds`, `eval_max_steps` | no training — deterministic eval grid |

Shared config fields:

| Field | Default | Meaning |
|---|---|---|
| `seed` | 42 | master seed → `seed_tree` splits env/torch/mutator/eval |
| `obs_preset` | `full23` | `full23` (all channels) or `state8` (subset — see §6) |
| `frame_stack` | 1 | stacked frames through the encoder |
| `prev_action` | false | append last action to obs |
| `initial_speed` | null | spawn speed m/s — **8.0 is the unlock** (see §7) |
| `demos` | null | `{"steps": N, "vmax": 14}` PD-controller buffer warmstart |
| `mutator` | null | obstacle/spawn/friction/noise randomization (see §5) |
| `reward` / `termination` | `drive_v1` / `term_v1` | presets — do not downgrade to legacy idle-safe rewards |
| `agent_overrides` | {} | per-algo hyperparams (see §8) |
| `resume.halves` | 1 | `2` = train half, cold-reload from ckpt, train half |
| `eval_episodes`, `eval_interval`, `ckpt_interval`, `num_envs` | 3 / 10k / 10k / 1 | trainer cadence |

## 5. Scenario mutator (obstacles / randomization)

```json
"mutator": {
  "n_obstacles": [0, 3],     # uniform int range per episode
  "min_gap_m": 15.0,         # min spacing between obstacles
  "lateral_frac": [-0.5, 0.5], "s_range": [0.05, 0.95],
  "entity_types": ["cone", "static_obstacle"],
  "spawn_jitter": [-2.0, 2.0],
  "friction": [0.7, 0.95],
  "noise": [1.5, 2.5]
}
```

Seeded and deterministic: same seed → same scenario draws. Entities are
scenario-spawned and cleared on reset (never accumulate).

## 6. Observation presets

| Preset | Contents | Verdict |
|---|---|---|
| `full23` | all 23 agent channels incl. checkpoint distance, waypoint preview, steering state | **required** — the only spec that produced real driving |
| `state8` | 8-channel subset | **insufficient** (E009): car moves fast (~9.9 m/s) but caps at ~1 checkpoint — lacks cornering information |

`ObservationSpec(frame_stack=k, prev_action=True)` composes on top of
either preset. Checkpoints store their spec — eval always uses it.

## 7. The recipe that works (measured, E001 ablation)

| Variant | Result |
|---|---|
| affine brake mapping | car parked forever — brake≥0.1 dominates throttle. **Fixed** via `pos_only` brake (`max(0,raw)`); keep it |
| `initial_speed=8` alone | movement but 0 checkpoints in 271 episodes |
| **`initial_speed=8` + 15k PD demos** | **real driving**: 11/16 ckpts, 0.688 lap, ~9.6 m/s |

Operational rules of thumb:

- **Always** set `"initial_speed": 8.0` — cold-start exploration dies at
  the first corner otherwise.
- **Always** seed demos for off-policy training: `"demos": {"steps": 15000}`
  (10k per phase in continual runs). The PD teacher only reaches ~5
  checkpoints — SAC exceeds it; demos seed the value landscape, not the
  ceiling.
- Keep `warmup` small (500–2000) once demos exist.
- `rehearsal_fraction: 0.25` for mixed/continual — 25% of every batch
  comes from previous-track buffers; this is what retained oval skill
  through later phases (oval eval 164→478, positive backward transfer).
- Budget: ~60k steps ≈ 30 min CPU. Deterministic `mu` lags the
  stochastic policy until late — expect the deterministic eval to be a
  cautious driver; judge training by stochastic episodes + evals with
  matched spawn (`initial_speed` in eval configs too — from-rest evals
  of mid-training checkpoints stall and are misleading).

## 8. Agent hyperparameters (`agent_overrides`)

SAC (defaults):

| Key | Default | Notes |
|---|---|---|
| `hidden_sizes` | (256,256) | actor + twin critics |
| `lr` | 3e-4 | shared for actor/critics/alpha |
| `gamma` | 0.99 | |
| `tau` | 0.005 | target-critic EMA |
| `warmup` / `random_steps` | 2000 | random-action steps before policy acts |
| `batch` | 256 | |
| `buffer` | 200k | per-track capacity |
| `rehearsal_fraction` | 0.25 | batch share from non-current buffers |
| `updates_per_step` | 1 | |
| `init_alpha` | 0.2 | entropy temperature (auto-tuned; ≈0.15 observed) |
| `grad_clip` | 10.0 | NaN/Inf batches are skipped + counted |

PPO (secondary): rollout storage, GAE, truncation bootstrap — same
`agent_overrides` channel; run with `--algos ppo` on any kind.

## 9. Checkpoints, resume, determinism

- `checkpoints/latest.pt` every `ckpt_interval` + `step_N.pt` snapshots.
- `load_agent(path)` refuses version mismatches (checkpoint contract
  versioned via `core/versioning`).
- Cold-resume: `"resume": {"halves": 2}` runs two fresh halves with a
  `resume_mid.pt` reload — replay buffer is *not* serialized (state_dict
  keeps only metadata); networks + train_state carry over. E010 verified.
- Seeding: `seed_tree(seed)` → `env`, `torch`, `mutator`, `eval` substreams.
  Same config + same seed = same experiment.

## 10. Metrics & artifacts

`metrics.jsonl` scopes:

- `episode`: `return, length, mean_speed, steer_smoothness,
  termination_reason, lap_progress, laps_completed, checkpoints` (+`track` in mixed)
- `update`: `critic_loss, actor_loss, alpha, q_mean, buf, nan_guard`
- `heartbeat` every 50 steps: `episodes, ep_return, ep_len, speed, buf, sps`
- `eval`: eval_fn metrics at `eval_interval`
- `failure`: env-contract errors (e.g. step-after-done) — count is a
  health signal; thousands of these means an integration bug (this is
  how the E004 stale-env bug surfaced)

Continual runs add `continual_report.json`: per-phase eval cells across
all seen tracks + holdouts, `derived.forgetting / retention_final /
transfer`, rehearsal buffer state.

## 11. Evaluation semantics

`evaluate_policy` / `EvalMatrix` act **deterministically** and classify
failures per episode: `collision`, `off_road`, `wrong_direction`,
`stall_timeout`, `timeout_progress`, `completion`.

- Evals must match training spawn conditions: include `initial_speed`
  in eval-kind configs (from-rest evals of mid-training checkpoints
  stall — measured, not theoretical).
- `completion_rate` requires a full lap; `mean_progress` is the lap
  fraction reached (16 checkpoints = 1.0).

## 12. Tracks

| ID | Source | Use |
|---|---|---|
| `oval` | tracks/basic_driving_proving_ground | canonical training track |
| `serpentine` | tracks/lane_following_serpentine_circuit | harder curves |
| `smoke` | generated open straight (file removed upstream) | tests |
| `gen_loop_0..3` | parametric seeded loops | train/generalization |
| `gen_loop_4..5` | parametric seeded loops | **holdouts** — never trained on |

Missing authored files are skipped (registry is resilient); generated
tracks are deterministic per seed.

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| speed 0.0, `stuck`, return ≈ −24 | brake mapping (pre-fix) or parked deterministic mu | verify `pos_only` adapter; use `--stochastic` or later checkpoint |
| collisions at ~130–200 steps, no checkpoints | cold-start exploration deadlock | `initial_speed: 8.0` + `demos.steps: 15000` |
| thousands of `failure` rows | env stepped after done | fixed upstream; check `self.env` rotation in mixed mode |
| deterministic eval = stall, stochastic = drives | SAC mid-training mu lag | more steps, or eval with `initial_speed` |
| `track file not found` / missing smoke | upstream removed track files | registry skips missing files; smoke is now generated |
| KeyError track `gen_loop_N` | N≥6 not generated | registry provides gen_loop_0..5 only |

## 14. Lab scripts (`tests/agent/`)

| Script | Purpose |
|---|---|
| `run_policy.py --ckpt X --track T [--tcp PORT]` | quick rollout metrics |
| `evaluate_policy.py --ckpt X --track T --out J` | deterministic eval JSON |
| `inspect_episode.py --ckpt X` | per-step channel dump |
| `compare.py` | side-by-side checkpoint comparison |
| `watch_agent.py --ckpt X [--stochastic]` | drive the rendered studio over TCP |
