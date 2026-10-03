---
noteId: "42046530bf7111f1a29f1fbaabbd87c8"
tags: []

---

# agentRL — Final Report

Continual reinforcement-learning autonomous-driving platform for the
Simulation repository. This document records **measured** results only;
every number below comes from `agentRL/experiments/runs/` artifacts or
`tests/agent/` output, not from inference or architecture claims.

## 1. What was built

| Layer | Path | Purpose |
|---|---|---|
| Core | `agentRL/core/` | versioned configs, seed tree, checkpoint contract |
| Observations | `agentRL/obs/` | `ObservationSpec` (channel subsets, frame stacking, prev-action), `ObsEncoder` |
| Actions | `agentRL/act/` | `ActionAdapter` — normalized `[-1,1]^3` ↔ env bounds, positive-half brake |
| Rewards | `agentRL/rewards/` | `DRIVE_V1` (stationary-exploit fix) + `TERM_V1` presets |
| Environments | `agentRL/envs/` | `TrackRegistry`, parametric `track_gen`, `EnvFactory`, `ScenarioMutator` |
| Memory | `agentRL/memory/` | `ReplayBuffer` + `TrackRehearsalBuffer` (terminated/truncated preserved) |
| Algorithms | `agentRL/algos/` | `SACAgent` (primary), `PPOAgent` (rollouts, GAE, truncation bootstrap) |
| Training | `agentRL/train/` | off/on-policy trainers, `MixedTrackTrainer`, `ContinualTrainer`, metrics.jsonl |
| Evaluation | `agentRL/eval/` | `evaluate_policy`, failure classifier, `EvalMatrix` |
| Baselines | `agentRL/baselines/` | calibrated PD lane-follower + `collect_demos` warmstart |
| Laboratory | `tests/agent/` | run/evaluate/compare/inspect scripts + TCP E2E |
| Experiments | `agentRL/experiments/` | config-driven matrix runner, E001–E010 |

## 2. Engineering findings (the "what does not work" section)

### 2.1 Brake-dominance action mapping (critical)

The simulator's brake channel dominates engine torque completely:
brake ≥ 0.1 parks the car at full throttle. An affine `[-1,1] → [0,1]`
mapping makes any near-zero-mean policy emit ≈ 0.5 brake forever.
The first E001 run therefore produced five consecutive 1200-step
episodes at exactly 0.00 m/s (`stuck` termination, return ≈ −24).

**Fix**: `ActionAdapter.pos_only` — the brake channel maps only its
positive half (`max(0, raw)`), so neutral policy output releases the
brake. Regression test: `test_pos_only_brake`.

### 2.2 Stationary-reward exploitation

The pre-existing reward defaults let a parked policy survive to the
horizon while collecting time/centering bonuses (observed in earlier
baseline experiments: near-zero speed, zero laps, max-duration episodes).
`DRIVE_V1` adds a per-step idle cost and weights progress/speed, making
stationary survival strictly unattractive (−0.02/step × 1200 = −24/ep,
worse than any attempt that moves).

### 2.3 Spawn velocity / exploration deadlock

E001b ablation: `initial_speed = 8` alone unlocks movement (returns
+2…+13, mean speed ≈ 7 m/s) but the policy plateaued at **0 checkpoints
in 271 episodes** — the car dies at corner 1 before discovering progress
reward. Movement ≠ learning.

### 2.4 Demo warmstart unlocks the value landscape

E001c (`initial_speed = 8` + 15k PD-controller demo transitions seeded
into the replay buffer) produced the first real driving:
return +434 at step 9.4k, then stable +500…+900 episodes —
11/16 checkpoints, 0.688 lap, ≈ 9.6 m/s.

The calibrated PD teacher (steer = 4·lat + 2·head − 0.8·yaw_rate,
corner-slowed throttle) itself only reaches ~5 checkpoints — the learned
policy **exceeds its teacher** on the stochastic policy, which is the
point of demos-as-buffer-seeding rather than imitation.

### 2.5 Deterministic vs stochastic gap

SAC kept α ≈ 0.15 through 60k steps; the actor mean `mu` lagged the
distribution's useful tail. Checkpoints 30k–50k evaluate deterministically
as *stationary* (−24, 0.00 m/s) while their stochastic samples drove
+900-return episodes. By 60k, `mu` entered the driving mode:
deterministic eval = 0 collisions, 2.5 m/s, 0.125 progress from rest
(`timeout_progress` classification). Consolidating `mu` further needs
more training steps than the 60k budget.

## 3. Experiment matrix

| ID | Kind | Question | Result |
|---|---|---|---|
| E001 | single | Does SAC learn on the oval? | see ablation below |
| E001b | single | initial_speed alone | negative — 0/271 episodes reach a checkpoint |
| E001c | single | + PD demos | **evidence gate passed** |
| E002 | single | recipe on serpentine | TBD |
| E003 | single | obstacle-mutated oval | TBD |
| E004 | mixed | simultaneous 4-track | TBD |
| E005 | continual | oval→serpentine→gen_loop_0 + retention | TBD |
| E006 | eval | unseen gen_loop_4/5 generalization | TBD |
| E007 | eval | obstacle generalization | TBD |
| E008 | eval | friction/noise robustness | TBD |
| E009 | single | state8 obs ablation | TBD |
| E010 | single | cold-resume integrity | TBD |

## 4. Measured results

### E001c — oval SAC (seed 42, 60k steps, 15k demos)

| Metric | Value | Source |
|---|---|---|
| Best episode return | +904.5 | metrics.jsonl @ 37k |
| Best lap progress | 0.688 (11/16 checkpoints) | metrics.jsonl |
| Mean speed (driving episodes) | ≈ 9.6 m/s | metrics.jsonl |
| Deterministic eval (60k) | return +170.3, 0 collisions, 2.52 m/s, 0.125 progress | eval_det.json |
| Failure class (deterministic) | timeout_progress | eval_det.json |

### Learning curve milestones (E001c)

| Step | Event |
|---|---|
| 9.4k | first +434 return, 6 checkpoints — demos consolidating |
| 34k | first +840, 11 checkpoints, 0.688 lap, survived 1500-step horizon |
| 37–44k | repeated +500…+900 episodes at ~9.6 m/s |
| 60k | deterministic policy: cautious zero-collision driver |

## 5. Reproduce

```powershell
python -m agentRL.experiments.matrix --exp E001c --algos sac
python tests/agent/evaluate_policy.py --ckpt <run>/checkpoints/latest.pt --track oval --out eval.json
```

## 6. Known limitations

- CPU-only training (~30-48 env-steps/s); 60k steps ≈ 30 min/run.
- Lap completion (progress = 1.0) was not reached inside the 60k budget;
  corner 3 (s ≈ 0.69) is the current frontier.
- The PD teacher is lane-centering only — it does not avoid obstacles;
  obstacle-mutated demos seed collision transitions deliberately.
- Stochastic ≫ deterministic at this training budget (see 2.5).
