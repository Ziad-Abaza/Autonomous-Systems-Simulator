---
noteId: "2d9b1740bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Evaluation System

**File:** `sim_experiment/evaluation.py`

Evaluation is strictly separate from training state: a frozen policy is
loaded from an opaque checkpoint artifact and run in a **fresh**
environment over a fixed seed list. Results persist as immutable
artifacts under `runs/<run_id>/evaluation/`.

## EvaluationResult

```json
{
  "eval_id": "eval_20261003_...",
  "checkpoint_path": "checkpoints/policy_final.pt",
  "algorithm": "ppo",
  "env_fingerprint": "...",
  "scenario_id": "basic_lane_following",
  "deterministic_policy": true,
  "seeds": [0, 1],
  "episodes": [
    {"seed": 0, "reward": 412.5, "length": 900,
     "termination_reason": "lap_completed",
     "completed": true, "collided": false, "off_road": false,
     "timed_out": false,
     "mean_lateral_error": 0.42, "mean_heading_error": 0.06,
     "mean_speed": 14.2}
  ],
  "aggregate": {
    "episode_count": 2, "mean_reward": 380.1, "std_reward": 31.2,
    "min_reward": ..., "max_reward": ...,
    "completion_rate": 0.5, "collision_rate": 0.0,
    "off_road_rate": 0.0, "timeout_rate": 0.5,
    "mean_episode_length": 812.0,
    "mean_lateral_error": 0.44, "mean_heading_error": 0.07,
    "mean_speed": 13.8
  }
}
```

## API

```python
policy = make_policy_from_checkpoint(path, algorithm="ppo",
                                     deterministic=True)
env = build_env_from_dicts(env_dict, scenario_dict, seed=0)
result = evaluate_policy(env, policy, seeds=[0,1], num_episodes=2,
                         step_observer=optional_per_step_fn)
result.save("evaluation/eval_x.json")
```

- `make_policy_from_checkpoint` is the **adapter seam**: PPO ActorCritic
  checkpoints are supported; other algorithms plug in without touching
  `evaluate_policy`.
- `step_observer(episode_idx, step_idx, action, reward, info)` lets the
  caller capture replays/trajectories during evaluation (the PPO trainer
  uses this to write `replays/eval_<step>_ep0.json`).

## When evaluation runs

| Trigger | Where |
|---------|-------|
| `eval_frequency` in `TrainingConfig` | Inside `ppo_trainer` after each qualifying update — fresh env, deterministic actor-mean policy, `eval_<step>.json` + registry entry + `evaluation`-scoped metrics row |
| CLI `evaluate` | `sim_experiment.cli evaluate <exp> <run> [--checkpoint]` — headless, post-hoc |
| UI `EVALUATE` button | Inspector TRAIN tab → same service path |

Every evaluation records `env_fingerprint` + `scenario_id` + seeds, so a
result is always attributable to an exact environment version.
