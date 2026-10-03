---
noteId: "4227b600bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Replay & Trajectory System

Two complementary record types, both linked to runs:

## Replays (playback)

`sim_recorder/` — frame-by-frame playback data for the UI replay viewer.

Recorder metadata now captures (Phase 3 hardening):
`simulator_version`, `protocol_version`, `scenario_config`,
`env_fingerprint`, `observation_schema`, `action_schema`,
`reward_config`, `track_name`, `seed`.

Run-linked replays: during training evaluation the PPO trainer writes
`replays/eval_<step>_ep0.json` (recorder-compatible JSON) for the first
eval episode and registers it — every eval leaves a playable artifact.

## Trajectories (data)

`sim_experiment/trajectory.py` — per-step transition datasets,
`trajectories/<episode_id>.jsonl`:

```jsonl
{"type": "header", "episode_id": "train_env0_ep0",
 "env_fingerprint": "...", "scenario_id": "...",
 "seed": 42, "observation_schema": {...}, "action_schema": {...}}
{"type": "step", "episode_id": "train_env0_ep0", "step": 0,
 "agent_data": {"obs": [...], "action": [...], "reward": 0.5,
                "terminated": false, "truncated": false,
                "termination_reason": "running"},
 "diagnostic_data": {"speed": 3.0, "lateral_offset": 0.02,
                     "heading_error": 0.01, "is_colliding": false,
                     "is_on_road": true, "checkpoints_passed": 0}}
```

**Critical invariant:** agent-facing data and diagnostic/oracle data are
separate keys — diagnostic fields can never leak into agent observations
by accident.

- Buffered writes (128 records); NaN/Inf sanitized; NumPy types
  converted.
- Enabled per-run via `algorithm_config.trajectory_episodes` (0 = off).
  Records the first N episodes per env during training.
- `TrajectoryReader.read()` → `{header, steps}`.

## Inspection

- CLI: `python -m sim_experiment.cli trajectories <exp> <run>`
- `run.json` references replays via `replays[]`; trajectories via the
  artifacts registry.
- Evaluation result JSONs carry per-episode stats; replay files carry
  per-frame state for visual playback.
