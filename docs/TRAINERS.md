---
noteId: "06dc8710bf0a11f1a29f1fbaabbd87c8"
tags: []

---

# Trainers & Capability System

## Capability declarations (`sim_experiment/capabilities.py`)

Every trainer declares what it can run:

```python
TRAINER_CAPABILITIES = {
  "ppo":  {"algorithms": ["ppo"], "action_types": ["continuous"],
           "observation_types": ["vector"], "multi_env": True, ...},
  "sac":  {"algorithms": ["sac"], "action_types": ["continuous"], ...},
  "dqn":  {"algorithms": ["dqn"], "action_types": ["discrete"], ...},
  "dummy":{"algorithms": ["dummy"], "action_types": ["continuous","discrete"], ...},
}
```

`check_compatibility(manifest, trainer)` runs **before any process
launch** (orchestrator) — incompatible algorithm/action-space/observation/
num_envs combinations are configuration errors, not mid-training crashes.
The trainer entrypoint re-checks and writes `incompatible_environment`
into `run_result.json` if reached anyway.

## Shared harness (`sim_experiment/trainers/_harness.py`)

Common contract plumbing used by ppo/sac/dqn: contract load/validate,
env building (`env_mode` inprocess|tcp, scenario + seed overrides),
resume resolution, `action_bounds`, curriculum setup/persistence,
`EpisodeTrajectoryRecorder` (obs/action/reward/done agent data +
diagnostics separated), and `run_periodic_eval` (frozen-policy eval +
first-episode replay capture + registry).

## Algorithms

### PPO — `sim_client/agents/ppo_baseline.py` + `ppo_trainer.py`
On-policy ActorCritic, GAE, contiguous per-env rollout segments.
Phase 5 fixes: truncation bootstraps `V(final_obs)` (not terminal),
deterministic per-reset seeds (`seed + epoch*1_000_003 + env*79_919 + count`),
`set_envs()` for curriculum swaps, `ep_len` uses env-authoritative
`info["step"]`.

### SAC — `sim_client/agents/sac_baseline.py` + `sac_trainer.py`
Off-policy continuous: Gaussian tanh policy, twin Q networks, soft target
updates, automatic entropy tuning. Actions sampled in [-1,1]^d and mapped
to declared env bounds. `algorithm_config`: `buffer_size`, `warmup_steps`,
`batch_size` (from training), `tau`, `alpha`, `auto_entropy_tuning`,
`target_entropy`, `updates_per_step`, `target_update_interval`,
`update_interval`. Buffer stores terminated-only as done (truncations
bootstrap).

### DQN — `sim_client/agents/dqn_baseline.py` + `dqn_trainer.py`
Off-policy discrete only: Q-network + hard target net, linear epsilon
schedule (`eps_start`→`eps_end` over `eps_decay_steps`), uniform replay.
`algorithm_config`: `buffer_size`, `warmup_steps`, `train_freq`,
`target_update_interval`, `eps_*`, `update_interval`.

### Checkpoints
All trainers save `model/optimizer state + obs/action dims + seed +
metrics + timestep + extra` (curriculum_state when present) to
`checkpoints/policy_*.pt`; `make_policy_from_checkpoint` in
`sim_experiment/evaluation.py` rebuilds frozen policies for evaluation
for all three algorithms.

## Adding a trainer

1. Implement a runner in `sim_client/agents/` honoring the hook contract
   (`on_update({global_step, episodes, ...})`, `on_step(rec)`,
   `set_envs`, `save/load_checkpoint` with `extra`).
2. Add `sim_experiment/trainers/<name>_trainer.py` entrypoint using
   `_harness` (thin: contract → runner → artifacts).
3. Register in `TRAINER_MODULES` and `TRAINER_CAPABILITIES`.
