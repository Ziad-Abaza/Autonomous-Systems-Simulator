---
noteId: "41cc0170bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Checkpoint & Artifact System

**File:** `sim_experiment/artifacts.py` — `ArtifactRegistry`

The platform tracks artifact **references and metadata**; artifact
payloads (model weights, replay frames) are opaque files. This boundary
keeps the platform algorithm-agnostic.

## Registry

Append-only `artifacts/registry.jsonl` inside each run directory:

```json
{"kind": "checkpoint", "path": "checkpoints/policy_40960.pt",
 "created_at": 1759484000.1, "step": 40960,
 "metadata": {"algorithm": "ppo", "experiment_id": "exp_...",
              "run_id": "run_...", "env_fingerprint": "..."}}
```

Kinds: `checkpoint`, `evaluation`, `replay`, `trajectory`, `other`.

## Safety

- Paths are resolved against the run directory; traversal outside the
  run directory is rejected (`ValueError`).
- Registering a non-existent file raises `FileNotFoundError`.

## Checkpoint naming (PPO trainer)

| File | When |
|------|------|
| `checkpoints/policy_<step>.pt` | Every `checkpoint_frequency` timesteps |
| `checkpoints/policy_final.pt` | At training end (`metadata.final: true`) |

PPO checkpoint payload contains `model_state_dict`,
`optimizer_state_dict`, `obs_dim`, `act_dim`, `seed`, `metrics`,
`timestep` — enough to both evaluate the policy and **resume** the
optimizer.

## Artifact linkage

- `orch.poll()` syncs registry entries into `run.json`
  (`checkpoints`, `evaluation_results`, `replays`) — run records always
  reflect what the trainer produced.
- Every registry entry carries `experiment_id`, `run_id`,
  `env_fingerprint` → an artifact is attributable to an exact experiment
  and environment version.
- `cli export` copies the whole experiment directory — all artifacts
  travel with the experiment.

## Resume linkage

`cli resume` / UI `RESUME` looks up `latest_of_kind("checkpoint")`,
creates a new run with `resume_from = {parent_run_id, checkpoint}`, and
the contract delivers the checkpoint path to the trainer. The old run is
never mutated.
