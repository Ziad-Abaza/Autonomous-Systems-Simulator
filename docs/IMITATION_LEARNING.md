---
noteId: "b0ab0d80bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Imitation Learning — Behavior Cloning

BC trains a supervised policy from a `transitions_v1` dataset (exported
run trajectories or evaluation transitions). It runs through the normal
trainer contract: `algorithm="bc"`, `trainer="bc"`, plus a
`bc.dataset_dir` contract field supplied via run override.

## Pipeline

1. **Export** — `export_dataset(run_dir, out)` filters trajectories by
   env fingerprint / schema hash / return / termination reason and writes
   `episodes.jsonl` + `manifest.json`. Agent-facing step fields only —
   diagnostics are structurally excluded.
2. **Validate** — `validate_dataset(dataset_dir)` enforces format
   version, manifest counts, required step fields, terminal last step,
   and diagnostic-leak isolation. Failures → `BCDataError` before any
   training work.
3. **Fingerprint gate** — the dataset's `env_fingerprint` must equal the
   contract's; observation dims must match.
4. **Split** — `split_dataset(dir, seed, ratios)` partitions episodes by
   `sha256(seed:episode_id)` — deterministic across machines — and
   persists `splits.json`. BC trains on `train`, reports `val` loss;
   `test` stays held out.
5. **Train** — `BCRunner` (MLP): MSE for continuous action spaces,
   cross-entropy for discrete. Fixed torch/np seeds + seeded batch
   permutation → deterministic runs. Checkpoints carry model+optimizer
   state, dims, action mode, dataset provenance.
6. **Evaluate** — the trained policy is evaluated frozen on the
   contract environment (`evaluate_policy`), same protocol as RL
   trainers; `make_policy_from_checkpoint` accepts `algorithm="bc"`.

## Usage

```powershell
python -m sim_experiment.cli dataset-export <exp> <run> --dest datasets/demo
python -m sim_experiment.cli dataset-validate datasets/demo
python -m sim_experiment.cli train-bc <exp> --dataset datasets/demo --epochs 20 --wait 600
```

Or programmatically: `orch.launch(manifest, exp_dir, trainer="bc",
run_overrides={"bc_dataset_dir": ds_dir})` with
`manifest.training.algorithm = "bc"`.

## Run artifacts

Standard run layout — `metrics.jsonl` gains the `bc` scope
(`epoch`, `train_loss`, `val_loss` rows), `checkpoints/policy_final.pt`,
`evaluation/eval_*.json`, `run_result.json`.

## Evaluation transition export

`evaluation.export_transitions(env, act_fn, seeds, num_episodes, out)`
runs any frozen policy and writes a complete `transitions_v1` dataset —
usable to turn eval rollouts into BC training data.
