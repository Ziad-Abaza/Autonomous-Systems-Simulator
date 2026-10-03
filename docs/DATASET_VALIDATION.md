---
noteId: "b0f69570bf1311f1a29f1fbaabbd87c8"
tags: []

---

# Dataset Validation, Splitting & Inspection

`transitions_v1` datasets (episodes.jsonl + manifest.json) are the
imitation-learning currency. These services keep them honest.

## `validate_dataset(dataset_dir)`

Independent structural validation — returns
`{valid, errors[], warnings[], episodes, steps, fingerprints, schema_hash}`:

- `dataset_format` == `transitions_v1`
- manifest episode/step counts match episodes.jsonl
- every episode: `episode_id` unique, non-empty `steps`, `length` field
  consistent
- every step: required agent fields (`obs`, `action`, `reward`) present;
  **no unexpected fields** — diagnostic keys (speed, offsets, etc.) are
  a contract violation, not a warning
- last step should be terminal (warning, not error)
- env_fingerprint consistency vs manifest (warning — multi-fingerprint
  datasets are legal when produced without a filter)

## `split_dataset(dataset_dir, seed, ratios)`

Deterministic episode-level partition into `train`/`val`/`test` (or any
named ratio map): episodes ordered by `sha256("{seed}:{episode_id}")`,
allocated by largest-remainder so counts sum exactly. Same inputs →
same splits on any machine. Persists `splits.json` alongside the dataset.

## `dataset_statistics` / `inspect_dataset`

`sim_experiment/dataset_inspect.py`: counts, return & length
distributions, termination-reason histogram, obs/action dims, fingerprint
and scenario inventories, split sizes. `inspect_dataset` bundles
validation + stats + manifest for the UI dataset surface.

## `sample_trajectories(run_dir, count, seed, strategy)`

Deterministic episode selection from a run's `trajectories/` —
replaces prefix-capture semantics with seed-keyed hashing:
`uniform` (default), `best` (top return), `mixed`.

## Trajectory header schema

Headers now carry provenance beyond the run seed:
`episode_seed` (exact reset seed), `env_index`,
`curriculum_stage_index`, plus the existing fingerprint / scenario /
schema fields. Readers are tolerant of older headers missing the new
keys; `_read_episode` requires `type == "header"` on the first record
(no positional assumptions).

## Trajectory capture control

`training.algorithm_config.trajectory_sampling` configures the recorder:
`first_n` (default, back-compat with `trajectory_episodes`), `every_n`,
`probability` (deterministic hash), `episodes` (explicit ids),
`min_return`, `termination_reasons`, `all`. Episodes are buffered and
written atomically at episode end — a dropped episode leaves no partial
file.

## CLI

```powershell
python -m sim_experiment.cli dataset-validate <dataset_dir>
python -m sim_experiment.cli dataset-split <dataset_dir> --val-frac 0.1 --test-frac 0.1 --seed 42
python -m sim_experiment.cli dataset-stats <dataset_dir>
```
