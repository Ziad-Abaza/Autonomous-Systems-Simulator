---
noteId: "4ba2ed30bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Experiment CLI Reference

Entry point: `python -m sim_experiment.cli [--root EXPERIMENTS_DIR] <command>`

The CLI is a thin layer over the domain services — identical semantics
to the UI.

## Commands

| Command | Purpose |
|---------|---------|
| `validate-env --project P.sim.json` | Run the Phase 3 RL gatekeeper; exit 1 if invalid |
| `create --project P --scenario S --name N --algorithm A --timesteps T ...` | Create + persist an experiment; prints `experiment_id` |
| `list` | List all experiments (summaries) |
| `show <exp_id>` | Dump the full manifest |
| `launch <exp_id> [--trainer ppo] [--env-mode inprocess\|tcp] [--wait S]` | Launch a run |
| `batch <exp_id> [--seeds ...] [--scenarios ...]` | Deterministic expansion → multiple runs |
| `runs <exp_id>` | List runs |
| `status <exp_id> <run_id> [--watch]` | Poll a run (progress + latest metrics) |
| `cancel <exp_id> <run_id>` | Cancel a live run |
| `resume <exp_id> <run_id>` | New run from the latest checkpoint |
| `evaluate <exp_id> <run_id> [--checkpoint PATH]` | Headless checkpoint evaluation → result artifact |
| `trajectories <exp_id> <run_id>` | List trajectory files |
| `reproduce <exp_id>` | Reproducibility checks; exit 1 if not reproducible |
| `export <exp_id> --dest DIR` | Copy the complete experiment bundle |
| `archive <exp_id>` | Archive flag |
| `benchmark [--envs 1,2,4] [--steps N]` | Env throughput benchmark |

## Example session

```powershell
python -m sim_experiment.cli create `
    --project presets/oval_circuit.sim.json `
    --scenario basic_lane_following --name ppo-base `
    --timesteps 50000 --ckpt-freq 10000 --eval-freq 10000 --seed 42
# -> Created experiment exp_0be32bc6...

python -m sim_experiment.cli launch exp_0be32bc6 --trainer ppo --wait 900
python -m sim_experiment.cli status exp_0be32bc6 <run_id> --watch
python -m sim_experiment.cli evaluate exp_0be32bc6 <run_id>
python -m sim_experiment.cli reproduce exp_0be32bc6
python -m sim_experiment.cli export exp_0be32bc6 --dest exports/base
```

## Exit codes

- `0` success / `1` validation or operational failure / `2` invalid
  contract.
- `reproduce` exits `1` when the experiment is not reproducible
  (CI-friendly gate).
