---
noteId: "145cd750bf0011f1a29f1fbaabbd87c8"
tags: []

---

# Experiment Identity & Versioning

## Two fingerprints

| Fingerprint | Source | Covers |
|-------------|--------|--------|
| `environment_fingerprint` | `EnvironmentVersionManager.compute_fingerprint(env_dict)` | The full serialized environment: agent (obs/action/reward/termination), episode config, curriculum, scenario, sensors, entities, road |
| `experiment_fingerprint` | `ExperimentManifest._compute_fingerprint()` | Environment fingerprint + scenario + schemas + episode/randomization/curriculum + **training config** + seed + simulator + protocol versions |

`experiment_id = "exp_" + fingerprint[:16]` — deterministic, human-
inspectable, collision-safe for a local experiment store.

## Identity payload

```python
{
  "environment_fingerprint": ...,
  "scenario_configuration": {...},
  "random_seed": 42,
  "simulator_version": "4.0.0",
  "protocol_version": "2.0",
  "observation_schema": {...}, "action_schema": {...},
  "reward_configuration": {...}, "termination_configuration": {...},
  "episode_configuration": {...}, "curriculum_configuration": {...},
  "randomization_configuration": {...},
  "training": {...}, "evaluation": {...},
}
```

Hashed via `json.dumps(..., sort_keys=True)` → SHA-256 — platform-stable.

## What changes identity

- Any agent/env/scenario structural change
- Training hyperparameters (common fields **and** `algorithm_config`)
- Seed, simulator version, protocol version
- Evaluation configuration

## What does NOT change identity

- `name`, `created_at`, `metadata`, `launched`, `archived`
- Environment `environment_version`/`schema_version`/`author` (stripped
  from the fingerprint — the version number is metadata, the fingerprint
  is the identity)

## Version mismatch detection

`check_reproducibility()` verifies:

- `environment.json` fingerprint == manifest's recorded fingerprint
- `scenario.json` == `scenario_configuration`
- `simulator_version` in known versions
- `protocol_version` in `SUPPORTED_PROTOCOL_VERSIONS`

A mismatch is reported explicitly — never silently corrected.

## Why env fingerprint strips version fields

`EnvironmentProject.compute_fingerprint()` now strips `version`,
`timestamp`, `last_saved`, `author`, `environment_version`,
`schema_version`, `fingerprint`. Without this, the serializer's
auto-increment (`1.0.0` → `1.0.1` on every save) would change the
fingerprint on every save — defeating the purpose of identity.
