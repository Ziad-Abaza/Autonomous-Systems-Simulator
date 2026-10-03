---
noteId: "317156f0bedc11f1a29f1fbaabbd87c8"
tags: []

---

# Environment Versioning & Configuration Diff Specification

## Overview

In Reinforcement Learning research, reproducibility is paramount. Minor unnoticed shifts in observation normalization, reward shaping weights, action bounds, or boundary friction can render policy checkpoints incompatible or bias empirical benchmarks.

The **AI Environment Simulation Studio** implements a deterministic structural versioning and configuration diffing architecture.

---

## 1. Semantic Versioning Model

Environments maintain a semantic version `MAJOR.MINOR.PATCH` managed by `EnvironmentVersion`:

- **MAJOR**: Incompatible space breaks (e.g., changes to observation vector dimension, discrete vs continuous action space switch, removed/added action channels).
- **MINOR**: Backward-compatible semantic additions (e.g., adding an optional sensor, expanding reward components, modifying scenario presets).
- **PATCH**: Incremental parameter tweaks (e.g., fine-tuning reward weights, adjusting boundary friction, moving control points).

---

## 2. Deterministic SHA-256 Structural Fingerprinting

Every environment project produces a deterministic 256-bit SHA-256 hash representing the physical and algorithmic configuration:

```python
from sim_env.versioning import EnvironmentVersionManager

fingerprint = EnvironmentVersionManager.compute_fingerprint(project_dict)
```

### Determinism Rules
- **Metadata Exclusion**: Volatile metadata fields (`timestamp`, `last_saved`, `author`, `version`) are stripped before hashing.
- **Key Sorting**: JSON keys are lexicographically ordered (`sort_keys=True`).
- **Invariance**: Identical track geometries, agent perception pipelines, reward terms, and physics coefficients produce identical SHA-256 digests across different machines and operating systems.

---

## 3. Automatic Version Increment on Change

When saving an `EnvironmentProject`, the serialization engine compares the new fingerprint with the fingerprint recorded at load time:

```python
proj = EnvironmentProject.load("oval.sim.json")
# ... user adjusts tire friction or reward weights in the Studio ...
proj.save("oval.sim.json")  # Detects fingerprint divergence and auto-increments PATCH version
```

If the project configuration was unchanged, the version is preserved without increment.

---

## 4. Structured Configuration Diff Engine

`EnvironmentVersionManager.compare_configurations()` generates machine-readable and human-formatted diff reports:

```python
report = EnvironmentVersionManager.compare_configurations(
    dict_a=v1_data,
    dict_b=v2_data,
    ver_a="1.0.0",
    ver_b="1.0.1"
)

print(report.format_text())
```

### Example Diff Output

```
Environment Diff: v1.0.0 (d87ffddb) -> v1.0.1 (a4e321bf):
  [TRACK] unchanged
  [OBSERVATION]
    * channel 'heading_error' enabled: False -> True
  [ACTION] unchanged
  [REWARD]
    * component 'centering' weight: 0.5 -> 1.0
  [TERMINATION] unchanged
  [ENTITIES & SCENARIO]
    * Scenario surface_friction_mult: 1.0 -> 0.85
```

---

## 5. Subsystem Diff Coverage

The diff engine inspects both Phase 3 declarative definitions and legacy configurations across six primary subsystems:

1. **Track**: Control point count, loop closure, boundary configurations.
2. **Observation**: Added/removed observation channels, channel enablement, shape/normalization.
3. **Action**: Continuous vs discrete space type, channel min/max bounds, dead zones, rate limits.
4. **Reward**: Component addition/deletion, weight changes, active state toggles.
5. **Termination**: Rule addition/deletion, enabled states, threshold parameters.
6. **Entities & Scenario**: Scene entity count, scenario weather, time of day, surface friction, lighting.
