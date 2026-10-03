---
noteId: "22b193f0bf0911f1a29f1fbaabbd87c8"
tags: []

---

# Legacy Paths Analysis

Phase 5 audit of legacy vs. modern execution paths, with retention decisions and migration evidence.

## 1. Legacy Environment Runtime (`SimulationEnvironment` without AgentDefinition)

**Path:** `env.agent is None` → `_build_observation()` (legacy branch, env_core.py), `_validate_action()` (legacy action config), `RewardEngine`/`TerminationEngine` legacy configs.

**Consumers:** `sim_ui/inspector.py`, `sim_net/server.py` (legacy envs), `benchmarks/` scripts, `sim_project/` legacy projects, `tests/test_env_core.py` and related suites.

**Evidence of continued use:** ~60 tests exercise the legacy path; the UI can still open Phase-3-era project files.

**Decision: RETAIN, documented.** Removing it would break the majority of the test suite and legacy project files with no functional replacement today. The legacy observation path binds sensors by name (`'lidar_rays'`, `'rgb_camera'`, `'imu'`) — this is a legacy *binding default*, not the authoritative contract; the Phase 5 observation contract (`sim_env/observation_contract.py`) is the canonical classification surface for new work.

**Constraint added:** `EnvironmentValidator` no longer assumes those sensor names — it derives availability from `agent.sensor_names`.

## 2. Legacy Observation Schema (`spaces.py::ObservationSchema`)

Flat `include_*` boolean schema vs. the `ObservationSpaceDefinition` channel model.

**Consumers:** `SimulationEnvironment.observation_schema`, `sim_net/server.py` HANDSHAKE (`vector_dim`, `flatten_vector`), `SimGymEnv` spaces, `build_env_from_dicts` (manifest environment config `observation_schema` key), `presets/*.sim.json`.

**Decision: RETAIN as wire/compat layer.** It is embedded in project serialization (`environment_json`), the TCP handshake (`vector_dim`), and experiment manifests. A migration would require a versioned project-file upgrade path; no code path is harmed by retaining it since AgentDefinition environments compile `ObservationSpaceDefinition` separately.

**Gap recorded:** `include_image_channel` in manifests (Phase-3 `export_schema`) is not honored by the legacy schema — image-observation environments cannot be consumed through the legacy `vector` path. Trainers are guarded by capability declarations (`observation_types`).

## 3. Legacy Project File Format (`*.sim.json` Phase 2/3)

**Consumers:** `sim_project/serializer.py` `EnvironmentProject` round-trips it; `presets/` ships 3 projects; CLI `validate-env`/`create` read it.

**Decision: RETAIN.** Phase 3 introduced `AgentDefinition` inside the same file format — old files simply carry `agent: null` and load fine. No migration needed; new fields are additive.

## 4. `sim_recorder` / `sim_client/agents` baseline

`PPOBaseline` in `sim_client` was the Phase 2 demo trainer; Phase 4 made `sim_experiment.trainers.ppo_trainer` the contract-driven entry. `sim_client` agents are now the algorithm *runners* (`PPORunner`, `SACRunner`, `DQNRunner`) — the trainers wrap them. `sim_client/train.py` remains a convenience driver for the PPO baseline.

**Decision: RETAIN with clear roles.** `sim_client` = algorithm implementations + env client; `sim_experiment` = experiment lifecycle.

## 5. Deprecated/Dead code found

- `sim_env/env_core.py` `_build_observation` legacy branch — retained (see §1).
- `agent.py::AgentDefinition` default `sensor_names` fallback — retained as a factory default, no longer treated as authoritative (validator fix).
- No unreferenced modules found; `sim_net` remains the transport layer used by `SimGymEnv`, headless pools, and remote workers reuse the pattern.

## Summary Table

| Path | Status | Migration? |
|---|---|---|
| Legacy env runtime (no agent) | Retained | None — tests + legacy projects depend on it |
| `ObservationSchema` (spaces.py) | Retained (wire/compat) | None — additive field in Phase-3 files |
| `*.sim.json` format | Retained | None — format is forward-compatible |
| `sim_client` baselines | Retained (role: algorithm runners) | None |
| Validator sensor fallback | **Fixed** | Sensors now derived from `agent.sensor_names` |
| `get_state` contract | **Fixed** | Payload now contract-driven (diagnostic classification) |
