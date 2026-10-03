---
noteId: "11e36750bf0a11f1a29f1fbaabbd87c8"
tags: []

---

# Observation Security & Leakage Model

## Classification contract

Every field the simulator exposes has a declared class — and the class,
never the field name, is authoritative:

| class | consumer | may enter agent obs? |
|---|---|---|
| `agent_observation` | policy | yes |
| `debug_telemetry` | UI/logging | **no** |
| `oracle_ground_truth` | privileged/verification | **no** |
| `diagnostic` | out-of-band (GET_STATE, info, trajectory diagnostics, dataset summaries) | **no** |

- Channel categories live in `ObservationChannelConfig.category`
  (`sim_env/observation_designer.py`); `validate_no_leakage()` is a
  **structural** check — a channel named "speed" with an oracle category
  is flagged, a channel named "oracle_*" with the agent class is fine.
- `sim_env/observation_contract.py` declares the diagnostic contract
  (`diagnostic_state_contract()`): the field names and metadata served by
  `GET_STATE`. Both the wire contract (DISCOVER_CONTRACT exposes
  `diagnostic_fields` + `observation_field_classes`) and the payload
  (`build_diagnostic_state`) derive from that one declaration.

## Enforcement surfaces

1. **Design time**: `ObservationSpaceDefinition.validate_no_leakage` is
   run by `EnvironmentValidator`; invalid definitions cannot be saved.
2. **Compile time**: `compiled_obs_pipeline` only emits agent-class
   channels into the observation vector.
3. **Runtime**: `info` dicts, `get_state`, and trajectory
   `diagnostic_data` are diagnostics — they are never merged into `obs`.
4. **Storage**: `trajectory.py` persists `agent_data` and
   `diagnostic_data` in separate keys; `dataset.export_dataset`
   whitelists agent fields only (diagnostics structurally excluded).
5. **Validation**: `EnvironmentValidator` derives required sensors from
   `agent.sensor_names` — no hardcoded sensor list is authoritative.

## Legacy path

`SimulationEnvironment._build_observation` (no `AgentDefinition`) binds
sensors by legacy names (`'lidar_rays'`, `'rgb_camera'`, `'imu'`). These
are legacy *binding defaults* — retained for Phase-2-era projects and
documented in `docs/LEGACY_PATHS_ANALYSIS.md`, not treated as the
authoritative contract.

## Verified by tests

- `tests/test_observation_contract.py` — structural classification,
  diagnostic contract, validator sensor derivation, ep_len consistency.
- `tests/test_environment_validator.py` — leakage rejection.
- `tests/test_analytics_dataset.py` — dataset export carries no
  diagnostic keys.
