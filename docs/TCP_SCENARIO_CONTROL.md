---
noteId: "b0626bc0bf1311f1a29f1fbaabbd87c8"
tags: []

---

# TCP Scenario Control (Protocol 2.1)

Protocol 2.1 adds `SET_SCENARIO` / `SET_SCENARIO_ACK` so external clients
can change a running headless simulator's scenario at runtime — the
capability that makes curriculum training work over `env_mode="tcp"` and
`env_mode="tcp_multi"`.

## Wire format

```json
{"type": "SET_SCENARIO", "payload": {
    "scenario": {<ScenarioDefinition dict>}   // XOR
    "scenario_id": "wet_adverse_weather",     // standard library lookup
    "reset": true,                            // apply + reset atomically
    "seed": 123                               // optional reset seed
}}
```

Response on success:

```json
{"type": "SET_SCENARIO_ACK", "payload": {
    "scenario": {"scenario_id": "...", "name": "...",
                 "weather": "...", "friction_mult": 0.85},
    "episode_state": "applied|reset",
    "obs": [...],   // present when reset=true
    "info": {...}
}}
```

## State machine

The server reports the connection's episode lifecycle as
`idle | mid_episode | terminated` (visible in `DISCOVER_CONTRACT`'s
`capabilities.episode_state`):

- `mid_episode` + `reset` not set → `ERROR scenario_update_rejected`
  (explicit, never silently applied mid-episode)
- `terminated` or `reset=true` → scenario applied; with `reset=true` the
  env is reset atomically and the ACK carries the fresh observation.

## Validation

- Protocol gating: `SET_SCENARIO` requires negotiated protocol ≥ 2.1;
  earlier clients get `unsupported_in_protocol_version` with the
  supported-version list.
- Payload: exactly one of `scenario` / `scenario_id`. Unknown
  `scenario_id` → `unknown_scenario` + known list; deserialization or
  `env.set_scenario` failures → `invalid_scenario` / `scenario_apply_failed`.
- `SimulationEnvironment.set_scenario` removes prior scenario-generated
  entities (no accumulation across resets) and refreshes the spatial
  broadphase — see `sim_env/environment.py`.

## Client

`SimGymEnv.set_scenario(dict, seed=None, reset=True)` → ACK or raises.
`SyncVectorEnv.set_scenario(dict)` broadcasts to all envs.

## Curriculum over TCP

`CurriculumController.stage_scenario_dict()` resolves each stage's
scenario + `environment_overrides` into a serialized dict; the harness
broadcasts it through the VectorEnv on every stage transition, and the
stage seed is derived deterministically. Curriculum runs identically in
`inprocess`, `tcp`, and `tcp_multi`.
