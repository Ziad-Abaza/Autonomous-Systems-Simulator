---
noteId: "36034b40bf1511f1a29f1fbaabbd87c8"
tags: []

---

# Phase 6 Final Report

Verification status per plan claim. Statuses: **VERIFIED** (test/proof in
repo), **PARTIAL** (implemented, limited coverage), **NOT IMPLEMENTED**.

## Execution backends

| Claim | Status | Evidence |
|---|---|---|
| `SyncVectorEnv`: vector reset/step, ordering, scenario broadcast, close | VERIFIED | `tests/test_vec_env.py` (8 tests) |
| `ProcessVectorEnv`: spawn isolation, deterministic seeds, ordering, error propagation, cleanup | VERIFIED | `tests/test_process_env.py` (8) |
| PPO/SAC/DQN accept list/Sync/Process envs; per-env episodes; reset derivation; callbacks; replay; resume; `set_envs` | VERIFIED | `tests/test_ppo_multienv.py`, `test_process_training.py` (28) |
| `env_mode="process"` end-to-end via contract | VERIFIED | process-mode PPO contract run test |
| `env_mode="tcp"` one sim per env, wrapped vector | VERIFIED | scenario-control + curriculum-over-TCP suite (14) |
| `env_mode="tcp_multi"` one process, N envs, one port | VERIFIED | `tests/test_tcp_multi.py` (5) incl. orchestrator E2E |
| `SimServerMulti` per-client isolation, slot reclaim, full rejection | VERIFIED | same file |

## Scenario & curriculum

| Claim | Status | Evidence |
|---|---|---|
| `set_scenario` runtime switching + entity cleanup + broadphase consistency | VERIFIED | `test_scenario_control.py` |
| Protocol 2.1 `SET_SCENARIO` + version negotiation + episode-state gating | VERIFIED | `test_scenario_control.py`, `test_external_protocol.py` |
| Curriculum advancement over TCP (stage overrides reach remote sims) | VERIFIED | `test_curriculum_training.py` |

## Learning validation

| Claim | Status | Evidence |
|---|---|---|
| Benchmark: baseline→train→eval→resume→eval, per-seed evals, config hash, env/platform recording | VERIFIED | `test_convergence.py` E2E (real PPO) |
| Convergence verdicts incl. stability regressions | VERIFIED | `test_convergence.py` (6 unit + E2E) |
| Learning *outcome* verdict on a long real run | PARTIAL | Framework verified at tiny timesteps; a long-run benchmark requires minutes of compute — `learn-bench --config` is the repeatable tool |

## Datasets & imitation

| Claim | Status | Evidence |
|---|---|---|
| `validate_dataset`: format, structure, isolation, counts | VERIFIED | `test_dataset_tools.py` (18) |
| `split_dataset` deterministic + complete + persisted | VERIFIED | same |
| Stats + inspect service | VERIFIED | same |
| Seed-keyed trajectory sampling (uniform/best/mixed) | VERIFIED | same |
| Header provenance: `episode_seed`, `env_index`, `curriculum_stage_index` | VERIFIED | same + `test_trajectory_sampling.py` |
| Positional-header parsing removed | VERIFIED | `_read_episode` requires `type=="header"` |
| Recorder predicate modes + atomic episode files + partial flush | VERIFIED | `test_trajectory_sampling.py` (8) |
| BC: load→validate→split→train→ckpt→resume→eval via contract | VERIFIED | `test_bc.py` (8, E2E on real PPO-sourced dataset) |
| `make_policy_from_checkpoint` supports `bc` | VERIFIED | adapter in `evaluation.py` |
| `export_transitions` produces valid transitions_v1 | VERIFIED | `test_dataset_tools.py` |

## Worker hardening

| Claim | Status | Evidence |
|---|---|---|
| worker_id registration + persistent registry (reconnect reuses record) | VERIFIED | `test_worker_hardening.py` (13) |
| Heartbeats + lease TTL → OFFLINE | VERIFIED | same |
| Dead worker job reclaimed without retry-budget burn | VERIFIED | same |
| Duplicate-dispatch lease guard | VERIFIED | same |
| Root checks on LAUNCH/POLL/CANCEL | VERIFIED | same |
| Auth + protocol versioning preserved | VERIFIED | same + `test_remote_workers.py` |
| Per-policy retryable types honored | VERIFIED | same |
| STATUS message | VERIFIED | same |

## UI & CLI

| Claim | Status | Evidence |
|---|---|---|
| Workers view from scheduler/registry | VERIFIED | `test_ui_services.py` provider tests |
| Dataset preview + episode list | VERIFIED | same |
| Multi-series comparison chart (shared axes) | VERIFIED | provider + renderer; headless boot smoke |
| CLI: learn-bench, dataset-*, train-bc, worker-*, --env-mode, --worker/--token | VERIFIED | `test_cli.py` + smoke |

## Performance

| Claim | Status | Evidence |
|---|---|---|
| Measured backend scaling (inprocess/process/tcp/tcp_multi @1/2/4 envs) | VERIFIED | `PHASE_6_PERFORMANCE.md` + raw JSON; `process` mode: 361→629→1057 sps |
| Perf parity/regression budget vs Phase 5 | PARTIAL | Phase 5 benchmark scripts still runnable; no automated cross-phase diff gate |

## Legacy

| Claim | Status | Evidence |
|---|---|---|
| Deprecation map + `sim_client/train.py` formally deprecated | VERIFIED | `LEGACY_MIGRATION.md`, `LEGACY_PATHS_ANALYSIS.md` |

## Known limitations

- `tcp` mode aggregate throughput is latency-bound (~180 sps at 4 envs);
  prefer `process` locally or `tcp_multi` for single-process hosting.
- BC consumes `transitions_v1` only; no reward-weighted or DAgger-style
  imitation.
- Worker protocol remains LAN-oriented (shared token); TLS/auth-rotation
  out of scope by design.
- Partial episodes written by `finalize()` carry non-terminal last steps
  (validator warns, does not error — intended).

## Test suite

Full repository suite: **360 passed, 0 failed** (~6 min, Windows,
Python 3.13, torch CPU).
