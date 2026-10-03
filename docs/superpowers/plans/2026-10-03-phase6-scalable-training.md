---
noteId: "7b53fa90bf0c11f1a29f1fbaabbd87c8"
tags: []

---

# Phase 6 — Scalable Simulation, Learning Validation & Imitation Learning

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans or subagent-driven-development. Steps use checkbox syntax.

**Goal:** Evolve the platform to real process-parallel training, TCP scenario control, measurable learning validation, transitions_v1-based BC, and hardened workers — all local-first, Windows-compatible.

**Architecture:** Add a `VectorEnv` duck-type consumed additively by existing runners; a `multiprocessing`-spawn process backend; protocol 2.1 with SET_SCENARIO; a benchmark runner producing machine-readable results; a BC trainer on the existing contract pattern; a scheduler-side worker registry with heartbeats + leases.

**Spec:** the Phase 6 prompt (user message). **Audit:** `docs/PHASE_6_AUDIT.md` — read first; it is the source of truth for the baseline.

## Global Constraints

- Preserve all 252 passing tests; trainers stay external (never in sim_core/sim_env); diagnostics never enter agent observations; deterministic seeds; no new third-party deps without justification; Windows `spawn` multiprocessing only; no cloud/K8s/AutoML; docs describe actual behavior; no silent legacy removal.

## Review Focus

1. Torch global RNG: vector stepping interleaves policy sampling vs Phase 5 sequential order — trajectories differ (correct but numerically different); keep both runner paths.
2. `env.reset()` appends `obstacle_overrides` entities per reset (`environment.py:282-294`) — `set_scenario` must remove prior scenario entities.
3. Mid-episode SET_SCENARIO over TCP: reject unless payload `reset:true`; curriculum-over-TCP sends explicit reset.
4. `set_envs` requires constant env count — VecEnv stage change must keep N workers, only swap scenario + reseed.
5. Pickle-ability of step results across the Pipe (info dicts contain only JSON-safe types — verify, sanitize in worker if needed).

---

## Task 1: VectorEnv abstraction + SyncVectorEnv

**Files:**
- Create: `sim_experiment/vec_env.py`
- Test: `tests/test_vec_env.py`

**Interfaces:**
- Produces:
  - `class VectorEnv` (duck type, not ABC): attrs `num_envs`; methods `reset_all(seeds: List[int]) -> List[Any]`, `reset_at(i: int, seed: int) -> Any`, `step_all(actions: List[Any]) -> List[Tuple[obs, reward, terminated, truncated, info]]`, `set_scenario(scenario_dict: Optional[dict]) -> None`, `close() -> None`, `supports("process"|"tcp"|"inprocess")`.
  - `class SyncVectorEnv(VectorEnv)`: wraps `List[env]`; `step_all` loops sequentially (exact Phase-5 semantics).
  - `is_vec_env(obj) -> bool` helper (`hasattr(obj, "step_all")`).

- [ ] Write tests: N envs → N independent trajectories; reset_all deterministic seeds; done envs reset via reset_at; set_scenario swaps on next reset; close idempotent.
- [ ] Implement. Reuse `build_env_from_dicts` for env construction.

## Task 2: ProcessVectorEnv (multiprocessing spawn + Pipe)

**Files:**
- Create: `sim_experiment/process_env.py` (worker loop + parent-side class)
- Test: `tests/test_process_env.py`

**Interfaces:**
- `class ProcessVectorEnv(VectorEnv)`: `__init__(env_dict, scenario_dict, seeds: List[int])` spawns one `mp.Process`+`Pipe` per env (spawn context). Worker cmd loop: `("reset", seed)`, `("step", action)`, `("set_scenario", dict)`, `("close",)`, `("ping",)`. Worker builds env via `build_env_from_dicts(env_dict, scenario_dict, seed=seed)` and calls `env.set_scenario` for scenario swaps (falls back to rebuild if absent).
- Failure handling: `EOFError`/`BrokenPipeError`/`ConnectionResetError` on a pipe → `EnvWorkerCrash(worker_index, message)`; `step_all`/`reset_all` raise it upward (trainer fails with `worker_crash` — retryable type exists).
- `close()` sends close, joins w/ timeout, then `terminate()`.

- [ ] Tests: spawn 2 procs on a template env; independent seeds; step_all returns N results; one worker killed → EnvWorkerCrash, others unaffected; clean close; deterministic reset.
- [ ] Implement. Serialize actions as plain lists; obs returned as np arrays are picklable.

## Task 3: Runner vector-step integration (PPO/SAC/DQN)

**Files:**
- Modify: `sim_client/agents/ppo_baseline.py` (train rollout section ~215-348)
- Modify: `sim_client/agents/sac_baseline.py` (train ~192-280), `dqn_baseline.py` (train ~145-185)
- Modify: `sim_experiment/trainers/_harness.py` (`build_envs_from_contract` gains `env_mode=="process"`; returns ProcessVectorEnv)
- Modify: `sim_experiment/trainer_contract.py` (allow `"process"`; curriculum allowed for process)
- Modify: `sim_experiment/orchestrator.py` (no spawn needed — trainer owns the pool)
- Test: `tests/test_process_training.py`

**Interfaces:**
- Runners: `self.is_vec = is_vec_env(env)`; `self.vec` holds the VecEnv; `self.envs` stays `num_envs`-length concept for stats (`len(self)` semantics unchanged — keep `self.envs` list for non-vec, and for vec keep `self.envs = None`, `self.num_envs` attr). `set_envs` accepts VecEnv (calls `vec.set_scenario`? No — trainer passes same vec object after broadcasting scenario; `set_envs` just marks dirty + bumps epoch).
- PPO vec rollout: `for t in range(spe): actions=[policy(obs_e) for e]; results=vec.step_all(actions); store idx=e*spe+t; on done → vec.reset_at(e, seed)`. Buffer layout + GAE unchanged.
- SAC/DQN vec: batch actions then `step_all`; per-env bookkeeping identical to round-robin.

- [ ] Tests: PPO 2-process smoke run via real contract subprocess (256 steps) completes; metrics written; SAC/DQN vec-mode unit tests on SyncVectorEnv prove identical bookkeeping; deterministic reseeding.
- [ ] Implement. Old sequential path kept verbatim for list inputs.

## Task 4: `SimulationEnvironment.set_scenario` + entity cleanup

**Files:**
- Modify: `sim_env/environment.py` (add `set_scenario(scenario_def)`, fix scenario-entity accumulation in reset ~282-294)
- Test: `tests/test_scenario_control.py`

**Interfaces:**
- `set_scenario(scenario_def: ScenarioDefinition) -> None`: replaces `self.scenario_def`, removes entities previously spawned from scenario `obstacle_overrides` (tag them, e.g. `entity.metadata["_scenario_spawned"]=True` or track ids), does NOT reset (caller resets).
- In `reset()`: before appending `obstacle_overrides` entities, drop any prior scenario-spawned entities.

- [ ] Test: apply scenario A w/ obstacles → set_scenario(B) → reset → B's obstacles only, no A leftovers; reset() idempotent (no duplicate entities across repeated resets).

## Task 5: TCP SET_SCENARIO (protocol 2.1)

**Files:**
- Modify: `sim_net/protocol.py` (MessageType + SET_SCENARIO/SET_SCENARIO_ACK; `PROTOCOL_VERSION="2.1"`, `SUPPORTED=("2.0","2.1")`)
- Modify: `sim_net/server.py` (per-conn negotiated version; episode-active tracking; handler)
- Modify: `sim_client/client.py` (`set_scenario(scenario_dict, seed=None, reset=False)`)
- Modify: `sim_client/gym_env.py` (SimGymEnv.set_scenario passthrough)
- Test: extend `tests/test_scenario_control.py`

**Rules:** server tracks `episode_active` (reset→active until terminated|truncated observed in STEP_ACK). SET_SCENARIO w/o `reset:true` while active → ERROR `scenario_update_rejected`. With `reset:true` (or while inactive): validate dict via `ScenarioDefinition.from_dict` (errors → ERROR `invalid_scenario`), `env.set_scenario`, `env.reset(seed=payload.seed)` → ACK `{scenario, obs, info}`. Clients negotiated at 2.0 → ERROR `unsupported_in_protocol_version`.

- [ ] Tests: negotiation 2.0/2.1, success, invalid scenario, mid-episode reject, forced reset, deterministic post-update reset, real headless subprocess E2E.
- [ ] Then relax `validate_contract` curriculum+tcp ban → curriculum over TCP: harness stage-advance for tcp envs calls `env.set_scenario(dict, reset=True)` per env then `runner.set_envs(same list)`; test real curriculum advance over headless TCP.

## Task 6: Learning benchmark framework

**Files:**
- Create: `benchmarks/phase6/benchmark_runner.py`, `benchmarks/phase6/benchmark_config.json`, `benchmarks/phase6/results/` (gitignored outputs), `sim_experiment/convergence.py` (reusable eval/compare logic)
- Create: `docs/PHASE_6_LEARNING_BENCHMARKS.md`
- Test: `tests/test_convergence.py`

**Interfaces:**
- `run_benchmark(config) -> results dict`: phases = untrained-baseline eval → train T1 → eval → train T2 → eval; per phase: `evaluate_policy` on fixed seed set, frozen policy from checkpoint; records config hashes, versions, hardware (platform/cpu count), wall time, SPS, per-seed aggregates.
- `convergence_report(results) -> dict`: improvement vs baseline (delta mean_reward, completion_rate), cross-seed mean/std, stability check (later evals not regressing beyond documented tolerance), explicit verdict strings — thresholds documented in config.

- [ ] Test with dummy/fast trainer path on tiny timesteps; report generation; stored JSON schema.
- [ ] CLI: extend `benchmark` cmd or add `learn-bench --config`. 

## Task 7: Dataset validation + deterministic split

**Files:**
- Create: `sim_experiment/dataset_validation.py`, `sim_experiment/dataset_split.py`
- Modify: `sim_experiment/dataset.py` (fix `_read_episode` type-check; split fp vs schema mismatch counters — additive keys, keep old key for compat)
- Modify: `sim_experiment/trajectory.py` (header gains `env_index`, `episode_seed` — additive, readers tolerant)
- Test: `tests/test_dataset_validation.py`

**Interfaces:**
- `validate_dataset(dataset_dir) -> {valid: bool, errors: [...], stats: {...}}`: format version, manifest fields, schema-hash consistency, obs/action dims+finiteness, episode integrity (steps non-empty, last step terminal).
- `split_dataset(dataset_dir, out_dir, val_frac, test_frac, seed) -> split manifest`: **episode-level** split keyed on `episode_id`, deterministic via sha256(seed+episode_id); manifest records dataset fingerprint, split seed, counts, source_run.
- `dataset_stats(dataset_dir) -> dict`: counts, return/length distributions, termination-reason histogram, scenario/seed distribution.

- [ ] Tests: malformed dataset rejected; schema mismatch rejected; deterministic split identical across calls; no episode leakage between splits.
- [ ] CLI: `dataset-validate`, `dataset-split`, `dataset-stats`.

## Task 8: BC trainer + agent

**Files:**
- Create: `sim_client/agents/bc_baseline.py` (`BCAgent`: MLP, continuous→MSE regression + discrete→CE classification), `sim_experiment/trainers/bc_trainer.py`, `sim_experiment/bc_pipeline.py` (orchestration: validate→split→train→checkpoint→eval)
- Modify: `sim_experiment/capabilities.py` (bc entry), `evaluation.py` (`make_policy_from_checkpoint` gains `"bc"`)
- Test: `tests/test_bc.py`

**Interfaces:**
- `BCTrainer` trains on a *dataset*, not env rollouts — it does not fit the env contract; give it its own run-dir layout `experiments/<exp>/bc_runs/<run_id>/` with `bc_result.json`, `bc_metrics.jsonl`, `checkpoints/bc_{epoch}.pt`. Checkpoint payload: `{model_state_dict, obs_dim, action_type, act_dim|num_actions, normalization stats, dataset_fingerprint, split_fingerprint, epoch, config}`.
- Eval: `evaluate_policy` + frozen BC policy on experiment env; comparison via `compare_runs`-style dict exposed for UI.

- [ ] Tests: end-to-end on a small real exported dataset: validate → split → train (loss decreases on toy data) → checkpoint → resume → eval → metrics. Diagnostic fields never enter features (dataset has none anyway — assert obs dim/schema).
- [ ] CLI: `train-bc --dataset <dir> --experiment <exp> [--epochs --val-frac --seed --eval]`.

## Task 9: Trajectory sampling

**Files:**
- Modify: `sim_experiment/trainers/_harness.py` (`EpisodeTrajectoryRecorder` → buffered, predicate-driven)
- Test: `tests/test_trajectory_sampling.py`

**Interfaces:**
- Config `training.algorithm_config.trajectory_sampling` (dict; `trajectory_episodes` int remains back-compat → `{mode:"first_n", n}`): `{mode: first_n|every_n|probability|episodes|min_return|termination_reasons|all, n, p, seed, episode_ids:[], min_return, termination_reasons:[], scenario_ids:[]}`.
- Recorder buffers per-env episode steps (agent+diagnostic data unchanged); on episode end evaluates predicate → write or drop. Probability mode: `sha256(seed, env_idx, ep_idx)` → uniform — deterministic.

- [ ] Tests: each mode deterministic; filters correct; episode integrity (no partial files); diagnostics still isolated from agent_data.

## Task 10: Worker registry, heartbeats, leases, reclaim

**Files:**
- Create: `sim_experiment/worker_registry.py` (persistent `workers/registry.json` under experiments root)
- Modify: `sim_experiment/remote_worker.py` (protocol "1.1": worker_id in HELLO_ACK, `STATUS` msg; keep "1.0" support), `sim_experiment/scheduler.py` (Worker gains `worker_id,last_heartbeat,lease_expires`; `register_worker`, heartbeat poll in `tick()`, lease expiry → requeue via `worker_crash`; `_poll_jobs` renews lease; duplicate-dispatch guard: never dispatch a job with active lease), `cli.py` (`worker-register`, `worker-list`, `worker-reclaim`, `batch-run --worker host:port --token`)
- Test: `tests/test_worker_hardening.py`

- [ ] Tests: registration persists; HELLO heartbeat updates state; dead worker → OFFLINE + job reclaimed; duplicate dispatch impossible; reconnect same worker_id reuses record; auth failure unchanged; POLL/CANCEL path confinement fix.
- [ ] Root-check POLL/CANCEL `experiment_dir` on the service side (security fix).

## Task 11: UI — charts, workers view, dataset preview

**Files:**
- Modify: `sim_ui/inspector.py` (richer chart row: multi-series + axes; new WORKERS section; dataset preview rows + episode browser), `sim_ui/app.py` (providers/actions)
- Test: `tests/test_ui_services.py` (provider-level data, not pixels)

- [ ] Provider tests: comparison payload feeds multi-series chart data; worker rows from scheduler/registry; dataset stats + episode list.
- [ ] Manual boot check `python main.py` headless-safe.

## Task 12: CLI wiring + legacy doc updates + docs

- `cli.py`: `learn-bench`, `dataset-validate`, `dataset-split`, `dataset-stats`, `train-bc`, `worker-register`, `worker-list`, `batch-run --worker/--token`; `--env-mode process` on launch/batch-run.
- Fix `LEGACY_PATHS_ANALYSIS.md` ppo_train.py drift; write `LEGACY_MIGRATION.md` (deprecation map; keep all, formally deprecate `ppo_train.py` demo path).
- Docs: `PARALLEL_ENVIRONMENTS.md` (rewrite: 3 modes), `TCP_SCENARIO_CONTROL.md`, `IMITATION_LEARNING.md`, `DATASET_VALIDATION.md`, `WORKER_HARDENING.md`, `PHASE_6_OVERVIEW.md`.

## Task 13: Performance measurement + final report

- `benchmarks/phase6/perf_runner.py`: seq vs process @1/2/4(/8) envs → `docs/PHASE_6_PERFORMANCE.md` + raw JSON.
- `docs/PHASE_6_FINAL_REPORT.md` with VERIFIED/PARTIAL/NOT VERIFIED/NOT IMPLEMENTED per claim.
