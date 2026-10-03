---
noteId: "e3bae560bf6311f1a29f1fbaabbd87c8"
tags: []

---

# agentRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production continual-learning RL driving platform in `agentRL/` that demonstrably learns to drive (fixing the proven stationary-policy exploit), trains across multiple tracks, measures retention/forgetting/generalization, and produces a full reproducible experiment matrix.

**Architecture:** agentRL owns the agent-side stack — `BaseRLAgent` ABC, SAC (primary) + PPO agents, ObservationSpec/ActionAdapter, reward/termination presets, EnvFactory + seeded ScenarioMutator, replay + track-rehearsal memory, Trainer/ContinualTrainer, eval matrix + failure classifier, versioned checkpoints. It consumes `sim_env`, `sim_project`, `sim_experiment.headless`, `sim_client` as libraries and never modifies sim internals.

**Tech Stack:** Python 3.13, PyTorch 2.10 (CPU), numpy, pytest. No new dependencies.

**Spec:** `agentRL/AGENT_RL_ARCHITECTURE.md` (design) + `agentRL/AGENT_RL_ARCHITECTURE_AUDIT.md` (evidence). User master task = governing requirements.

## Global Constraints

- CPU-only: MLPs ≤ 2×256; SAC buffer ≤ 200k; batch 256.
- Only AGENT_OBSERVATION channels feed the policy; GET_STATE/info are diagnostics only.
- `terminated or truncated` = done; bootstrap value only on pure truncation.
- No edits inside `sim_*/` packages; integration via public APIs only (`build_env_from_dicts`, `env.set_scenario`, `env.reset/step`, `env.track.spline` for mutator placement).
- Declarative AgentDefinition path only — never legacy engine configs.
- Run artifacts under `agentRL/runs/` (added to .gitignore in Task 1); code + docs are tracked.
- `python -m pytest tests/agent -q` is the lab test command.
- Same ObservationSpec across all tracks in an experiment; track identity never enters obs.

## Review Focus

1. **Stationary exploit recurrence** — reward presets must make idle return ≤ ~0; test pins the math analytically AND a short live rollout.
2. **Truncation bootstrap** — time-limit ends (checkpoint_timeout/max_steps) must bootstrap, not zero-out; collision/off_road must not. Tested per algorithm.
3. **Step-after-done both-flags-true** — env returns terminated+truncated simultaneously on invalid calls; trainer must not double-count or bootstrap wrongly.
4. **Seeded mutator determinism** — same seed must reproduce identical obstacle layouts; different seeds must differ. Tested.
5. **Checkpoint resume equivalence** — resumed run must continue counters/optimizer/replay state, not silently restart; version/spec mismatch must refuse.

---

### Task 1: Package skeleton, versioning, seeding, config

**Files:**
- Create: `agentRL/__init__.py`, `agentRL/core/__init__.py`, `agentRL/core/versioning.py`, `agentRL/core/seeding.py`, `agentRL/core/config.py`
- Modify: `.gitignore` (append `agentRL/runs/` under a new "agentRL artifacts" comment)
- Test: `tests/agent/test_core.py`, `tests/agent/conftest.py`

**Interfaces:**
- Produces: `AGENT_VERSION = "0.1.0"`, `CKPT_SCHEMA_V = 1`, `OBS_SPEC_V = 1`, `ACT_SPEC_V = 1`; `VersionError`; `seed_tree(base_seed) -> dict(env=..., track=..., torch=..., mutator=...)`; `AgentConfig(algo_id, hidden_sizes, lr, gamma, extra: dict)`; `TrainConfig(total_steps, eval_interval, ckpt_interval, num_envs, seed, run_dir)`; `set_global_seeds(seed)` seeds random/np/torch.

- [ ] **Step 1: Write failing tests** — `test_seed_tree_deterministic` (same base → same derived ints, different base → different), `test_version_constants_exist`, `test_set_global_seeds_reproducible` (np.random draws identical after reseed).
- [ ] **Step 2: Run tests, verify failures** — `python -m pytest tests/agent/test_core.py -v` → module-not-found errors.
- [ ] **Step 3: Implement** — dataclasses + seed derivation via `np.random.SeedSequence.spawn`; `set_global_seeds` sets `random.seed`, `np.random.seed`, `torch.manual_seed`.
- [ ] **Step 4: Verify pass** — same pytest command green.
- [ ] **Step 5: Commit** — `git add agentRL tests/agent .gitignore; git commit -m "feat(agentRL): core scaffolding — versioning, seeding, config"`

### Task 2: ObservationSpec + encoder

**Files:**
- Create: `agentRL/obs/__init__.py`, `agentRL/obs/spec.py`, `agentRL/obs/encoder.py`
- Test: `tests/agent/test_obs_spec.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class ObservationSpec:
    channel_names: tuple[str, ...]   # subset of default channel names, order = layout
    frame_stack: int = 1
    prev_action: bool = False
    image: bool = False              # reserved; encoder raises NotImplementedError with clear msg
    @property
    def vector_dim(self) -> int      # sum of channel shapes
    @property
    def input_dim(self) -> int       # vector_dim*frame_stack + (3 if prev_action)
    def to_dict/from_dict
DEFAULT_CHANNELS: dict[str, int]     # name → flat width (from create_default_space)
PRESETS = {"state8": (...7 state channels...), "full23": (...all 8...)}
```
`ObsEncoder(spec)` → `reset() -> np.ndarray`; `encode(obs_flat, prev_action=None) -> np.ndarray` (frame deque concat); pure numpy.

- [ ] **Step 1: Failing tests** — `test_spec_dims` (state8→8, full23→23, +prev_action→+3, frame_stack 4→*4), `test_encoder_stack_order` (oldest→newest concat), `test_channel_subset_validation` (unknown name → ValueError), `test_image_reserved` (image=True → NotImplementedError on encode).
- [ ] **Step 2-4:** run-fail / implement / verify-pass cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): observation spec + encoder`

### Task 3: ActionAdapter

**Files:**
- Create: `agentRL/act/__init__.py`, `agentRL/act/adapter.py`
- Test: `tests/agent/test_action_adapter.py`

**Interfaces:**
- `ActionAdapter(low=np.ndarray, high=np.ndarray)` → `to_env(action_raw)` maps agent space [-1,1]^3 → env bounds via per-dim affine (throttle/brake are [0,1]); `from_env(action_env) -> [-1,1]^3`; `validate` (finite check → clamp, returns (action, valid)); `reset()` clears prev_action; `prev_action` property for obs encoder.

- [ ] **Step 1: Failing tests** — `test_roundtrip` (to_env∘from_env ≈ identity), `test_bounds` (raw -1→low, +1→high, 0→mid), `test_nan_inf_handling` (NaN→valid=False, replaced with zeros).
- [ ] **Step 2-4:** cycle. **Step 5: Commit** — `feat(agentRL): action adapter`

### Task 4: Reward + termination presets

**Files:**
- Create: `agentRL/rewards/__init__.py`, `agentRL/rewards/presets.py`
- Test: `tests/agent/test_reward_presets.py`

**Interfaces:**
- `reward_preset(name) -> RewardFunctionDefinition`; `termination_preset(name) -> TerminationDefinition`; `PRESETS` registry.
- `DRIVE_V1` components (types from `reward_designer`): progress +2.0; speed +0.5 `{"target_speed_ms": 15.0}`; centerline +0.05 `{"max_distance_m": 6.0}`; heading +0.05; smooth_steer −0.02; checkpoint +5.0; completion +100.0; collision −50.0; off_road −25.0; reverse −1.0; time_penalty −0.12.
- `TERM_V1`: collision/off_road/wrong_direction(120°)/course_completion(1 lap) terminate; checkpoint_timeout `{"max_seconds": 20.0}` + max_steps `{"max_steps": 1500}` truncate.

- [ ] **Step 1: Failing tests** — `test_stationary_return_nonpositive` (instantiate `CompiledRewardEngine(DRIVE_V1)`; feed `delta_s=0, lat=0, road=12, speed=0, head=0` for 100 steps → accumulated ≤ 0.0 — pins the exploit fix analytically); `test_driving_beats_idle` (same engine, speed=15 aligned → step reward > idle step reward); `test_preset_components_valid` (each component_type in the known closed set).
- [ ] **Step 2-4:** cycle — tune weights inside test bounds only if assertions fail.
- [ ] **Step 5: Commit** — `feat(agentRL): reward/termination presets (stationary exploit fix)`

### Task 5: TrackRegistry + EnvFactory

**Files:**
- Create: `agentRL/envs/__init__.py`, `agentRL/envs/track_registry.py`, `agentRL/envs/factory.py`
- Test: `tests/agent/test_env_factory.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class TrackSpec: track_id: str; project: dict; tags: tuple[str,...]
TRACK_FILES = {"oval": "tracks/basic_driving_proving_ground.sim.json",
               "serpentine": "tracks/lane_following_serpentine_circuit.sim.json",
               "smoke": "tracks/smoke_test.sim.json"}
class TrackRegistry: load(track_id) -> TrackSpec; list() -> [ids]; register(track_id, path, tags)
class EnvFactory:
    def __init__(self, reward="drive_v1", termination="term_v1",
                 obs_spec=PRESETS["full23"], sensor_names=("vehicle_state","lidar_rays"))
    def build(track: TrackSpec, seed: int) -> SimulationEnvironment
```
`build` deep-copies `track.project`, writes `agent.observation_space` (default channels filtered/reordered to `obs_spec.channel_names`), `agent.reward_function`, `agent.termination_rules`, trims `sensor_configs` to `sensor_names` (all `enabled`), then `build_env_from_dicts(project_dict, seed=seed)`.

- [ ] **Step 0 (investigation, part of Step 1):** inspect `tracks/*.sim.json` with the env validator — pick 3rd/4th usable track for continual phases + holdout (fallback: clone `smoke_test` geometry variants via `road_definition` edits in tests, not shipped tracks).
- [ ] **Step 1: Failing tests** — `test_factory_builds_env` (oval → env, obs vector dim == spec.vector_dim, reset works), `test_reward_preset_installed` (env.reward breakdown keys ⊇ preset component_ids), `test_obs_channel_subset` (state8 spec → 8-dim obs), `test_deterministic_reset` (two envs same seed → identical first obs), `test_registry_lists_tracks`.
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): track registry + env factory`

### Task 6: ScenarioMutator (seeded obstacle/spawn/friction randomization)

**Files:**
- Create: `agentRL/envs/scenario_gen.py`
- Test: `tests/agent/test_scenario_gen.py`

**Interfaces:**
```python
class ScenarioMutator:
    def __init__(self, seed: int, n_obstacles=(0,3), entity_types=("cone","barrier"),
                 lateral_frac=(-0.4,0.4), s_range=(0.2,0.95), min_gap_m=15.0,
                 spawn_jitter=(-2.0,2.0), friction=(0.85,1.1), noise=(0.8,1.5))
    def draw(env) -> ScenarioDefinition        # uses env.track.spline.sample_at_distance
    def apply(env, scenario) -> None           # env.set_scenario(scenario)
```
Placement: `s ~ U(s_range)*track_len`, `lat ~ U(lateral_frac)*half_width` → world pos via spline point + left-normal, `yaw ~ U(0,2π)`; enforce pairwise |Δs| ≥ min_gap (resample up to 10 tries). Output uses `obstacle_overrides` dicts `{name, entity_type, pos:[x,y,z], yaw}` + `spawn_override`/`surface_friction_mult`/`sensor_noise_mult`.

- [ ] **Step 1: Failing tests** — `test_deterministic` (same seed → identical `to_dict()`), `test_different_seeds_differ`, `test_obstacles_on_road` (each pos projects back: |lateral| ≤ half_width via `env.track_queries.query_vehicle_pose`-equivalent or spline closest point), `test_apply_spawns_entities` (env reset → `_scenario_spawned` entities present, count == n drawn), `test_min_gap_respected`.
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): seeded scenario mutator`

### Task 7: Replay + TrackRehearsalBuffer

**Files:**
- Create: `agentRL/memory/__init__.py`, `agentRL/memory/replay.py`
- Test: `tests/agent/test_replay.py`

**Interfaces:**
- `ReplayBuffer(capacity, obs_dim, act_dim, seed)`: `add(o,a,r,no,term,trunc)`, `sample(batch)->dict of np`, `__len__`. done flag stored = `term and not trunc` (truncation bootstraps).
- `TrackRehearsalBuffer(per_track_capacity, rehearsal_fraction=0.25, seed)`: `add(track_id, transition)`; `begin_track(track_id)` rotates active buffer; `sample(batch)` → fraction from union of *previous* tracks' buffers, rest from current; `state_dict()/load_state_dict()` (sizes + per-track counts only, not arrays).

- [ ] **Step 1: Failing tests** — `test_buffer_capacity_eviction`, `test_sample_shapes`, `test_done_flag_semantics` (truncated transition marked non-terminal), `test_rehearsal_fraction` (with 2 tracks, sample(1000) → ~25% carry tag from track A), `test_state_roundtrip`.
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): replay + rehearsal buffers`

### Task 8: BaseRLAgent + SACAgent

**Files:**
- Create: `agentRL/core/agent.py`, `agentRL/algos/__init__.py`, `agentRL/algos/sac.py`
- Test: `tests/agent/test_sac.py`

**Interfaces:**
```python
class BaseRLAgent(ABC):  # as designed in ARCHITECTURE §4
class SACAgent(BaseRLAgent):
    algo_id = "sac"
    # actor: Linear(obs_dim→256→256→{mu,log_std}), twin Q: (obs+act)→256→256→1
    # cfg.extra: buffer=200_000, warmup=2000, batch=256, lr=3e-4, tau=0.005,
    #            target_entropy=-act_dim, updates_per_step=1, grad_clip=10.0
```
`act` returns env-bounds-space action via adapter owned by caller; internally agent works in [-1,1] via tanh-squash. `observe` stores to replay (buffer or rehearsal). `update` → per-step critic/actor/α updates post-warmup; returns losses dict or {}.

- [ ] **Step 1: Failing tests** — `test_act_bounds` (100 random acts → |a|≤1), `test_observe_and_update_smoke` (fill warmup+100, update returns finite losses, no NaN), `test_checkpoint_roundtrip` (same obs → same deterministic act after save/load; optimizer state restored flag), `test_truncation_bootstrap` (Q target uses (1-done) where done excludes truncation — assert via crafted transition), `test_nan_guard` (poisoned batch → update logs `nan_guard`, params unchanged).
- [ ] **Step 2-4:** cycle. Reference math: `sim_client/agents/sac_baseline.py` (log_std clamp [-5,2], auto-α).
- [ ] **Step 5: Commit** — `feat(agentRL): SACAgent`

### Task 9: PPOAgent

**Files:**
- Create: `agentRL/algos/ppo.py`
- Test: `tests/agent/test_ppo.py`

**Interfaces:**
- `PPOAgent(BaseRLAgent)`, `algo_id="ppo"`, `is_on_policy = True`.
- Actor 2×128 tanh diag-Gaussian (learnable log_std), critic 2×128 tanh.
- `observe` appends to rollout store (obs, act, logp, val, rew, done-flag, trunc-flag); `update()` no-ops until `steps_per_rollout` transitions → GAE(λ .95) → 4 epochs × minibatch 256, clip .2, value clip, entropy .01, adv norm; truncation bootstrap via stored next-value (final_values pattern from `ppo_baseline.py:447-529`).
- `collect_rollout(envs) -> int` helper OR trainer drives stepping — plan choice: **rollout collection lives in OnPolicyTrainer** (Task 10) to keep agent env-agnostic; `observe` just stores.

- [ ] **Step 1: Failing tests** — `test_rollout_buffer_fills`, `test_update_after_rollout` (losses finite, params changed), `test_gae_truncation_value` (truncated end bootstraps V(next_obs), terminated end does not — numeric check on 1-step synthetic rollout), `test_checkpoint_roundtrip`.
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): PPOAgent`

### Task 10: Trainer (off-policy + on-policy loops), metrics, checkpoints

**Files:**
- Create: `agentRL/train/__init__.py`, `agentRL/train/metrics.py`, `agentRL/train/trainer.py`, `agentRL/checkpoints/__init__.py`, `agentRL/checkpoints/io.py`
- Test: `tests/agent/test_trainer.py`, `tests/agent/test_checkpoint.py`

**Interfaces:**
- `MetricsLogger(run_dir)` → append-only `metrics.jsonl` rows `{seq, ts, timestep, scope, metrics}` (NaN→null), `episode()`/`update()`/`eval()` helpers.
- `OffPolicyTrainer(factory, track, agent, train_cfg, mutator=None)` → `train()` loop: reset→act→step→observe→update; auto-reset on done; per-episode metrics (return, len, mean_speed from info, termination_reason); periodic `eval_fn` + `save`.
- `OnPolicyTrainer` — vector env (SyncVectorEnv over N factories), rollout collection, same metrics/checkpoint surface.
- `save_checkpoint(agent, path, extra)` writes schema-v1 payload; `load_checkpoint(path)` → refuses on schema/spec mismatch (`VersionError`); `resume_training(run_dir)` reconstructs agent + counters + rehearsal state.

- [ ] **Step 1: Failing tests** — `test_metrics_jsonl` (write/parse roundtrip, NaN→null), `test_trainer_smoke` (200 steps on smoke track → metrics file has ≥1 episode or timeout row, ckpt exists), `test_checkpoint_version_refusal` (mutate schema_version → VersionError), `test_resume_continues_counters` (timestep restored, not 0), `test_step_after_done_guard` (trainer never calls step post-done — count env calls).
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): trainers + metrics + checkpoints`

### Task 11: Eval suite — evaluate, matrix, failures

**Files:**
- Create: `agentRL/eval/__init__.py`, `agentRL/eval/evaluate.py`, `agentRL/eval/failures.py`, `agentRL/eval/matrix.py`
- Test: `tests/agent/test_eval.py`, `tests/agent/test_failures.py`

**Interfaces:**
- `evaluate_policy(factory, track, agent, seeds=(42..46), mutator=None) -> {per_episode[], aggregate{mean_return, completion_rate, collision_rate, off_road_rate, timeout_rate, mean_progress, mean_speed, steer_smoothness}}` — frozen `deterministic=True` policy; progress from `info["lap_progress"]`, smoothness = std(Δsteer) via `info["last_action"]`.
- `classify_failure(episode_record) -> str`: map termination_reason → {collision→`collision_barrier|collision_obstacle` (via is_colliding vs obstacle flag in info where available, else `collision`), off_road→`off_track`, wrong_direction→`wrong_direction`, checkpoint_timeout|max_duration→`stall_timeout` if mean_speed<0.5 else `timeout_progress`, course_completed→`completed`, invalid_call→`protocol_error`} + `oscillation` post-check (steer sign flips > 30% of steps & low progress).
- `EvalMatrix.run(checkpoints:list, tracks:list) -> dict grid` → writes `eval_matrix.json`.
- `baseline_policies()` → `{"random": fn, "stationary": fn, "pid": fn}` adapters (pid wraps `sim_client.agents.pid_driver` logic in-process — if not importable for in-process env, mark PID as optional baseline behind import guard).

- [ ] **Step 1: Failing tests** — `test_evaluate_returns_aggregate` (random policy on smoke track → keys present, rates in [0,1]), `test_classify_failure_mapping` (synthetic records → expected class each), `test_matrix_grid_shape` (2 ckpts × 2 tracks → 2×2 cells), `test_failure_rows_written` (failures.jsonl contains step/reason/state).
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): eval suite + failure analysis`

### Task 12: ContinualTrainer + retention report

**Files:**
- Create: `agentRL/train/continual.py`
- Test: `tests/agent/test_continual.py`

**Interfaces:**
```python
@dataclass Phase: track_id: str; steps: int; mutator: ScenarioMutator|None; tag: str=""
class ContinualTrainer:
    def __init__(self, factory, agent, train_cfg, holdout_tracks=[...])
    def run(phases: list[Phase]) -> Path  # continual_report.json
```
Per phase: (re)configure envs on phase track, `agent.memory.begin_track(track_id)` (SAC), train `phase.steps` via OffPolicyTrainer, checkpoint `phase_{i}_{track}.pt`, then EvalMatrix over all seen tracks + holdouts → append cell metrics + derived `forgetting`/`transfer`/`retention` (formulas per ARCHITECTURE §7) into `continual_report.json`.

- [ ] **Step 1: Failing tests** — `test_report_schema` (synthetic 2-phase run on smoke envs, tiny steps → JSON has cells for each seen×phase + holdout + derived fields), `test_forgetting_math` (hand-fed evals → correct delta), `test_resume_mid_sequence` (run killed after phase 1 → resume completes phase 2 + report has both).
- [ ] **Step 2-4:** cycle.
- [ ] **Step 5: Commit** — `feat(agentRL): continual trainer + retention report`

### Task 13: tests/agent laboratory scripts

**Files:**
- Create: `tests/agent/run_policy.py`, `tests/agent/evaluate_policy.py`, `tests/agent/compare_runs.py`, `tests/agent/inspect_episode.py`, `tests/agent/test_env_tcp.py`

**Interfaces (CLI):**
- `run_policy.py --ckpt P --track oval --episodes 3 [--tcp PORT]` — drives in-process env (or SimGymEnv over TCP), prints per-episode summary.
- `evaluate_policy.py --ckpt P --track T [--seeds 42-46] --out eval.json`
- `compare_runs.py run_dir_a run_dir_b` — metric table from metrics.jsonl.
- `inspect_episode.py --ckpt P --track T --seed 42 --out ep.jsonl` — per-step obs/action/reward_breakdown dump.
- `test_env_tcp.py` — launches `HeadlessSimProcessPool(1)`, SimGymEnv round-trip: reset → 50 random steps → asserts 5-tuple contract + `reward_breakdown` + SET_SCENARIO applies (marked `slow`).

- [ ] **Step 1: Failing test** — `test_env_tcp` round-trip.
- [ ] **Step 2-4:** implement + verify (manual run of each script on a smoke ckpt).
- [ ] **Step 5: Commit** — `feat(agentRL): agent laboratory scripts`

### Task 14: Experiment matrix + E001 real training

**Files:**
- Create: `agentRL/experiments/__init__.py`, `agentRL/experiments/matrix.py`, `agentRL/experiments/configs/*.json`, `agentRL/AGENT_RL_EXPERIMENTS.md`
- Test: `tests/agent/test_experiment_matrix.py`

**Interfaces:**
- `python -m agentRL.experiments.matrix --exp E001 [--steps-override N]` — resolves config → ContinualTrainer/Trainer → writes `agentRL/runs/E001/` + updates `AGENT_RL_EXPERIMENTS.md` results table programmatically appended JSON (`experiments/results/E001.json`).
- Configs per ARCHITECTURE §10: E001 oval SAC+PPO 60k; E002 serpentine 120k; E003 obstacles 120k; E004 mixed 150k; E005 sequential 3×100k; E006-E008 eval-only; E009 ablations 40k; E010 resume 2×30k.

- [ ] **Step 1: Failing test** — `test_experiment_config_loads` (all 10 configs parse, reference valid tracks/presets), `test_e001_dry_run` (steps-override 500 → run dir + metrics + ckpt produced).
- [ ] **Step 2-4:** implement.
- [ ] **Step 5: RUN E001 FOR REAL** — SAC 60k steps on oval; record SPS, episode returns/speeds/progress into EXPERIMENTS.md; compare vs `stationary` baseline. If policy still degenerate → iterate reward preset (Task 4) within test bounds, rerun. Evidence requirement: mean_speed > 2 m/s AND mean_progress > 0 at final eval, else document failure honestly.
- [ ] **Step 6: Commit** — `feat(agentRL): experiment matrix + E001 results`

### Task 15: Remaining experiments + docs + final report

**Files:**
- Create: `agentRL/AGENT_RL_TRAINING.md`, `agentRL/AGENT_RL_CONTINUAL_LEARNING.md`, `agentRL/AGENT_RL_EVALUATION.md`, `agentRL/AGENT_RL_FAILURE_ANALYSIS.md`, `agentRL/AGENT_RL_FINAL_REPORT.md`

- [ ] **Step 1:** Run E002 (curves) → record. Tune only via presets if needed.
- [ ] **Step 2:** Run E003 (obstacles) → record; verify randomized layouts (different seeds → different collisions pattern).
- [ ] **Step 3:** Run E004 + E005 → `continual_report.json` — extract retention/forgetting/transfer numbers.
- [ ] **Step 4:** Run E006-E008 eval-only suites on final E005 checkpoint.
- [ ] **Step 5:** Run E009 ablations (scaled) + E010 resume check.
- [ ] **Step 6:** Write docs set — actual implementation only, real numbers, honest limitations.
- [ ] **Step 7:** Write FINAL_REPORT with the spec's 24 required sections, exact test counts (`pytest --collect-only -q | tail -1`), exact experiment metrics.
- [ ] **Step 8: Commit** — `feat(agentRL): experiments E002-E010 + final report`

---

## Execution Order & Evidence Gates

T1→T5 sequential (foundations). T6-T9 parallelizable by file but sequential in practice. T10 needs T7-T9. T11-T13 need T10. T14 starts real training (gate: E001 must show movement evidence before proceeding). T15 closes.

**Hard evidence gate between T14 and T15:** E001 must produce `mean_speed > 2 m/s` and `mean_progress > 0` on final eval OR the report documents precisely why not (with failure-classification evidence). No proceeding on vibes.
