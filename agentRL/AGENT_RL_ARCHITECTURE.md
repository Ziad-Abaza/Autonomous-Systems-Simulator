---
noteId: "4ad42280bf6311f1a29f1fbaabbd87c8"
tags: []

---

# AGENT_RL Architecture

**Status:** Design — pending implementation
**Depends on:** `agentRL/AGENT_RL_ARCHITECTURE_AUDIT.md` (all claims below cite it)
**Approved decisions:** hybrid reuse of existing infra · SAC primary / PPO
supported · moderate compute budget · docs at `agentRL/` root

---

## 1. Purpose & Non-Goals

`agentRL` is the agent-side platform for learning autonomous driving on this
simulator. It owns: agent abstractions, algorithms, observation/action
specification, reward presets, multi-track/continual training, evaluation,
failure analysis, and the experiment matrix.

It does **not** own: physics, sensors, track geometry, reward *computation*,
termination *evaluation*, protocol, experiment manifest/scheduler infra —
those remain in `sim_*` packages and are consumed as libraries.

**Non-goals (this iteration):** pixel-based RL (camera obs supported in the
pipeline but CNN experiments are ablations only — headless fallback rasterizer
is low-fidelity and CPU-bound), recurrent policies (frame-stack config exists;
LSTM deferred pending evidence), distributed training beyond local
ProcessVectorEnv, EWC/regularized continual learning (deferred until forgetting
is measured, not assumed).

---

## 2. Governing Constraints (from audit)

1. CPU-only torch → small MLPs, off-policy sample efficiency matters,
   ProcessVectorEnv for throughput.
2. ~350 SPS/env in-process → experiment sizes in the 50k–200k steps/phase range.
3. Default reward config is provably exploitable (stationary policy = +0.8/step)
   → all training uses `agentRL` reward presets, never the template defaults.
4. Declarative `AgentDefinition` path only — no legacy engine configs.
5. Policy consumes AGENT_OBSERVATION channels only; GET_STATE/info dict are
   diagnostics for logging/analysis, never policy input.
6. `terminated or truncated` = done; bootstrap only on pure truncation.
7. Env obs for policy must be identical across tracks (same ObservationSpec) —
   track identity never enters obs.

---

## 3. Package Layout

```
agentRL/
  __init__.py
  core/
    agent.py            BaseRLAgent ABC
    config.py           AgentConfig, TrainConfig, EvalConfig dataclasses
    versioning.py       AGENT_VERSION, schema versions, compat checks
    seeding.py          deterministic seed tree (global/env/track/worker)
  envs/
    track_registry.py   TrackSpec registry + load from tracks/*.sim.json
    factory.py          EnvFactory → configured SimulationEnvironment
    scenario_gen.py     seeded ScenarioMutator (random obstacles, spawn, friction)
    vecenv.py           thin adapters: SyncVecEnv / ProcessVecEnv wrappers
  obs/
    spec.py             ObservationSpec (channel selection, frame_stack,
                        prev_action) → policy-input layout descriptor
    encoder.py          ObsEncoder: env obs → normalized np/tensor input
  act/
    adapter.py          ActionAdapter: validation, [-1,1]→env bounds map,
                        prev-action tracking (env already rate-limits)
  rewards/
    presets.py          declarative RewardFunctionDefinition presets
  algos/
    ppo.py              PPOAgent (CleanRL-style, adapted from proven baseline)
    sac.py              SACAgent (twin-Q, auto-α; primary)
  memory/
    replay.py           ReplayBuffer + TrackRehearsalBuffer (per-track mixing)
  train/
    trainer.py          Trainer: single-phase training loop, metrics, ckpt
    continual.py        ContinualTrainer: phase sequence + retention probes
  eval/
    evaluate.py         evaluate_policy over an env factory + seeds
    matrix.py           EvalMatrix: tracks × checkpoint → metric grid
    failures.py         failure classification from termination + episode stats
  checkpoints/
    io.py               save/load/resume + schema-version gate
  experiments/
    matrix.py           E001-E010 experiment registry + runner
    configs/            JSON experiment configs
  tests helpers are in tests/agent/ (outside the package, per spec)
AGENT_RL_ARCHITECTURE_AUDIT.md      (done)
AGENT_RL_ARCHITECTURE.md            (this doc)
AGENT_RL_TRAINING.md                (written when training infra lands)
AGENT_RL_CONTINUAL_LEARNING.md      (written when continual lands)
AGENT_RL_EVALUATION.md              (written when eval matrix lands)
AGENT_RL_FAILURE_ANALYSIS.md        (written when classifier lands)
AGENT_RL_EXPERIMENTS.md             (results log, updated per experiment)
AGENT_RL_FINAL_REPORT.md            (Stage 12)
```

---

## 4. Core Interfaces

```python
# core/agent.py
class BaseRLAgent(ABC):
    algo_id: ClassVar[str]
    def __init__(self, cfg: AgentConfig, obs_spec: ObservationSpec,
                 act_low: np.ndarray, act_high: np.ndarray, device: str): ...
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def begin_episode(self) -> None: ...            # resets hidden/prev-action state
    def observe(self, obs, action, reward, next_obs, done, truncated, info) -> None: ...
    def update(self, step: int) -> dict[str, float]: ...   # {} when no update
    def save(self, path: Path, extra: dict) -> None: ...
    def load(self, path: Path) -> dict: ...
```

```python
# envs/track_registry.py
@dataclass(frozen=True)
class TrackSpec:
    track_id: str            # "oval", "serpentine", "holdout_a", ...
    project: dict            # loaded .sim.json (road_definition + agent contract)
    tags: tuple[str, ...]    # ("train",) ("holdout",) ("obstacle",)

# envs/factory.py
class EnvFactory:
    def __init__(self, reward: str, termination: str | None, obs_channels: ObservationSpec): ...
    def build(self, track: TrackSpec, seed: int,
              mutator: ScenarioMutator | None = None) -> SimulationEnvironment: ...
```

`EnvFactory.build` clones the project dict, injects the reward preset +
termination rules + sensor suite into `project["agent"]`, loads
`EnvironmentProject`, returns the env. Mutator wraps episode resets: before
each `env.reset(seed=k)` it draws a seeded `ScenarioDefinition` (obstacle
placement, spawn override, friction/noise multipliers) and applies it —
in-process via the env's scenario mechanism (same semantics as SET_SCENARIO;
`tests/agent/` validates the TCP path separately).

```python
# obs/spec.py
@dataclass(frozen=True)
class ObservationSpec:
    vector_channels: tuple[str, ...]     # e.g. ("state8","lidar15") → 23 dims
    image: bool = False
    frame_stack: int = 1
    prev_action: bool = False
    @property
    def input_dim(self) -> int: ...      # vector_dim*stack + act_dim*prev_action
```

```python
# rewards/presets.py  (concrete starter set — tuned in E001)
DRIVE_V1 = components:
    progress    +2.0     # dominant term
    speed       +0.5     # min(1, v/15) — target lowered to attainable
    centerline  +0.1     # shaping only
    heading     +0.1
    smooth_steer -0.02
    checkpoint  +5.0
    completion  +100.0
    collision   -50.0
    off_road    -25.0
    reverse     -1.0
    time_penalty -0.05   # idling is strictly negative vs +0.8 baseline
TERMINATION_V1: collision, off_road, wrong_direction → terminate;
                checkpoint_timeout(20 s), max_steps(1500) → truncate;
                lap completion → terminate.
```

Stationary math after the fix: idle ≈ −0.05 + shaping ≤ 0.15/step (if
shaping fires at all) vs +0.8 baseline — idling is a losing strategy by
construction, not by tuning luck.

---

## 5. Algorithms

### SAC (primary — `algos/sac.py`)
Tanh-squashed Gaussian actor (2×256 ReLU), twin Q (2×256), auto-α
(target entropy = −act_dim), τ=0.005, γ=0.99, buffer 200k, warmup 2k,
batch 256, update-per-step 1. Action mapped tanh-out → env bounds via
ActionAdapter. Basis: `sim_client/agents/sac_baseline.py` math (already
tested in-repo), reorganized under BaseRLAgent with general act_dim,
obs_spec-driven input, checkpoint schema v1.

### PPO (secondary — `algos/ppo.py`)
Separate actor/critic 2×128 tanh, diag Gaussian, GAE(λ=.95), clip .2,
value clip, entropy .01, truncation bootstrap. Basis:
`sim_client/agents/ppo_baseline.py` (its GAE/truncation correctness is
already covered by tests/test_ppo_multienv.py); act_dim generalized.

Both: CPU device, `torch.manual_seed` from the seed tree, NaN guard on
loss (abort update + log `nan_guard` event), grad-norm clip.

---

## 6. Memory / Rehearsal

`ReplayBuffer` — fixed circular, seeded sampling (adapts
`sim_client/agents/replay_buffer.py`; done = terminated only).

`TrackRehearsalBuffer` — K per-track buffers with per-phase capacity;
`sample(batch)` draws `rehearsal_fraction` (default 0.25) from previous
tracks' buffers + remainder current. This is the continual-learning
mechanism for SAC. PPO (on-policy) instead gets periodic re-evaluation +
optional short "rehearsal fine-tune" phases — measured, not assumed.

---

## 7. Continual Learning Protocol

```python
ContinualTrainer(train_sequence=[
    Phase(track="oval",       steps=100_000, mutator=none),
    Phase(track="serpentine", steps=150_000, mutator=spawn_jitter),
    Phase(track="hairpin",    steps=150_000, mutator=obstacles),
])
```

- After each phase: checkpoint `phase_<i>_<track>.pt`; run EvalMatrix over
  **all tracks seen so far + holdout probes** (frozen policy, N seeded
  episodes each) → append to `continual_report.json`.
- Metrics per cell: mean_return, completion_rate, collision_rate,
  off_road_rate, mean_progress, mean_speed, steer_smoothness.
- Derived: `forgetting[track] = best_eval_before − eval_after_latest_phase`;
  `transfer = holdout_metric_after − holdout_metric_before`;
  `retention = eval_after / eval_right_after_learning`.
- Rehearsal buffer carries across phases (SAC) within total capacity.
- Multi-track mixed mode: `TrackSampler` assigns each env slot a track
  (round-robin or weighted); same code path, sequence = one phase with
  mutator=track_sampler.

This answers spec §3's six questions mechanically — every claim in the
final report cites `continual_report.json` cells.

---

## 8. Evaluation & Failure Analysis

`eval/evaluate.py` — `evaluate_policy(factory, track, agent, seeds, n)` →
per-episode + aggregate records (extends sim_experiment's metric set with
progress/speed/smoothness).

`eval/matrix.py` — grid over {tracks} × {checkpoint phases} → JSON grid +
markdown summary.

`eval/failures.py` — classifier on termination_reason + episode stats:
`collision_obstacle | collision_barrier | off_track | stall_timeout |
wrong_direction | oscillation` (steer-variance + low progress heuristic) |
`timeout_progress` (survived but low progress) | `completed`.
Each failure row: track, seed, step, speed, lat_offset, reason,
reward_breakdown tail. Stored `failures.jsonl` per eval — "why did it fail"
is a query, not a guess.

---

## 9. Checkpoints & Versioning

`checkpoints/io.py` — `torch.save` payload:
`{agent_version, schema_version, algo_id, obs_spec, action_spec, nets,
optimizers, replay_meta {sizes per track}, train_state {step, episode,
phase}, seeds, env_fingerprints, reward_preset, termination_preset}`.
Load refuses on `schema_version` mismatch or obs/action spec mismatch —
no silent partial loads (spec §29). `resume_training()` restores optimizers
+ step counters + rehearsal metadata.

Versions: `AGENT_VERSION` semver; `OBS_SPEC_V`, `ACT_SPEC_V`,
`CKPT_SCHEMA_V` integers.

---

## 10. Experiment Matrix (moderate budget)

| ID | Goal | Tracks | Algo | Steps | Pass evidence |
|---|---|---|---|---|---|
| E001 | learn to drive | oval | SAC+PPO | 60k each | mean_speed ≫ 0, progress > 0, beats stationary return; failure-class mix vs baseline |
| E002 | curves | serpentine | SAC | 120k | completion or progress gain vs E001 start |
| E003 | obstacles | oval + rand obstacles | SAC | 120k | collision rate ↓ vs untrained; not memorized positions (per-seed layouts) |
| E004 | mixed multi-track | oval+serpentine | SAC | 150k | both tracks improve |
| E005 | sequential continual A→B→C | 3 tracks | SAC | 3×100k | retention matrix: forgetting bounded, report emitted |
| E006 | generalization | holdout | SAC | eval-only | nonzero progress/completion vs random/PID refs |
| E007 | sensor noise | oval × noise mults | SAC | eval-only | graceful degradation curve |
| E008 | recovery | offset spawns | SAC | eval-only | recovery_rate vs random |
| E009 | ablations | oval | SAC | 40k each | state8 vs state8+lidar vs +prev_action vs frame_stack |
| E010 | resume correctness | oval | SAC | 2×30k | resumed == continuous (metric equivalence) |

Baselines every eval compares against: `random`, `pid_driver` (strong
reference), `stationary` (documents the exploit the reward fix kills).

---

## 11. tests/agent/ Laboratory

```
tests/agent/
  conftest.py            in-process env fixtures, track fixtures, seeds
  test_obs_spec.py       channel layouts, frame_stack, prev_action
  test_action_adapter.py bounds/map/validation
  test_reward_presets.py component wiring, stationary math ≤ 0 proof
  test_replay.py         capacity, sampling, rehearsal fractions
  test_checkpoint.py     save/load/resume equivalence, version refusal
  test_env_factory.py    track load, mutator determinism, obs contract
  test_env_tcp.py        E2E headless server round-trip (marked slow)
  test_failures.py       classifier mapping
  test_continual.py      retention matrix math on synthetic evals
  run_policy.py          drive the sim (TCP or in-process) with a checkpoint
  evaluate_policy.py     eval CLI → JSON
  compare_runs.py        metric comparison table
  inspect_episode.py     dump per-step obs/action/reward breakdown for an episode
```

Plus a `manual_vs_agent.py` for the spec's human-vs-agent comparison path
(reuses sim's existing manual WASD mode + recording).

---

## 12. Implementation Phases

1. **Skeleton**: package, versioning, config, seeding, obs/act specs + tests
2. **Env layer**: track registry, factory, reward/termination presets, scenario
   mutator + tests — *exit: factory builds envs for 3 tracks deterministically*
3. **SAC + PPO agents** under BaseRLAgent + replay + tests — *exit: E001*
4. **Trainer**: loop, metrics.jsonl, checkpoint/resume — *exit: E010*
5. **Eval suite**: evaluate, matrix, failures — *exit: eval JSON on baseline ckpts*
6. **Multi-track + ContinualTrainer + rehearsal** — *exit: E004/E005 reports*
7. **Obstacle mutator + experiments E002/E003/E007/E008**
8. **Ablations E009, docs set, final report**

Each phase: tests first where the contract is testable (TDD), real numbers
recorded in AGENT_RL_EXPERIMENTS.md as they land.

---

## 13. Explicit Risk Register

- **Reward rebalancing may need 1-2 iterations** — E001 exists precisely to
  measure it; presets are config, cheap to revise.
- **CPU throughput may force smaller step budgets** — report actual SPS and
  scale phases while keeping eval matrices complete.
- **SAC on this env is unproven here** — PPO fallback exists; E001 runs both.
- **In-process mutator vs SET_SCENARIO parity** — `tests/agent/test_env_tcp.py`
  pins the TCP semantics; in-process path verified against it.
- **Existing working tree has uncommitted sim changes** — `agentRL` touches
  no `sim_*` files; all integration via public APIs only.
