---
noteId: "d25b5bf0bf6911f1a29f1fbaabbd87c8"
tags: []

---

# MASTER PROJECT STATUS

Status counts computed from `MASTER_REQUIREMENTS_TRACEABILITY.md` (136 requirements +
5 explicitly superseded). No composite score — factual counts only.

## Executive summary

| Status | Count |
|--------|-------|
| COMPLETE | 103 |
| PARTIAL | 27 |
| IMPLEMENTED-BUT-BROKEN | 4 |
| IMPLEMENTED-BUT-UNVERIFIED | 2 |
| DOCUMENTED-ONLY | 0 |
| DEMO/MOCK/STUB | 0 |
| MISSING | 0 |
| SUPERSEDED | 5 |
| **Total audited** | **136 (+5 superseded)** |

Evidence levels across COMPLETE rows: predominantly B (behavioral tests) + a
meaningful set of A (runtime: fresh vehicle-dynamics suite, live PPO + SAC training
runs, experiment lifecycle via CLI, template drive matrix, open-track repro,
31/31 UI smoke, 15/15 TCP e2e, 93 screenshots).

Two structural findings dominate everything else:

1. **The platform has never produced a learning result.** Fresh evidence this
   audit: a real PPO run (`experiments/exp_38eba258e68bf064`, 4,096 steps) produced
   2 episodes of exactly 1,501 steps each, `max_duration_exceeded`, mean_speed
   0.008–0.013 m/s — a stationary policy earning ~0.79/step (measured idle reward
   0.790/step vs driving 0.831/step). The infrastructure works; the objective
   function permits idling. The agentRL phase is the correct fix — during this audit
   E001 began producing the first-ever movement signal (episodes ending in collision
   at ~2 m/s mean speed, vs `stuck` stationary episodes early in training). Gate pending.
2. **The standalone-package requirement is currently broken** (no `_MEIPASS`,
   data→`_internal/`, torch excluded while UI imports it, `tools/` unpackaged,
   no dist artifact exists).

## Phase-by-phase status

| Phase | Total | COMPLETE | PARTIAL | BROKEN | UNVERIFIED | Overall state |
|-------|-------|----------|---------|--------|------------|---------------|
| 0 — Platform | 19 | 13 | 4 | 2 | 0 | Working platform; packaging broken, adapter/open-track defects |
| 1 — Hardening | 10 | 6 | 2 | 2 | 0 | Contract edge defects remain |
| 2 — Authoring | 11 | 8 | 3 | 0 | 0 | Functional; viz gizmos + list caps missing |
| 3 — Env Designer | 14 | 10 | 4 | 0 | 0 | Strong; dead controls + dead scenario fields |
| 4 — Experiments | 16 | 12 | 4 | 0 | 0 | Verified live this audit |
| 5 — Scale-out | 13 | 12 | 0 | 0 | 1 | Remote workers never exercised |
| 6 — Scalable training | 12 | 9 | 3 | 0 | 0 | Real; learn-bench never executed |
| 7 — Studio UX | 10 | 6 | 4 | 0 | 0 | Functional; theme/polish debt |
| 7.5 — Repair | 12 | 10 | 2 | 0 | 0 | Fixes verified; template completion config |
| VD — Dynamics | 10 | 10 | 0 | 0 | 0 | Validated within stated envelope |
| agentRL | 9 | 7 | 1 | 0 | 1 | T1–T14 committed; **E001 running** — early learning signal (moving+colliding agent) |

## Critical blockers (summary — detail in MASTER_BLOCKERS.md)

**Critical (2):** stationary-policy reward exploit → no demonstrated learning;
standalone packaging non-functional.

**High (5):** HANDSHAKE legacy spaces for authored agents; open-track end/completion
semantics; step-after-done both-flags; headless recording gap; orphaned obs channel
when sensor disabled (validator catches, runtime fills constants).

## Readiness highlights (full matrix in MASTER_PROJECT_AUDIT.md §34)

- Launch/create/author/simulate/record/replay/dataset-ui: YES (A evidence).
- Configure sensors/cameras incl. multi-camera: YES (A, screenshots).
- External AI over TCP: YES mechanically; contract defect for authored agents.
- Run RL training: YES (live PPO + SAC runs).
- Produce a *learning* agent: NO verified evidence anywhere (E001 pending).
- Standalone distribution: NO.

---

# REPAIR-PHASE UPDATE (2026-10-04)

Implementation/repair pass after the baseline audit. Historical counts above
reflect the pre-repair state; the report at `docs/REPAIR_IMPLEMENTATION_REPORT.md`
carries the authoritative post-repair table.

## Post-repair deltas

- Stationary-policy exploit closed (motion-gated rewards) and **learning is now
  demonstrated**: learn-bench baseline −18 → +797 mean return at 5.24 m/s in
  6,000 PPO steps (`benchmarks/phase6/benchmark_results.json`), plus agentRL
  E001c at 9–12 m/s sustained with +207..+694 returns.
- TCP authored-agent handshake serves the real contract; step-after-done is a
  documented sentinel, not a bug.
- Open tracks enforce endpoint bounds and complete; headless recording covers
  externally-driven TCP episodes including body-frame velocity; FIFO'd
  recordings are marked truncated rather than silently dropping frames.
- `cli batch` now delegates to the canonical scheduler; orphaned observation
  channels are refused at build; inspector dead controls (scenario presets,
  weather/time, action min/max, camera roll/near/far, spawn point) are wired.
- Packaging built and executed: `dist/AI_Environment_Simulator` (236.8 MB)
  serves the authored contract over TCP. Torch remains excluded by design —
  the single UI evaluate-from-checkpoint path degrades with a clear message.
- Remote workers verified live on a two-process TCP dispatch (and a real
  port-bind bug was found and fixed in the process).
- Dead scenario fields consumed: legacy obstacles spawn, entity errors surface
  in `info.scenario_warnings`, `target_speed_override` restores its baseline,
  authored ambient/weather/time mirror onto the render path, `global_seed` is
  honored via a per-reset SeedSequence.
- Vehicle dynamics re-validated quantitatively: timestep-independent (30/60 Hz
  within ~2%), near-reference steady sideslip in the normal regime (3.7° at
  0.85 g), bounded at-limit transients.
- HUD theme-completed (82 hardcoded literals → theme tokens); scene list
  uncapped (scrolls); smoke harness green with exit 0.

## Honest open items

- E002–E010 full-scale matrix: in progress (parallel iteration on E001
  variants); E001c proves learning but not converged safe driving at 60k.
- Continual-learning *skill-retention* claim unproven at scale (machinery
  verified end-to-end via a reduced A→B→C run).
- PH0-VEH-005 stays PARTIAL: single-track model documented; 4-wheel/Pacejka
  is a spec overclaim, not a defect to hack.
- Camera near/far/roll affect the GL offscreen path only — the procedural
  2.5D fallback rasterizer ignores them (documented).
- TRAIN/DATA panel per-frame scans and editor gizmos remain PARTIAL.
