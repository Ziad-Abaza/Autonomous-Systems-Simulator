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
