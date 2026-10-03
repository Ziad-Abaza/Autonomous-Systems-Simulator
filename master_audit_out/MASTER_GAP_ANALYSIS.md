---
noteId: "0ee40860bf6a11f1a29f1fbaabbd87c8"
tags: []

---

# MASTER GAP ANALYSIS

One entry per non-COMPLETE requirement. Fields: current state → target → missing →
dependencies → recommended implementation → validation required.

---

## GAP-01 — Demonstrated learning (PH2-RL-010 + feeds ARL-009, PH5-TRN-*, PH6-BEN-007)

- **Current:** real trainers + full pipeline verified mechanically; all artifacts (incl. a fresh 4,096-step PPO run this audit) produce stationary policies ≈0.79/step idle reward.
- **Target:** an agent that drives — E001 verdict mean_speed>2 m/s, completions>0.
- **Missing:** training on the anti-exploit reward to a verdict; learn-bench execution; curriculum/batch/BC/remote run artifacts.
- **Dependencies:** agentRL T14/T15; advisable B-H3 contract fix first.
- **Implementation:** land `experiments/matrix.py` (fix `algo` key), E001 running at audit time — early learning signal observed; complete it and run E002–E010, produce convergence verdict + continual report.
- **Validation:** artifacts on disk; `learn-bench` results/; `compare_runs.py` output.

## GAP-02 — Standalone packaging (PH0-PKG-018)

- **Current:** spec+script exist; build cannot produce a correct app.
- **Target:** `package_windows.py` produces a dist that launches on a clean dir; user data outside `_internal`; training path works or fails clean.
- **Missing:** `_MEIPASS`-aware data-root; bundled-presets path fix; torch policy (bundle vs probe-gate); `tools/` packaging or decouple; `datas` for assets; clean-machine verification.
- **Dependencies:** none hard; ideally after contract fixes to avoid re-verifying twice.
- **Implementation:** path-resolution module honoring frozen mode; spec datas/hiddenimports updates; `--no-training` gate or bundled torch; smoke-build script.
- **Validation:** fresh build launches, drives, saves track to user dir, records episode; UI training behaves per chosen policy.

## GAP-03 — TCP contract correctness (PH0-NET-013, PH1-ENV-007)

- **Current:** HANDSHAKE legacy spaces; DISCOVER_CONTRACT correct; SimGymEnv uses HANDSHAKE.
- **Target:** clients receive the effective contract (compiled pipeline spaces).
- **Missing:** single source of truth in HANDSHAKE.
- **Implementation:** env_handler HANDSHAKE branches on agent presence; or SimGymEnv prefers DISCOVER_CONTRACT.
- **Validation:** authored-agent TCP client receives matching spaces; regression test.

## GAP-04 — Open-track end semantics + template completion (PH1-TRK-009, PH75-TPL-012)

- **Current:** `is_on_road=True` past open ends (projection clamps); `term_completion` disabled in all 7 templates; `empty`/`slalom` never terminate.
- **Target:** finish terminates (or explicit past-end boundary); open templates shippable-correct.
- **Missing:** `past_end`/`off_course` semantic; completion enabled in open templates; validator warning for open+completion-off.
- **Implementation:** track_queries past-end check OR termination rule; template regeneration with `term_completion enabled:true` on open routes; validator rule.
- **Validation:** open-route episode terminates at finish; regression on overshoot repro.

## GAP-05 — Gym contract edge (PH1-ENV-004)

- **Current:** step-after-done → terminated=True AND truncated=True.
- **Target:** single-flag (or documented raise).
- **Implementation:** pick one flag; update ACTION_SCHEMA doc; adjust trainers' bootstrap expectations.
- **Validation:** contract test; trainer truncation-bootstrap unit test.

## GAP-06 — Sensor disable → orphaned channel (PH75-SEN-007)

- **Current:** dead channel filled `[1.0]*15`; validator ERRORs but runtime tolerates.
- **Target:** either refuse to build (fail-fast) or auto-prune channel + warn.
- **Validation:** env build fails or prunes; validator ERROR retained.

## GAP-07 — Recording/replay integrity (PH0-REC-015, PH7-REC-006)

- **Current:** GUI-only step recording; replay zeroes vy; silent FIFO drop; dest not persisted; missing termination_reason sidecar.
- **Target:** headless TCP steps recorded; state-faithful replay; truncation marked.
- **Implementation:** move `_record_step_if_needed` into env/server step path; store vel_body in frames; truncation marker + sidecar field; persist last record dir.
- **Validation:** headless server records external steps; replay reproduces drift states; unit tests.

## GAP-08 — UI dead controls + unreachable capabilities (PH1-UI-010, PH3-DSN-003, PH3-UI-011, PH7-DAT-008)

- **Current:** scen_* enum/weather/time rows + act_min/max rows have no handlers; sp_*/vc_* handlers have no emitters; spawn yaw/elev/speed uneditable; dataset export only in TRAIN.
- **Target:** every row mutates state or is removed; spawn params editable; export reachable from DATA.
- **Implementation:** wire handlers both directions; emit sp_/vc_ rows; DATA export action.
- **Validation:** smoke check per control; no dead clicks dirty the doc.

## GAP-09 — Scenario dead fields (PH0-ENV-011, PH3-DSN-006)

- **Current:** weather/time/ambient + legacy obstacles + global_seed serialized-dead; target_speed leaks; entity-create errors swallowed; RANDOM_CHECKPOINT dead.
- **Target:** no silently-dead serialized field.
- **Implementation:** mark fields presentation-only or consume; restore target_speed on set_scenario; log entity errors; implement/remove enum.
- **Validation:** serialization audit test asserting each field consumed-or-documented.

## GAP-10 — Sensor depth (PH0-SEN-007, PH75-SEN-009 residuals)

- **Current:** multi-camera works; camera lacks near/far/roll; fallback ignores pose/FOV; PiP>3 overflow; no preview.
- **Validation:** rear camera renders rear view headless; >4 cams don't overflow.

## GAP-11 — Curriculum/batch/remote/dataset artifacts (PH5-CUR-001, PH5-BAT-007, PH5-WRK-009, PH5-DAT-011, PH6-BEN-007)

- **Current:** machinery verified by tests; zero production artifacts.
- **Target:** one artifact each: curriculum run, batch_result, remote-worker run, exported dataset, learn-bench results.
- **Validation:** artifacts on disk + CLI flows exercised.

## GAP-12 — Contract hygiene medium batch (PH6-VEC-006, PH0-NET-012, PH4-CLI-013, PH4-PPO-015)

- **Items:** dirty-env recycle; rx-buffer cap + timeout; `cli batch`; PPO act_dim/unclamped/eval-clips; dual-path key drift decision; `get_state` msg_type.
- **Validation:** per-item tests.

## GAP-13 — Theme/accessibility/polish (PH7-THM-002, PH7-*)

- **Current:** 5 palettes exist; HUD/editor hardcoded; ui_scale fonts-only; no keyboard nav; unbounded text paths; terminology drift; per-frame scans.
- **Validation:** light-theme screenshots in all tabs; bounded-text audit; keyboard-only dialog flow.

## GAP-14 — agentRL completion (ARL-007, ARL-009)

- **Current:** T1–T13 committed; T14 failing in flight; no continual run; E001 pending.
- **Validation:** continual A→B→C run with retention numbers; experiment matrix executed; E001 gate.
