# Documentation Audit

Audit of the curated documentation system (`README.md` + `docs/<topic>/`
+ `docs/README.md`), performed after the rewrite. Method: automated
link/image/anchor scan across all 29 doc pages + README, source-code
cross-checks of commands, settings, error strings, and feature claims,
and visual inspection of generated screenshots.

**Result: all machine-checkable items pass. Open items are listed
below.**

## Broken links

**None.** 333 internal links, image references, and anchors resolve
(link scan run on 2025-build; 8 bad anchors were found and fixed during
the audit — see Fixes applied).

## Missing screenshots

**None.** All 39 referenced screenshots exist under `docs/screenshots/`
or `assets/screenshots/` and were generated from the current
application (simulator 4.0.0). 11 additional unreferenced screenshots
remain in `docs/screenshots/` as spare captures
(`editor-overview`, `editor-serpentine`, `simulation-obs-inspector`,
`simulation-wide`, `tab-*` ×5, `theme-light-simulation`).

## Undocumented or partially documented features

Features that exist in code but have thin or no user-facing docs:

| Feature | State | Why thin |
|---|---|---|
| `agentRL` experiment matrix (`E001–E010`, `python -m agentRL.experiments.matrix --exp E001`) | Functional standalone runner | Documented in training.md as an early-stage library; per-config docs do not exist |
| `learn-bench` CLI subcommand | Works | Covered in cli-reference + experiments.md at flag level; no deep guide |
| Remote worker protocol (`worker-serve`/`worker-register`) | Works | Documented in cli-reference; the worker wire protocol itself is not specified |
| `benchmarks/` scripts | Work | Listed in performance.md reproduce table |
| `sim_render` internals | Works | Covered at architecture level only — no renderer API doc |
| Widget layer (`sim_ui/widgets.py`) | Works | Documented as an extension point; no widget catalog |
| Trajectory sampling modes (7 modes + 3 strategies) | Works | Listed in datasets.md; per-mode trade-offs not expanded |

## Questionable claims — verified or corrected

| Claim | Resolution |
|---|---|
| "390+ tests" (old README) | Corrected — measured `pytest --collect-only`: **542 tests**; full suite run: **542 passed** (~8 min) |
| "4-wheel tire model" (code comment) | Corrected in docs — implementation is a single-track (per-axle) model with friction-ellipse saturation |
| `noteId` frontmatter | Injected by the editor on file write; stripped from all docs in a cleanup pass |
| Performance numbers (330–350 steps/s etc.) | Sourced from `benchmarks/*` result files, not re-measured in this pass — treated as representative, flagged in performance.md |
| Extra screenshot helper script | `.notebook/extra_shots.py` printed all saves + clean shutdown but exited code 1; cause undiagnosed — output images verified visually |
| pygame `pkg_resources` deprecation warning | Harmless upstream warning; noted in troubleshooting |

## Commands not verified this pass

Everything in the docs was either run in this session or read directly
from `argparse`/source. Items documented from source but **not executed
end-to-end**:

| Command | Basis |
|---|---|
| `python -m sim_experiment.cli launch/evaluate/...` (full experiment lifecycle) | Verified `--help` + source; no live training run executed (torch training takes minutes–hours) |
| `python -m agentRL.experiments.matrix --exp E001` | Source-verified (`__main__` exists); not executed |
| `python build/package_windows.py` | Build script + spec verified; previous `dist/` artifacts inspected, build not re-run this pass |
| `tools/smoke_interactions.py`, `tools/tcp_recording_e2e.py` | Listed from source; earlier session evidence per audit notes, not re-run now |
| `benchmarks/*.py` | Listed; results read from stored JSON |

## Settings not verified

No unverified settings found — every documented setting was checked
against `sensor_config.py`, `road_config.py`/`track_config.py`,
`vehicle.py`, `action_space`/`observation_space`/`reward_designer`/
`termination_designer`/`scenario_designer` defaults, and
`studio_settings.py`. Documented-but-inert fields are explicitly flagged
as such in [Compatibility](reference/compatibility.md) and
[Project Schema](configuration/project-schema.md) rather than silently
dropped.

## Fixes applied during the audit

- 8 bad markdown anchors → corrected to real heading slugs.
- Test count `481`/`480` → `542` in README + 3 docs.
- `agentRL` described as "no algos/CLI" → corrected to reflect current
  `algos/`, `train/`, `eval/`, `experiments/matrix.py`.
- Bogus "T key opens widget library" claim removed.
- `docs/master_audit/` added to `.gitignore` (internal audit files).
- `noteId` frontmatter stripped from all docs.

## Known documentation gaps

- No API reference (autodoc-style) — the codebase has no docstring
  convention to extract; extension points are documented with file/line
  pointers instead.
- No versioning/changelog doc — simulator 4.0.0 has no CHANGELOG.
- No contributor workflow doc beyond `development.md` (no PR template
  exists in this repo).
- Remote-worker wire protocol is CLI-documented but not spec'd.
- Flat legacy docs (`docs/*.md` phase reports/audits) are intentionally
  untracked and remain as-is; they may contain stale claims — the
  curated tree is the authoritative documentation.
- `.notebook/` contains audit fact sheets and helper scripts used to
  produce this documentation (working artifacts, untracked).
