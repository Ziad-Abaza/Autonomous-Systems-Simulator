# Standalone Windows Build

How to package Simulation Studio as a self-contained Windows executable with PyInstaller — and what is (and is not) inside the bundle.

## Build

```powershell
python build/package_windows.py
```

This drives PyInstaller with the equivalent of:

```powershell
python -m PyInstaller --name=AI_Environment_Simulator --noconfirm --onedir --windowed `
    --distpath=dist --workpath=build_temp <hidden imports> <excludes> main.py
```

Output layout:

```text
dist/AI_Environment_Simulator/
├── AI_Environment_Simulator.exe    # entry point (windowed — no console)
├── _internal/                      # bundled Python runtime + all packages
├── presets/                        # *.sim.json track presets (written post-build)
└── README.md                       # copied from the repo root
```

Post-build steps performed by the script (not by PyInstaller): `sim_project.presets.save_default_presets()` writes the three default preset `.sim.json` files into the bundle's `presets/` directory, and `README.md` is copied alongside the exe.

`AI_Environment_Simulator.spec` (repo root) mirrors the script's PyInstaller arguments (`console=False`, `upx=True`, same hidden imports and excludes). You can build via `pyinstaller AI_Environment_Simulator.spec` instead — but the spec has no post-build step, so `presets/` and `README.md` will not be copied into `dist/`; `main.py` regenerates presets only if the `presets/` directory is entirely missing, which is the case in a spec-only build.

## Bundled modules

Hidden imports (`build/package_windows.py` → `hidden_imports`):

| Import | Why it's there |
|---|---|
| `moderngl`, `glcontext`, `OpenGL`, `OpenGL.GL` | renderer stack |
| `pygame` | windowing, input, 2D UI |
| `numpy` | core math |
| `cv2` | camera sensor image handling |
| `PIL`, `shapely`, `scipy` | declared hidden imports — **not actually imported by the codebase**; they only add size |
| `gymnasium` | `SimGymEnv` adapter |
| `sim_core`, `sim_env`, `sim_net`, `sim_client`, `sim_ui`, `sim_render`, `sim_recorder`, `sim_project` | all first-party packages the studio uses |

Excluded modules (`excludes`): `torch`, `torchvision`, `tensorflow`, `tensorboard`, `pandas`, `pyarrow`, `botocore`, `boto3`, `openpyxl`, `weasyprint`, `pdf2image`, `pypdf`, `PyPDF2`, `matplotlib`, `IPython`, `notebook`, `jupyter`, `sympy`, `transformers`, `ultralytics`, `unsloth`, `peft`, `scikit_learn`, `sklearn`, `seaborn`, `pytest` — large data-science/ML libraries deliberately kept out of the bundle.

## What the bundle cannot do — read this before shipping

The exclusion list and hidden-import list have real functional consequences:

- **No torch trainers.** `torch` is excluded, so the PPO/SAC/DQN/BC trainers and the `sim_client` baseline libraries cannot run inside the bundle. Any code path that imports them will fail.
- **No experiment platform.** `sim_experiment` is **not** in `hiddenimports`, so the experiment/training subsystem (TRAIN inspector tab, datasets pipeline, orchestration) is not packaged. Experiment launch/training requires a source checkout with torch installed.
- **Dead weight.** `glcontext`, `PIL`, `shapely`, `scipy`, and `OpenGL` are bundled despite never being imported — they inflate bundle size but are harmless.

## Target machine requirements

| Requirement | Notes |
|---|---|
| Windows | `--windowed` build; exe is `AI_Environment_Simulator.exe` |
| GL-capable GPU + driver | required for the interactive studio (ModernGL context). Headless mode does not open a window |
| Python / runtimes | none — the `_internal/` dir carries the interpreter and all bundled deps |
| Network | loopback TCP only (`127.0.0.1`) for the agent server |

## Verifying a build

```powershell
# Studio smoke test — launch the exe, confirm home screen renders
.\dist\AI_Environment_Simulator\AI_Environment_Simulator.exe

# Headless smoke test — env server on port 8765, then connect a client
.\dist\AI_Environment_Simulator\AI_Environment_Simulator.exe --headless --port 8765
python -m sim_client.agents.pid_driver 127.0.0.1 8765   # from a source env, or any NDJSON peer
```

The headless path exercises `SimulationStudioApp(headless=True)` + `SimulationServer`, so it validates sim_core/sim_env/sim_net inside the bundle without needing a GL context. In `--windowed` mode `sys.stdout`/`sys.stderr` are `None`; `main.py` redirects them to `os.devnull`, so expect no console output from the exe.

## Limitations

| Limitation | Detail |
|---|---|
| Windows only | `package_windows.py` is Windows-oriented; no macOS/Linux packaging exists |
| No training in the bundle | `torch` excluded + `sim_experiment` not hidden-imported (see above) |
| One env per studio instance | the embedded `SimulationServer` serves a single env; multi-env serving needs `--num-envs N` headless (`SimServerMulti`) |
| Unused hidden imports | `glcontext`/`PIL`/`shapely`/`scipy`/`OpenGL` ship despite being unreferenced |
| Spec ≠ script for assets | `AI_Environment_Simulator.spec` lacks the presets/README copy step |
| `upx=True` | UPX compression is enabled; some AV scanners flag UPX-packed binaries |
| `tests/`, `tools/`, `benchmarks/` not bundled | dev-only; use a source checkout for QA harnesses |

## See also

- [Development guide](development.md) — setup, tests, extension points
- [Architecture](../architecture/architecture.md) — package map and process model
- [Installation](../getting-started/installation.md)
- [Experiment CLI reference](../experiments/cli-reference.md) — source-install-only features
- [Troubleshooting](../troubleshooting/troubleshooting.md)
- [Performance](../performance/performance.md)
