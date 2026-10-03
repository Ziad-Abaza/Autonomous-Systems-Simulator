# Troubleshooting

Diagnose and fix the most common problems with Simulation Studio. Each entry follows the format **Symptom → Cause → Solution → Verify**. If your problem isn't listed, check the [FAQ](../reference/faq.md) or the [CLI reference](../experiments/cli-reference.md).

---

## Installation & startup

### `python main.py` fails with `ModuleNotFoundError`

- **Symptom:** `ModuleNotFoundError: No module named 'pygame'` (or `moderngl`, `numpy`, …) on launch.
- **Cause:** The repository has **no `requirements.txt`** — dependencies must be installed manually.
- **Solution:** Install the core packages:

  ```
  pip install pygame moderngl glcontext numpy
  ```

  Optional packages, only needed for specific features:

  | Package | Needed for |
  |---|---|
  | `gymnasium` | `SimGymEnv` adapter (falls back to legacy `gym` if installed) |
  | `torch` | `sim_client/agents/ppo_train.py` and all `sim_experiment` trainers (ppo/sac/dqn/bc) |
  | `pyinstaller` | `python build/package_windows.py` standalone build |

- **Verify:** `python main.py --headless` starts and prints `External AI Server Port: 8765`.

### `pkg_resources` UserWarning on startup

- **Symptom:** A `UserWarning: pkg_resources is deprecated...` (or similar) printed when pygame imports.
- **Cause:** Known cosmetic warning from `pygame` under newer Python/setuptools versions.
- **Solution:** None required — it is harmless and does not affect functionality. It can be ignored or filtered with `python -W ignore::UserWarning main.py`.
- **Verify:** The studio window opens normally after the warning.

### Window won't open / GL context errors

- **Symptom:** Crash or exception during startup mentioning `moderngl`, `glcontext`, or "cannot create GL context"; black window that immediately closes.
- **Cause:** Interactive studio mode requires a working GPU/OpenGL context (via `moderngl` + `glcontext`). Remote desktops, VMs without GPU passthrough, and missing graphics drivers can all fail to provide one.
- **Solution:**
  - Update graphics drivers; on Windows, ANGLE and osmesa-backed contexts are known to work (`tools/ui_shots.py` notes "osmesa/angle works").
  - If you only need the simulator as a training server, skip rendering entirely: `python main.py --headless`. Headless mode still hosts the TCP server on `--port`.
- **Verify:** `python tools/ui_shots.py` produces screenshots without errors (it requires a GL context too), or the headless process stays alive and accepts connections.

---

## TCP connection problems

The studio **always** hosts a `SimulationServer` on `127.0.0.1:<port>` at startup — there is no toggle to enable it. Default port is `8765`.

### `ConnectionRefusedError`

- **Symptom:** Client raises `ConnectionRefusedError: [Errno 111/10061] Connection refused`.
- **Cause:** No simulator is listening on that host/port — the studio isn't running, or it's on a different port.
- **Solution:** Start the simulator first: `python main.py` (interactive) or `python main.py --headless` (server only). If you launched with `--port 9000`, connect to 9000. Note the server binds to `127.0.0.1` only — remote hosts cannot connect.
- **Verify:** `python -m sim_client.agents.pid_driver` connects and drives (it defaults to `127.0.0.1:8765`).

### `getaddrinfo failed` / name resolution errors

- **Symptom:** `socket.gaierror: [Errno 11001/11002] getaddrinfo failed`.
- **Cause:** Bad host argument. A frequent cause is passing flags to `pid_driver` — it takes **positional** arguments, not `--host`/`--port` flags. `python -m sim_client.agents.pid_driver --host 127.0.0.1` passes the literal string `--host` as the hostname.
- **Solution:** `python -m sim_client.agents.pid_driver 127.0.0.1 8765`. In your own code use `SimulationClient(host="127.0.0.1", port=8765)` with a resolvable host.
- **Verify:** The client prints the HANDSHAKE_ACK contents (track name, protocol version) on connect.

### Socket timeout on connect or step

- **Symptom:** `socket.timeout` from `SimulationClient` (default `timeout=10.0` s).
- **Cause:** Server is alive but the single connection slot is occupied (single-env server accepts max 1 client; extras sit in the backlog), or a firewall is silently dropping the connection.
- **Solution:** Close the other client — only one external agent can drive an env at a time. For parallel agents use `python main.py --headless --num-envs N` so each client gets its own env slot. You can also raise the client timeout: `SimulationClient(port=8765, timeout=30.0)`.
- **Verify:** `client.connect()` returns the env spec dict instead of timing out.

### `server_full: all N env slots busy`

- **Symptom:** Server replies `{"error": "server_full: all 4 env slots busy"}` and closes the connection.
- **Cause:** You connected more clients than the multi-env server has environments (`SimServerMulti` assigns envs FIFO from a fixed pool; there is no queueing).
- **Solution:** Restart with more slots — `python main.py --headless --num-envs 8 --port 8765` — or wait for a connected client to disconnect (its env returns to the pool).
- **Verify:** Number of simultaneous clients ≤ `--num-envs`; each new client receives HANDSHAKE_ACK.

### Port conflict / "address already in use"

- **Symptom:** Startup fails with `OSError: [Errno 10048] Only one usage of each socket address...` or the client connects to an unexpected simulator instance.
- **Cause:** Another studio/headless instance (or another app) already holds port 8765.
- **Solution:** Pick a different port: `python main.py --headless --port 8800`, and connect your agent to the same port.
- **Verify:** Startup log shows `External AI Server Port: 8800` and the client handshake succeeds on that port.

### `SimGymEnv()` raises immediately

- **Symptom:** `SimGymEnv()` constructor throws a `ConnectionError` before any `reset()`.
- **Cause:** `SimGymEnv.__init__` connects to the simulator **eagerly** — it does not wait for `reset()`. If no server is running, construction itself fails. It also has no `track` parameter; the track is whatever the studio has loaded.
- **Solution:** Always start the simulator **before** constructing the env:

  ```python
  from sim_client.gym_env import SimGymEnv
  env = SimGymEnv(host="127.0.0.1", port=8765)  # requires a running simulator
  ```

- **Verify:** `env.reset()` returns `(obs, info)` without exceptions.

---

## Protocol errors

See [TCP protocol](../agents/tcp-protocol.md) for the full message reference. Error strings below are exact.

### `protocol_version_mismatch: no mutually supported protocol version`

- **Symptom:** HANDSHAKE is answered with ERROR and the connection is refused.
- **Cause:** Your client offered no version in the server's supported set `{2.0, 2.1}`.
- **Solution:** Send `{"type":"HANDSHAKE","payload":{"protocol_versions":["2.1"]}}` (or `["2.0","2.1"]`). Sending no version field at all defaults to the current version (2.1) — it's a *mismatch* that fails, not an absent field.
- **Verify:** HANDSHAKE_ACK returns `protocol_version`, `action_space`, `observation_schema`, `track_name`, `physics_hz: 60`.

### `unsupported_in_protocol_version: SET_SCENARIO requires protocol >= 2.1`

- **Symptom:** SET_SCENARIO fails while other messages work.
- **Cause:** The connection negotiated protocol 2.0, which does not support mid-session scenario switching.
- **Solution:** Reconnect offering `"2.1"` in `protocol_versions`.
- **Verify:** HANDSHAKE_ACK shows `"protocol_version": "2.1"`; SET_SCENARIO_ACK is returned.

### `scenario_update_rejected: episode is active; ...`

- **Symptom:** SET_SCENARIO rejected even on protocol 2.1.
- **Cause:** An episode is mid-flight (steps have run since the last reset) and the request did not ask for a reset.
- **Solution:** Send `"reset": true` in the SET_SCENARIO payload to apply the scenario **and** reset the episode, or call RESET first then SET_SCENARIO before stepping.
- **Verify:** SET_SCENARIO_ACK shows `"episode_state": "reset"` (or `"applied"` when at episode start).

### `unknown_scenario: 'X' is not in the standard scenario library`

- **Symptom:** SET_SCENARIO by `scenario_id` fails.
- **Cause:** The id isn't one of the six built-in scenarios: `basic_lane_following`, `high_speed_racing`, `wet_adverse_weather`, `obstacle_evasion`, `sensor_noise_challenge`, `full_domain_randomization`.
- **Solution:** Use a valid id, or send a full `{"scenario": {...}}` dict instead of `scenario_id`.
- **Verify:** SET_SCENARIO_ACK echoes `scenario.scenario_id` and `name`.

### `Unknown message type: X`

- **Symptom:** Server replies `{"error": "Unknown message type: X"}`.
- **Cause:** The `type` field isn't one of the client→server types: `HANDSHAKE`, `DISCOVER_CONTRACT`, `RESET`, `STEP`, `GET_STATE`, `SET_SCENARIO`. Check spelling, case, and that each line is exactly one JSON envelope `{"type": ..., "payload": {...}}` terminated by `\n`.
- **Solution:** Fix the message type/envelope format (NDJSON — one JSON object per line).
- **Verify:** Server responds with the matching `*_ACK` message.

---

## Agent behavior

### Agent doesn't move, or steers the wrong way

- **Symptom:** Car stays still, spins, or turns opposite to commands.
- **Cause:** The action vector is ordered **[steer, throttle, brake]** — a common bug is sending `[throttle, steer, brake]` or a single scalar. Steering sign: **−1 = left, +1 = right** (scaled by `max_steering_angle`, default 0.58 rad). Throttle and brake are `[0,1]`.
- **Solution:**
  - Send `[-1..1, 0..1, 0..1]` e.g. `[0.3, 0.6, 0.0]`.
  - Mind the default action **dead zones**: steering changes below `0.02` and throttle/brake below `0.01` snap back to the channel default (0). Tiny actions literally do nothing.
  - Rate limits also apply (steer 4.0/s, throttle 5.0/s, brake 6.0/s) — a step change reaches target over multiple 1/60 s steps.
  - If the car reverses into a termination, note `wrong_direction` triggers when |heading error| exceeds **120°**.
- **Verify:** `info["last_action"]` echoes the decoded `[steer, throttle, brake]`; `info["action_valid"]` is `True`; speed rises in `info["speed"]`.

### `invalid_call_after_done` after episode end

- **Symptom:** `step()` returns `terminated=True, truncated=True` with `info["error"]` and `termination_reason = "invalid_call_after_done"`.
- **Cause:** The environment is already done (`is_done`) — calling `step()` again instead of `reset()`.
- **Solution:** Call `reset()` after every terminated/truncated step. `episode_config.auto_reset_on_done` exists in config but is **not consumed** — auto-reset does not happen by itself.
- **Verify:** After `reset()`, steps return real rewards and `termination_reason` is a real rule (`collision`, `max_steps`, …).

### Empty or degenerate observations

- **Symptom:** Lidar slice is all `1.0`s; image channel is all zeros; whole obs looks off.
- **Cause:** Deliberate fallbacks in the observation pipeline, not a crash:
  - No lidar attached → the 15 lidar channels fill with `ones(15)` (= "all clear").
  - Image channel declared but camera missing/disabled → `zeros` `uint8` frame.
  - `NaN` → `0`, `±Inf` → `±1` anywhere in the vector.
- **Solution:** Attach the sensor your observation schema expects (SENSORS tab in the inspector, or `agent.sensor_configs` in the `.sim.json`). For image obs, ensure a camera named `rgb_camera` is enabled or set `image_channels` explicitly.
- **Verify:** `python -m sim_experiment.cli validate-env --project <file.sim.json>` reports no `Observation`/`Sensors` ERRORs; obs values vary as the car moves.

---

## Validation errors

`python -m sim_experiment.cli validate-env --project <file.sim.json>` prints a JSON report and exits `0` when `is_valid_for_rl` is true, `1` otherwise. Each issue has:

- `severity`: `ERROR` (blocks `is_valid_for_rl`), `WARNING`, or `INFO`.
- `subsystem`: `Agent | Track | Spawn | Action | Observation | Reward | Termination | Sensors | Environment`.
- `message` and `remediation`: what is wrong and how to fix it.

In the studio, the VALIDATE inspector tab shows the same report and each issue links to the owning tab.

Common issues:

| Message (paraphrased) | Severity | Fix |
|---|---|---|
| No agent defined / empty `agent_id` | ERROR | Add an `agent` block (loading a legacy file auto-creates one) |
| Fewer than 3 control points / CP pair < 0.5 m | ERROR | Add or spread control points in the track editor |
| `num_checkpoints < 2` | ERROR | Raise `num_checkpoints` (TRACK tab, 4–64) |
| Spawn off road / spawn within 3.5 m of a collidable | ERROR | Move the spawn marker onto the road, away from entities |
| Spawn heading off track tangent > 45° | WARNING | Align spawn yaw with the road direction |
| Discrete action space with < 2 options | ERROR | Add `DiscreteActionOption`s |
| Observation channel `source_sensor` not attached | ERROR | Enable that sensor in SENSORS or remove the channel |
| **SECURITY LEAKAGE** — channel category is `debug_telemetry`/`oracle_ground_truth` | ERROR | Recategorize the channel to `agent_observation`; agents must not see oracle fields |
| Duplicate sensor names | ERROR | Rename sensors so `sensor_configs` names are unique |
| No termination rules / no truncation rule | ERROR / WARNING | Enable rules on the TERM tab (at least one termination + one truncation recommended) |
| Image channel enabled but no camera | ERROR | Attach a `camera_rgb` sensor or disable `include_image_channel` |
| No progress-producing reward component | WARNING | Enable a `progress`/`centerline`/`speed` component (REWARD tab) |
| Track length < 20 m / open-route notice / deterministic 60 Hz stepping | INFO | Informational only |

---

## Experiments & training

### `launch` fails the compatibility check

- **Symptom:** `python -m sim_experiment.cli launch <exp_id>` is rejected before a run starts.
- **Cause:** `check_compatibility` validates the trainer against the manifest: algorithm id, action-space type, image-observation support, and multi-env support.
- **Solution:** Run `python -m sim_experiment.cli trainers` to see the capability table. Typical mismatches:
  - `--algorithm dqn` on a **continuous** action space (DQN is discrete-only) — use `ppo`/`sac` or switch the env to discrete actions.
  - `sac`/`ppo` on a **discrete** action space — only `dqn` (and `bc`) handle discrete.
  - Image observations with `ppo`/`sac`/`dqn`/`bc` — only the `dummy` trainer declares image support.
  - `num_envs > 1` — all four real trainers support `multi_env`, so this is rarely the cause.
- **Verify:** `launch` reaches `RUNNING` (`status <exp> <run>` shows status + timesteps).

### Trainer crashes or the run goes `FAILED`

- **Symptom:** Run status `FAILED`, error `"trainer_crash"`, nonzero exit.
- **Cause:** The trainer subprocess (`python -m sim_experiment.trainers.<name> --run-dir <dir>`) died without writing `run_result.json`.
- **Solution:** Read `experiments/<exp_id>/runs/<run_id>/logs/stdout.log` and `stderr.log` — the Python traceback is there. Common causes: missing `torch`, bad `algorithm_config` JSON, or a `bc` dataset fingerprint mismatch.
- **Retry behavior:** `trainer_crash` **is** in the batch scheduler's retryable set (with `trainer_exception`, `launch_failure`, `timeout`, `worker_crash`). Non-retryable causes include `invalid_contract`, `invalid_curriculum`, `unsupported_algorithm`, `protocol_version_mismatch`, `incompatible_environment` — fix configuration, not retry.
- **Verify:** `run_result.json` exists with `"status": "completed"` and `run.json` shows `COMPLETED`.

### `resume` doesn't continue the same run

- **Symptom:** After `python -m sim_experiment.cli resume <exp_id> <run_id>`, a new run appears.
- **Cause:** By design — resume creates a **new run** whose `resume_from` points at `{parent_run_id, checkpoint}`. The original run is untouched.
- **Solution:** Track the child run via `runs <exp_id>`; it continues from the parent's latest registered checkpoint.
- **Verify:** Child `run.json` has `resume_from.parent_run_id` set; metrics seq continues via `metrics.jsonl`.

### `dataset-export` skips episodes / `train-bc` rejects the dataset

- **Symptom:** `manifest.json` `skipped` counts are high, or `train-bc` fails on fingerprint grounds.
- **Cause:** `dataset-export --env-fingerprint` filters episodes to those recorded against a matching environment fingerprint; episodes from a different environment/track version are skipped. `bc` training requires the dataset fingerprint to **match the experiment's environment** (identity is part of the contract).
- **Solution:** Check `manifest.json` → `skipped` for the reasons/counts, `env_fingerprint` for what was recorded, and `schema_hash`. Export from a run trained on the same environment version you will BC-train against, or omit `--env-fingerprint` only if you intend to train against whatever is in the dataset.
- **Verify:** `dataset-validate <dir>` reports `valid: true`; `dataset-stats` shows expected episode/step counts and a single `env_fingerprint`.

---

## agentRL

### Track files missing / "file tracks" empty

- **Symptom:** `TrackRegistry.default()` can't load `oval`, `serpentine`, or `smoke` file tracks.
- **Cause:** `tracks/*.sim.json` is **gitignored** (`tracks/` and `data/` are in `.gitignore`) — a fresh clone has no track files.
- **Solution:** Open the studio and save a track (`tracks/` is the writable library root), or copy a preset from `presets/` (`oval_circuit.sim.json`, `serpentine_track.sim.json`, `obstacle_challenge.sim.json`) into `tracks/`. Presets are regenerated automatically only if the entire `presets/` directory is missing.
- **Verify:** `ls tracks/*.sim.json` is non-empty; `TrackRegistry.default()` lists the file tracks alongside `gen_loop_*`.

### No agentRL training entry point

- **Symptom:** Looking for `python -m agentRL ... train` fails.
- **Cause:** agentRL (v0.1.0) is partially implemented — `core/`, `obs/`, `act/`, `rewards/`, `envs/`, `memory/` exist, but `algos/`, `train/`, `eval/`, `checkpoints/`, and any CLI do not.
- **Solution:** Use it as a library (`EnvFactory`, `ObservationSpec`, presets), or use the supported path: `python -m sim_experiment.cli create/launch`.
- **Verify:** `tests/agent/` exercises the implemented surface.

---

## Performance problems

See the full guide in [Performance](../performance/performance.md). Quick fixes:

- **Sluggish studio UI:** shrink the window (`--width 960 --height 540`), disable or shrink camera sensors (the camera is the most expensive sensor — offscreen render + readback), and lower `ui_scale` in settings.
- **Training too slow:** run headless — `python main.py --headless` (or `env_mode=process`/`tcp_multi` in experiments). Rendering is the dominant cost; a state-only env steps ~4× faster than the full sensor suite.
- **TCP feels capped:** the headless multi-env poll loop sleeps 0.0005 s per pass → ~2000 polls/s **shared** across all clients, and each env only steps on a `STEP` request.

---

## Datasets & replay

- **Recording missing in the REPLAY picker:** recordings are scanned under `<data_root>/recordings/<name>/episode.json`. `data_root` defaults to `<repo>/data`; if you changed it in SETTINGS, new recordings go to the new root. A legacy `last_episode.json` at repo root is also picked up.
- **Recording won't save:** `data_root` must be writable — `validate_data_root()` write-probes it when you change the setting. Pick another folder in SETTINGS.
- **DATA tab is empty:** it scans `<data_root>`, `experiments/`, and `<repo>/data/experiments` for `manifest.json`/`episode.json(.gz)`. Recordings made before a `data_root` change stay at the old location.

---

## Build (standalone exe)

- **`python build/package_windows.py` fails with "PyInstaller not found":** `pip install pyinstaller` — it's a build-time-only dependency, not needed to run from source.
- **Exe lacks training features:** intentional — the PyInstaller spec excludes `torch`/`torchvision` (and `sim_experiment` is not bundled), so the standalone app has no trainers. Run experiments from source instead.
- **Build is Windows-only:** `package_windows.py` targets Windows; source runs on any platform with Python + the deps above.

---

## Tests

- **`tests/agent/` failures about missing tracks:** the agentRL file tracks (`tracks/*.sim.json`) must exist — they are gitignored, so generate/save them via the studio or copy presets first (see agentRL section above).
- **Suite command:** `python -m pytest tests/ -q` (542 tests at audit time, ~8 min).

---

## See also

- [FAQ](../reference/faq.md) — conceptual questions and capability answers
- [Compatibility](../reference/compatibility.md) — version and protocol reference
- [TCP protocol](../agents/tcp-protocol.md) — wire format and error catalog
- [CLI reference](../experiments/cli-reference.md) — all `sim_experiment.cli` subcommands
- [Performance](../performance/performance.md) — benchmarks and tuning
