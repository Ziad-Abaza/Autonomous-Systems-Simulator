# External Agents

How to connect your own AI to Simulation Studio. The simulator **always**
hosts a TCP server on `127.0.0.1` (default port **8765**) — in interactive
studio mode and in headless mode alike. Your agent process connects, runs
the `reset → step` loop in lockstep, and disconnects. The environment never
advances on its own; one `STEP` message = one 60 Hz physics step.

## Starting the server

```powershell
# Interactive studio (GUI) — serves 1 env while the app runs
python main.py --port 8765

# Headless single env — no window, physics-only
python main.py --headless --port 8797

# Headless multi-env — one process, one port, N env slots
python main.py --headless --num-envs 4 --port 8797 --track oval
```

`--track` accepts `oval`, `serpentine`, `obstacle`, or a path to a
`.sim.json` project. Multi-env mode assigns envs to clients FIFO — the first
N connections each get a private env; a further client gets
`server_full: all N env slots busy` and is disconnected. There is no `env_id`
on the wire; a disconnect returns the env to the pool.

## Python SDK — `SimulationClient`

`sim_client/client.py` is a zero-dependency (numpy only) client:

```python
from sim_client.client import SimulationClient

client = SimulationClient(host="127.0.0.1", port=8765, timeout=10.0)
spec = client.connect()      # HANDSHAKE → env spec dict
obs, info = client.reset(seed=42)
obs, reward, terminated, truncated, info = client.step([0.0, 0.3, 0.0])
state = client.get_state()   # diagnostic snapshot
contract = client.discover_contract()   # full declarative contract
obs, resp = client.set_scenario("wet_adverse_weather", reset=True)
client.close()
```

`connect()` returns the `HANDSHAKE_ACK` payload:

| Key | Contents |
|-----|----------|
| `protocol_version` | Negotiated version (`"2.1"`) |
| `supported_versions` | `["2.0", "2.1"]` |
| `action_space` | `{"type", "continuous_low", "continuous_high", "discrete_actions"}` |
| `observation_schema` | `include_*` flags, `flatten_vector` |
| `vector_dim` | Flat observation size (23 by default) |
| `track_name` / `track_length` | Current road definition |
| `physics_hz` / `dt` | 60 / 0.01667 |

Method summary: `connect()` → spec dict (`ConnectionError` on non-ACK) ·
`reset(seed, options)` → `(obs float32 ndarray, info)` ·
`step(action)` — ndarray, list, or int — → 5-tuple ·
`set_scenario(scenario, seed, reset)` → `(obs_or_None, resp)`; accepts a
scenario dict, a standard-library `scenario_id` string, or a
`ScenarioDefinition` object (protocol ≥ 2.1; `reset=True` applies atomically
mid-episode, otherwise rejected while an episode is active) ·
`get_state()` → diagnostic dict `{speed, pos, yaw, sim_time, total_reward,
reward_breakdown}` (returns payload even on ERROR) ·
`discover_contract()` → full `CONTRACT_ACK` · `close()`.

## Gymnasium adapter — `SimGymEnv`

`sim_client/gym_env.py` wraps the client in a standard `gym.Env`
(`import gymnasium`, falls back to `import gym`):

```python
from sim_client.gym_env import SimGymEnv

env = SimGymEnv(host="127.0.0.1", port=8797)   # connects in __init__
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

- **Connects immediately** in `__init__` — construction fails if no server
  is listening. There is no `track` parameter; the track is whatever the
  server loaded.
- `action_space`: `Discrete(len(discrete_actions))` when the env is
  discrete, else `Box(continuous_low, continuous_high, float32)` — default
  `Box([-1,0,0],[1,1,1])`.
- `observation_space`: `Box(-inf, inf, (vector_dim,), float32)` when flat;
  `Dict{"vector": Box, "image": Box(0,255,(84,84,3),uint8)}` when
  `include_camera_rgb`; a per-channel `Dict` (`speed`, `vel_body`,
  `yaw_rate`, `steering_angle`, `distance_from_center`, `heading_error`,
  `distance_to_checkpoint`, `lidar_ranges`, `camera_rgb`) when
  `flatten_vector=False`.
- Full 5-tuple `reset()`/`step()` API — SB3/CleanRL compatible.
- `set_scenario(scenario, seed=None, reset=True)` — note the default
  `reset=True` (vs. `False` on `SimulationClient`) so stage transitions
  apply atomically.
- `metadata = {"render_modes": ["human"]}` — but **`render()` is not
  implemented**. To watch the agent, run the simulator interactively
  (non-headless) and connect; the 3D window is the render stream.

### Minimal working example

```python
import numpy as np
from sim_client.gym_env import SimGymEnv

env = SimGymEnv(host="127.0.0.1", port=8797)
obs, info = env.reset(seed=0)
total = 0.0

for step in range(2000):
    action = env.action_space.sample()          # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        print(f"done: {info['termination_reason']}  return={total:.1f}")
        print(f"breakdown: {info['reward_breakdown']}")
        obs, info = env.reset()
        total = 0.0

env.close()
```

## Included agents

| Module | What it does | Run command | Deps |
|--------|--------------|-------------|------|
| `pid_driver.py` | Closed-loop lane tracking — steering PID(0.85, 0.005, 0.05) on `lat·0.35 + heading·0.75 + lidar_bias·0.2`; speed PID → target `10.5 + 8.5·curvature` (~11 m/s cruise); decodes `obs[0]*45`, `obs[5]*half_width`, `obs[6]*π`, `obs[8:23]*40` | `python -m sim_client.agents.pid_driver [host] [port]` (positional, 1000 steps) | numpy |
| `random_agent.py` | Random baseline — 300 steps of `[steer±0.5, throttle 0.2–0.8, brake 0]`, auto-reset on done | `python -m sim_client.agents.random_agent` (no args; 127.0.0.1:8765) | numpy |
| `ppo_train.py` | Minimal PPO **rollout demo** — ActorCritic samples actions over `SimGymEnv`; **no update code — not a trainer** | `python -m sim_client.agents.ppo_train` | torch |
| `ppo_baseline.py` | CleanRL-style PPO: ActorCritic 128×128 Tanh; `PPORunner` (lr 3e-4, γ 0.99, GAE λ 0.95, clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5, 1024 steps, 4 epochs, batch 256, CPU, hooks, resume) | library only — used by `sim_experiment` trainers | torch |
| `sac_baseline.py` | SAC continuous: actor 256×256 (log-std −5..2, tanh), twin Q 256×256; `SACRunner` (lr 3e-4, τ 0.005, buffer 100k, warmup 1k, batch 256, α 0.2, auto-entropy) | library only | torch |
| `dqn_baseline.py` | DQN discrete only: QNet 128×128 ReLU; `DQNRunner` (lr 1e-3, buffer 50k, warmup 1k, batch 128, train_freq 4, target_update 500, ε 1.0→0.05 over 10k) | library only | torch |
| `replay_buffer.py` | `ReplayBuffer(capacity, obs_dim, act_dim, discrete_actions, seed)` circular numpy buffer; `done` = terminated only (truncation bootstraps) | — | numpy |

All runners accept a single env, an env list, or a `VectorEnv`, plus
`set_envs()` for curriculum and `on_step`/`on_update` hooks.

## What an external agent can and cannot access

**Can:** the observation vector/dict, `reward`, `terminated`, `truncated`,
the full `info` dict (`speed`, `lateral_offset`, `heading_error`,
`road_width`, `is_on_road`, `is_colliding`, `checkpoints_passed`,
`laps_completed`, `termination_reason`, `reward_breakdown`, `last_action`,
…), `GET_STATE` diagnostics (`pos`, `yaw`, `sim_time`, `total_reward`), and
the env spec / declarative contract.

**Cannot:** environment internals beyond `info`/`GET_STATE` — no direct
access to the physics state, entity list, or track spline — and **no render
stream** (the video lives in the studio window, not on the socket).

## Other languages

The wire protocol is newline-delimited JSON over TCP — any language with a
socket library works. Send `{"type": "...", "payload": {...}}` lines; read
one JSON response per request. See [TCP Protocol](tcp-protocol.md) for the
message catalog and a raw-socket example.

## Multiple environments for training

- **Ad-hoc:** `python main.py --headless --num-envs N --port P` — N envs in
  one process on one port; N clients each get a private env (FIFO).
- **Experiments:** `env_mode` picks the parallelism strategy —
  `inprocess` (default), `process` (one `mp.Process` per env), `tcp`
  (N headless processes / N ports), `tcp_multi` (1 process, 1 port, N envs).
  See [Training](../reinforcement-learning/training.md#environment-modes-env_mode).

## Example session

Real output — headless server plus the PID driver (track: default oval):

```text
$ python main.py --headless --port 8797
=================================================================
      3D AI ENVIRONMENT SIMULATION PLATFORM
      Domain: Autonomous Vehicle Reinforcement Learning
=================================================================
Mode: Headless Training
External AI Server Port: 8797
Initial Environment: studio home (oval)
```

```text
$ python -m sim_client.agents.pid_driver 127.0.0.1 8797
[PID Driver] Connecting to simulator at 127.0.0.1:8797...
[PID Driver] Connected! Track: Proving Ground Circuit, Length: 342.0m, Physics: 60.0Hz
[PID Driver] Environment reset. Starting autonomous drive loop for 1000 steps...
Step    0 | Speed:   0.0 m/s | LatErr: -0.00m | HeadErr:  -0.0° | Reward:    0.8 | Laps: 0
Step   60 | Speed:   4.5 m/s | LatErr: -0.03m | HeadErr:  -3.1° | Reward:   51.6 | Laps: 0
Step  120 | Speed:   8.2 m/s | LatErr: -0.32m | HeadErr:  -4.1° | Reward:  109.5 | Laps: 0
Step  180 | Speed:  10.0 m/s | LatErr: -0.28m | HeadErr:   0.2° | Reward:  170.3 | Laps: 0
Step  240 | Speed:  10.7 m/s | LatErr: -0.19m | HeadErr:  -0.8° | Reward:  243.6 | Laps: 0
Step  480 | Speed:  11.9 m/s | LatErr: -0.07m | HeadErr:  -0.1° | Reward:  523.7 | Laps: 0
Step  720 | Speed:  11.7 m/s | LatErr: -0.13m | HeadErr:  -1.1° | Reward:  811.9 | Laps: 0
Step  960 | Speed:  11.0 m/s | LatErr: -0.08m | HeadErr:   0.6° | Reward: 1090.5 | Laps: 0
[PID Driver] Completed autonomous drive run of 1000 steps!
```

## See also

- [TCP Protocol](tcp-protocol.md) — wire reference
- [RL Overview](../reinforcement-learning/rl-overview.md) — obs/action/reward contract
- [Training](../reinforcement-learning/training.md) — real trainers
- [Experiments](../experiments/experiments.md) · [CLI Reference](../experiments/cli-reference.md)
