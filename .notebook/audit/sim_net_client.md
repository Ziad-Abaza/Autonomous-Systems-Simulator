---
noteId: "8badb5b0bf6711f1a29f1fbaabbd87c8"
tags: []

---

# sim_net + sim_client Fact Sheet (audited 2026-10-03)

## Wire protocol (`sim_net/protocol.py`)
- NDJSON: UTF-8 JSON per line, `\n`-terminated. Envelope: `{"type": <msg type>, "payload": {...}}`. `json.dumps(default=convert_numpy)` (ndarray→list, np scalars→py). decode → (type, payload); empty line → ("",{}).
- `PROTOCOL_VERSION = "2.1"`; `SUPPORTED_PROTOCOL_VERSIONS = ("2.0","2.1")`; `SET_SCENARIO_MIN_VERSION = "2.1"`.
- Message types:
  - Client→Server: HANDSHAKE, DISCOVER_CONTRACT, RESET, STEP, GET_STATE, SET_SCENARIO
  - Server→Client: HANDSHAKE_ACK|ERROR, CONTRACT_ACK, RESET_ACK, STEP_ACK, STATE_ACK, SET_SCENARIO_ACK|ERROR, ERROR
- No disconnect message — client closes socket. No recording-control messages.
- Flow (lockstep request/response):
  - HANDSHAKE payload `{"protocol_versions":[...]}` (legacy key `client_protocol_versions` accepted)
  - RESET `{"seed":int|null, "options":dict|null}`
  - STEP `{"action": list|int}`; server default action=[0,0,0]
  - SET_SCENARIO `{"scenario":dict} XOR {"scenario_id":str}` + `seed`, `reset`
  - GET_STATE/DISCOVER_CONTRACT: `{}`
- HANDSHAKE_ACK payload: protocol_version, supported_versions, action_space, observation_schema, vector_dim, track_name, track_length, physics_hz, dt.
- CONTRACT_ACK: protocol_version, supported_versions, environment_id, environment_version, physics_hz, dt, action_schema, observation_schema, reward_schema, termination_schema, sensors, observation_field_classes, diagnostic_fields, scenario{name,weather,friction_mult}, capabilities{scenario_update:true, episode_state}.
- RESET_ACK: `{obs, info}`. STEP_ACK: `{obs, reward, terminated, truncated, info}`. STATE_ACK = diagnostic state: {speed, pos[x,y,z], yaw, sim_time, total_reward, reward_breakdown} — all classified DIAGNOSTIC.
- SET_SCENARIO_ACK: `{scenario:{scenario_id,name,weather,friction_mult}, episode_state:"applied"|"reset"}` + obs/info when reset=true.
- Exact error strings: "protocol_version_mismatch: no mutually supported protocol version"; "Unknown message type: X"; "unsupported_in_protocol_version: SET_SCENARIO requires protocol >= 2.1"; "scenario_update_rejected: episode is active; ..."; "unknown_scenario: 'X' is not in the standard scenario library"; "invalid_scenario: ..."; "scenario_apply_failed: ..."; "server_full: all N env slots busy"; generic `{error: str(e)}`.

## Single-env server (`sim_net/server.py`)
SimulationServer(env, host="127.0.0.1", port=8765). listen(1) — max 1 client (extras wait in backlog). Non-blocking sockets + select(timeout=0); host calls poll_and_process() (True iff a STEP processed). SO_REUSEADDR + TCP_NODELAY. recv(16384), per-conn rx buffer. Metrics: total_steps_served, last_latency_ms. Lockstep only — env steps only on STEP.

## Multi server (`sim_net/multi_server.py`)
SimServerMulti(env_factories[], host="127.0.0.1", port=8765). One process, one port, N envs. Each client pops env from free pool → _ClientSlot; empty pool → "server_full" ERROR + close. Disconnect returns env to pool. No env_id on wire — FIFO assignment. Single-threaded select loop.
Launch: `main.py --headless --num-envs N --port P --track T` → envs built from track project, seed=42+i, host hardcoded 127.0.0.1; loop `srv.poll_and_process(); time.sleep(0.0005)`.
HeadlessSimProcessPool(shared_process=True) spawns this; shared_process=False spawns N separate `main.py --headless` on consecutive ports (timeout_s=30/port).

## Session handler (`sim_net/env_handler.py`)
SimulationCommandHandler per connection; client_version tracked. Version negotiation: empty/missing → current PROTOCOL_VERSION; else highest mutual; none → ERROR. _episode_state: terminated|mid_episode|idle. SET_SCENARIO state machine: <2.1 → ERROR; mid_episode without reset → scenario_update_rejected; accepts scenario dict or scenario_id from standard library; set_scenario(); reset=true additionally resets.

## Client SDK (`sim_client/client.py`)
SimulationClient(host="127.0.0.1", port=8765, timeout=10.0). TCP_NODELAY, settimeout, recv(16384).
- connect() → env_spec dict (HANDSHAKE_ACK); ConnectionError on non-ACK
- discover_contract() → CONTRACT_ACK dict; RuntimeError on fail
- reset(seed=None, options=None) → (ndarray float32, info)
- step(action ndarray|list|int) → (obs, reward, terminated, truncated, info)
- set_scenario(scenario, seed=None, reset=False) → (obs_or_None, resp); scenario=dict|scenario_id str|object.to_dict()
- get_state() → STATE_ACK dict (ignores msg type — returns payload even on ERROR)
- close() — socket close
External agent CAN access: obs, reward, terminated/truncated, info keys (lateral_offset, speed, termination_reason, checkpoints_passed, is_colliding, is_on_road, step, road_width, laps_completed, heading_error), GET_STATE diagnostics, contract/spec. CANNOT access env internals beyond info/GET_STATE; no render stream.

## Gym adapter (`sim_client/gym_env.py`)
SimGymEnv(gym.Env); `import gymnasium` w/ `import gym` fallback. metadata={'render_modes':['human']} — NO render() implemented.
__init__(host="127.0.0.1", port=8765) — connects immediately (fails if no server); NO track param.
action_space: Discrete(len(discrete_actions)) if type==discrete else Box(continuous_low, high, float32).
observation_space: flatten_vector → Box(-inf,inf,(vector_dim,),f32); include_camera_rgb → Dict{vector, image:Box(0,255,(84,84,3),u8)}; non-flatten → Dict{speed, vel_body(2), yaw_rate, steering_angle, distance_from_center, heading_error, distance_to_checkpoint, lidar_ranges(15), camera_rgb} gated by include_* flags.
reset(seed,options)→(obs,info) 5-tuple compliant; step→5-tuple; set_scenario(scenario,seed=None,reset=True) [NOTE default reset=True vs client default False]; close().

## Agents (`sim_client/agents/`)
| module | description | run command | deps |
|---|---|---|---|
| random_agent.py | Random baseline; 300 steps of random [steer±0.5, throttle 0.2-0.8, brake 0]; auto-reset | `python -m sim_client.agents.random_agent` (no args; 127.0.0.1:8765) | numpy |
| pid_driver.py | Closed-loop PID lane tracking. Steering PID(0.85,0.005,0.05) on lat_offset*0.35+heading_err*0.75+lidar_bias*0.2; speed PID(0.25,0.01,0.05)→target 10.5+8.5*curvature_factor; brake min(0.7,−err*0.15). Decodes obs[0]*45=speed, obs[5]*road_w/2=lat, obs[6]*π=heading, obs[8:23]*40=lidar | `python -m sim_client.agents.pid_driver [host] [port]` (positional; 1000 steps) | numpy |
| ppo_train.py | Minimal demo PPO rollout over SimGymEnv — samples actions only, NO update code | `python -m sim_client.agents.ppo_train` | torch |
| ppo_baseline.py | Full CleanRL-style PPO: ActorCritic 128×128 Tanh, PPORunner(env(s), lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent=0.01, vf=0.5, max_grad_norm=0.5, num_steps=1024, epochs=4, batch=256, seed=42, device='cpu', hooks, resume_checkpoint); per-env rollout segments + GAE; truncation bootstraps V; checkpoints via torch.save | library only (no __main__); used by sim_experiment trainers | torch |
| dqn_baseline.py | DQN discrete only. QNetwork 128×128 ReLU; DQNRunner(lr=1e-3, gamma=0.99, buffer 50k, warmup 1k, batch 128, train_freq 4, target_update 500, eps 1.0→0.05 over 10k, update_interval 1000, seed 42) | library only | torch |
| sac_baseline.py | SAC continuous. SACActor 256×256 (LOG_STD −5..2, tanh+log-det), twin Q 256×256; SACRunner(lr=3e-4, gamma=0.99, tau=0.005, buffer 100k, warmup 1k, batch 256, alpha 0.2, auto_entropy, target_entropy=−act_dim, updates_per_step 1, target_update 1, update_interval 1000); maps tanh [−1,1] → env bounds | library only | torch |
| replay_buffer.py | ReplayBuffer(capacity, obs_dim, act_dim, discrete_actions=False, seed) circular np buffers; done=termination only | — | numpy |

All runners accept single env, env list, or VectorEnv; set_envs() for curriculum (constant count); on_step/on_update hooks.

## Throughput notes
recv(16384) both sides; no message cap or rate limiting beyond socket buffers. Lockstep: env advances once per STEP. Headless multi loop sleeps 0.0005 s/poll → ~2000 polls/s shared. Single-env headless runs inside studio app loop.

## Caveats
- SimGymEnv.set_scenario default reset=True vs client default False.
- get_state() returns payload even on ERROR.
- SimGymEnv connects in __init__ — construction fails without server.
- Baseline runners are libraries; runnable entry points are sim_experiment trainers/CLI.
- `main.py --headless` (num-envs 1) goes through SimulationStudioApp(headless=True)+SimulationServer, not SimServerMulti.
- ppo_train.py is a rollout demo, not a trainer.
