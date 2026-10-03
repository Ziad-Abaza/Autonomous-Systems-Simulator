---
noteId: "20772820bf6811f1a29f1fbaabbd87c8"
tags: []

---

# sim_experiment + sim_recorder + agentRL + build/tools/tests Fact Sheet (audited 2026-10-03)

## sim_experiment — training & experiment platform
Domain layer owning manifests, run lifecycle, metrics, artifacts, orchestration, trainers, evaluation, datasets, batches, remote workers. sim_core/sim_env remain independent.

### Manifest (manifest.py)
MANIFEST_VERSION="1.0". Fingerprints = SHA-256 json.dumps(sort_keys, default=str). experiment_id = f"exp_{fingerprint[:16]}".
TrainingConfig: algorithm="ppo" (ppo|sac|dqn|custom:<name>), total_timesteps=50000, rollout_length=1024, batch_size=256, epochs=4, learning_rate=3e-4, discount_factor=0.99, gae_lambda=0.95, eval_frequency=5000 (0=off), checkpoint_frequency=10000 (0=off), logging_frequency=1, num_envs=1, max_wall_seconds=0 (unlimited), algorithm_config={} (algo-specific hparams).
EvaluationConfig: eval_seeds=[0], num_episodes=5, deterministic_policy=True, scenario_id=None.
ExperimentManifest.to_dict keys: manifest_version, experiment_id, experiment_fingerprint, name, created_at, environment_name, environment_version, environment_fingerprint, environment (full project snapshot), scenario_id, scenario_configuration, random_seed=42, simulator_version, protocol_version, observation_schema, action_schema, reward_configuration, termination_configuration, episode_configuration, curriculum_configuration, randomization_configuration, training, evaluation, launched, archived, metadata.
Identity payload = env_fingerprint, scenario_configuration, random_seed, simulator_version, protocol_version, obs/action schemas, reward/termination/episode/curriculum/randomization configs, training, evaluation. name/created_at are metadata, NOT identity.

### Manager (manager.py)
default_experiments_root() = <repo>/experiments. Layout: experiments/<experiment_id>/{experiment.json, environment.json, scenario.json, runs/}. create() validates + rejects re-creating launched experiment (manifests immutable post-launch). export = copytree of whole dir. list_experiments skips legacy experiment.json lacking manifest_version.

### Runs (run.py)
RunStatus: CREATED, QUEUED, STARTING, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED, INTERRUPTED. TERMINAL = COMPLETED|FAILED|CANCELLED|INTERRUPTED.
Run fields: run_id, experiment_id, status, seed=42, created_at, started_at, ended_at, pid, current_timestep, episode_count, latest_metrics, checkpoints[], evaluation_results[], replays[], resume_from{parent_run_id,checkpoint}|None, error, exit_code, trainer, env_mode="inprocess", num_envs=1.
run_id = run_%Y%m%d_%H%M%S_<8hex>. create_run makes subdirs logs, checkpoints, evaluation, replays, trajectories, artifacts. run.json atomic.

### Metrics (metrics.py)
metrics.jsonl record: {scope: step|episode|evaluation|curriculum|bc|run, seq, ts, timestep, metrics:{}}. MetricsWriter buffered (64), NaN/Inf→null, seq survives restarts. MetricsReader: iter_rows/read_all/by_scope/latest/latest_metrics/tail/aggregate → <key>/mean|min|max|last.

### Artifacts (artifacts.py)
artifacts/registry.jsonl append-only: {kind: checkpoint|evaluation|replay|trajectory|other, path (run-relative fwd slashes), created_at, step, metadata}. register requires file exists, rejects escaping run dir. latest_of_kind, entries, find.

### Trainer contract (trainer_contract.py)
TRAINER_CONTRACT_VERSION="1.0". contract.json keys: contract_version, experiment_id, experiment_fingerprint, run_id, environment_name, environment_version, environment_fingerprint, scenario_id, scenario, seed, simulator_version, protocol_version, observation_schema, action_schema, curriculum, curriculum_fingerprint, training, evaluation, env_mode, tcp{host,ports}, paths{experiment_dir, run_dir, environment_json, scenario_json, metrics_file, checkpoints_dir, evaluation_dir, trajectories_dir, replays_dir, logs_dir, run_result}, resume, capabilities. env_mode ∈ {inprocess, tcp, tcp_multi, process}; tcp modes require tcp.ports. Trainers must write run_result.json {"status": "completed"|...}.

### Capabilities (capabilities.py)
TRAINER_CAPABILITIES: ppo [ppo] continuous vector multi_env torch eval✓; sac [sac] continuous vector multi_env torch eval✓; dqn [dqn] discrete vector multi_env torch eval✓; bc [bc] continuous+discrete vector multi_env torch eval✓; dummy [dummy] cont+disc vector+image multi_env raw eval✗. check_compatibility(manifest, trainer) validates algorithm membership, action type, image-obs support, multi_env.

### Orchestrator (orchestrator.py)
TRAINER_MODULES: ppo→sim_experiment.trainers.ppo_trainer, sac→sac_trainer, dqn→dqn_trainer, bc→bc_trainer, dummy→dummy_trainer.
launch(manifest, exp_dir, trainer="ppo", env_mode="inprocess", run_id, resume_from, run_overrides): compatibility+curriculum validation first; run_overrides keys seed, scenario_dict, algorithm_config, bc_dataset_dir. TCP modes spawn HeadlessSimProcessPool first (tcp=N processes/N ports; tcp_multi=1 process/1 port/N envs). Launches [python -m module --run-dir rd], logs→logs/stdout.log+stderr.log. Status CREATED→QUEUED→STARTING→RUNNING.
poll(): syncs metrics tail→progress, artifact registry→refs, max_wall_seconds→INTERRUPTED("timeout"), finalize on exit via run_result.json (nonzero exit w/o result → FAILED "trainer_crash"). wait(timeout_s=600), cancel (graceful terminate→kill, 5s grace).

### Headless helpers (headless.py)
build_env_from_dicts(env_dict, scenario_dict=None, seed=42) → SimulationEnvironment — canonical factory.
HeadlessEnvPool(env_dict, scenario_dict, num_envs, base_seed=42) — N in-process envs seed=base+i.
HeadlessSimProcessPool(num_envs, base_port=None, shared_process=False) — spawns main.py --headless --port P (shared adds --num-envs N); logs logs/sim_<port>.log; timeout_s=30.

### Vec envs (vec_env.py, process_env.py)
VectorEnv duck-type: reset_all(seeds), reset_at(index,seed), step_all(actions), set_scenario(dict), close(). SyncVectorEnv wraps env list. ProcessVectorEnv: one spawn mp.Process per env, pipe cmds (reset,step,set_scenario,ping,close), pipelined; dead worker → EnvWorkerCrash(worker_index) retryable; startup_timeout=120.

### Batch + scheduler (batch.py, scheduler.py)
expand_run_specs(manifest, seeds, scenario_ids) — cross product.
BatchScheduler(experiments_root, max_workers=2, orchestrator, workers, heartbeat_interval_s=10, lease_ttl_s=30). Persists <exp>/batches/<batch_id>/batch.json + batch_result.json. batch_id=batch_%Y%m%d_%H%M%S_<8hex>. JobStatus QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED. WorkerState IDLE/STARTING/RUNNING/FAILED/STOPPING/OFFLINE.
RetryPolicy: RETRYABLE = {trainer_crash, trainer_exception, launch_failure, timeout, worker_crash}; NON_RETRYABLE = {invalid_contract, invalid_curriculum, invalid_experiment, unsupported_algorithm, curriculum_mismatch, curriculum_state_missing, protocol_version_mismatch, launch_rejected, incompatible_environment}; unknown → not retried. create_batch(manifest, exp_dir, specs|seeds+scenario_ids, trainer, env_mode, retry_policy); run_until_complete, status, cancel_batch.

### Remote workers (remote_worker.py, worker_registry.py)
WorkerService(host, port, experiments_root, token) — TCP NDJSON req/resp; WORKER_PROTOCOL_VERSION="1.0". Messages carry {v,type,token}; auth hmac.compare_digest. HELLO→HELLO_ACK, REGISTER→REGISTER_ACK{worker_id}, HEARTBEAT→HEARTBEAT_ACK, STATUS→STATUS_ACK, LAUNCH→LAUNCH_ACK{run_id}, POLL→POLL_ACK{summary}, CANCEL→CANCEL_ACK, else ERROR. LAUNCH/POLL/CANCEL enforce experiment_dir containment. worker_id=worker_<12hex>.
RemoteWorkerAdapter(host,port,token,timeout=30): handshake/register/heartbeat/capabilities/launch/poll/cancel.
WorkerRegistry → <experiments_root>/workers/registry.json {workers:{id:{worker_id,capabilities,meta,status ONLINE|OFFLINE,...}}}.

### Evaluation (evaluation.py)
EvaluationResult: eval_id, checkpoint_path, algorithm, env_fingerprint, scenario_id, deterministic_policy, seeds[], episodes[], aggregate{}, created_at. eval_id=eval_%Y%m%d_%H%M%S_<8hex>.
evaluate_policy(env, act_fn, seeds, num_episodes, deterministic=True, ...): cycles seeds[i%n]. Episode keys: seed, reward, length, termination_reason, completed, collided, off_road, timed_out, mean_lateral_error, mean_heading_error, mean_speed. Aggregate: episode_count, mean/std/min/max_reward, completion_rate, collision_rate, off_road_rate, timeout_rate, mean_episode_length, mean_lateral_error, mean_heading_error, mean_speed.
export_transitions → transitions_v1 dataset (episodes.jsonl+manifest.json), episode ids eval_ep{i}, source "evaluation_export".
make_policy_from_checkpoint(ckpt, algorithm, deterministic): ppo ActorCritic, sac SACActor (tanh→bounds), dqn QNetwork argmax, bc BCPolicy.

### Trajectories (trajectory.py, trajectory_sampler.py)
runs/<run_id>/trajectories/<episode_id>.jsonl. Header: {type:"header", episode_id, env_fingerprint, scenario_id, seed, episode_seed, curriculum_stage_index, env_index, observation_schema, action_schema}. Step: {type:"step", episode_id, step, agent_data:{obs,action,reward,terminated,truncated,termination_reason}, diagnostic_data:{speed,lateral_offset,heading_error,is_colliding,is_on_road,checkpoints_passed}}. TrajectoryWriter(buffer_size=128); TrajectoryReader.read()→{header,steps}.
EpisodeTrajectoryRecorder (trainers/_harness.py): per-env episodes → train_env{i}_ep{j}; selection via algorithm_config.trajectory_sampling modes: first_n{n} default, every_n{n}, probability{p}, episodes{ids}, min_return{v}, termination_reasons{...}, all.
sample_trajectories(run_dir, count, seed, strategy): uniform|best|mixed (deterministic).

### Dataset export — transitions_v1 (dataset.py, dataset_inspect.py)
DATASET_FORMAT="transitions_v1". export_dataset(run_dir, output_dir, env_fingerprint=None, min_return=None, termination_reasons=None):
- episodes.jsonl: {episode_id, seed, env_fingerprint, scenario_id, total_return, length, termination_reason, steps:[{obs,action,reward,terminated,truncated,termination_reason}]} — whitelisted agent fields only; diagnostics excluded.
- manifest.json: {dataset_format, source_run_dir, schema_hash, env_fingerprint, episodes, steps, filters, skipped{...counts}, isolation}. schema_hash=sha256({obs_schema,action_schema})[:16].
validate_dataset(dir): manifest+episodes exist/parse, format==transitions_v1, unique episode_ids, non-empty steps, required {obs,action,reward}, allowed step fields only (extras=error), length consistency (warn), last step terminated||truncated (warn), counts match → {valid,errors,warnings,episodes,steps,fingerprints,schema_hash,step_field_violations}.
split_dataset(dir, seed, ratios train .8/val .1/test .1) → splits.json {seed,ratios,episodes,splits{name:[ids]}} deterministic hash "{seed}:{episode_id}".
inspect_dataset(dir) → {dataset_dir, dataset_format, manifest, statistics{episode_count,step_count,return{mean,std,min,max},length{...},termination_reasons{},obs_dims,action_dims,fingerprints,scenario_ids,splits}, valid, errors, warnings}.

### Convergence (convergence.py)
convergence_report(results, min_reward_delta, stability_tolerance): verdicts insufficient_data|regressed|unstable|improved|no_improvement. Keys: verdict, stable, regressions[], improvement{delta_mean_reward,delta_completion_rate}, baseline/final_mean_reward, cross_seed{}, phases_analyzed, thresholds.

### Curriculum runtime (curriculum_runtime.py)
CURRICULUM_STATE_VERSION="1.0". _METRIC_ALIASES: mean_return→mean_reward, lap_completion_rate→completion_rate + others. _LOWER_IS_BETTER={collision_rate,off_road_rate,timeout_rate,mean_lateral_error,mean_heading_error}. environment_overrides map: target_speed→target_speed_override, time_limit→time_limit_override, surface_friction_mult, sensor_noise_mult, ambient_light, weather, time_of_day.
CurriculumController: stage_seed=base+env_index+stage*10000; stage_scenario_dict applies overrides; evaluate_advancement requires metric+episodes_in_stage≥min_episodes+threshold pass+next stage; history records; to_state→runs/<run>/curriculum_state.json.

### Analytics (analytics.py)
compare_runs(run_dirs, metric, scope="episode", smooth_window=10) → {metric,scope,series:[{label,run_id,raw,smoothed,max,min,final,mean,count}]}. metric_summary(run_dir) → {run_id,episodes,mean_return,best_return,final_return,mean_length,best/final_eval_mean_reward,final_eval_completion,total_timesteps,status,wall_clock_time,sps,termination_reasons}.

### Reproduce (reproduce.py)
check_reproducibility(experiment_dir): experiment.json exists/parses (fatal), manifest_version=="1.0", env fingerprint match (fatal), scenario match, obs/action schemas, termination rules, episode config, simulator_version ∈ KNOWN, protocol_version ∈ SUPPORTED, training config → {reproducible, exact_reproduction_guaranteed:False, checks[], fatal_failures[], warnings[]}.

### Trainers (trainers/)
All run as `python -m sim_experiment.trainers.<name> --run-dir <dir>`; shared trainers/_harness.py (load_contract, build_envs_from_contract honoring env_mode, resolve_resume_checkpoint, action_bounds [defaults [-1,0,0]/[1,1,1]], obs_to_vec, EpisodeTrajectoryRecorder, run_periodic_eval [eval_<step>.json + replays/eval_<step>_ep0.json with frame keys {step,t,pos,yaw,speed,action,reward,breakdown,lat_offset,heading_err,collision}], write_result).
- ppo_trainer → PPORunner; alg keys clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, trajectory_episodes=0. Episode metrics {reward,length,termination_reason,mean_lateral_error,mean_speed,checkpoints_passed}; run metrics {sps,policy_loss,value_loss,entropy,approx_kl,explained_variance,episodes_completed}; final policy_final.pt. Exit 0/2/1.
- sac_trainer → SACRunner; continuous only; alg keys buffer_size=100000, warmup_steps=1000, tau=0.005, alpha=0.2, auto_entropy_tuning, target_entropy, updates_per_step, target_update_interval, update_interval=500; ckpt_freq 2048, eval_freq 8192.
- dqn_trainer → DQNRunner; discrete only; alg keys buffer_size=50000, warmup_steps=1000, train_freq=4, target_update_interval=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=10000, update_interval=500.
- bc_trainer → sim_experiment.bc.BCRunner over contract.bc.dataset_dir (transitions_v1, fingerprint must match); alg keys bc_epochs=20, batch_size, bc_lr, hidden_sizes=(64,64), val_ratio=0.2; BCPolicy=Tanh MLP→action head; MSE cont / CE disc, Adam CPU.
- dummy_trainer → test trainer; alg keys tick_seconds, fail_at_step, emit_checkpoint_at.

### CLI (cli.py) — `python -m sim_experiment.cli`, global --root (default <repo>/experiments)
| Subcommand | Key args |
|---|---|
| validate-env | --project (req) |
| create | --project req, --scenario basic_lane_following, --name experiment, --algorithm ppo, --timesteps 50000, --rollout 1024, --batch-size 256, --epochs 4, --lr 3e-4, --gamma 0.99, --gae-lambda 0.95, --eval-freq 0, --ckpt-freq 10000, --num-envs 1, --max-wall-seconds 0, --seed 42, --eval-seeds "", --eval-episodes 5, --alg-config ""(JSON), --force |
| list / show <id> | — |
| launch | experiment_id, --trainer ppo, --env-mode inprocess|process|tcp|tcp_multi, --wait 0 |
| batch | experiment_id, --seeds int*, --scenarios str* |
| batch-run | experiment_id, --seeds, --scenarios, --trainer ppo, --env-mode, --workers 2, --worker host:port, --token, --max-retries 0, --timeout 600 |
| trainers | dump TRAINER_CAPABILITIES |
| curriculum | experiment_id run_id |
| compare | experiment_id, --metric reward, --scope episode, --smooth 10, --runs "" |
| dataset-export | experiment_id run_id, --dest req, --min-return, --env-fingerprint, --termination-reasons csv |
| dataset-validate | dataset_dir |
| dataset-split | dataset_dir, --val-frac 0.1, --test-frac 0.1, --seed 42 |
| dataset-stats | dataset_dir |
| train-bc | experiment_id, --dataset req, --epochs 20, --val-frac 0.2, --seed 42, --wait 0 |
| worker-serve | --host 127.0.0.1, --port 9100, --token req, --worker-root |
| worker-status/-register/-list | --host, --port req, --token req |
| runs | experiment_id |
| status | experiment_id run_id, --watch |
| cancel | experiment_id run_id |
| resume | experiment_id run_id, --trainer ppo (NEW run, resume_from={parent_run_id,checkpoint}) |
| evaluate | experiment_id run_id, --checkpoint (latest registered default) |
| trajectories | experiment_id run_id, --json |
| reproduce | experiment_id |
| export | experiment_id, --dest req |
| archive | experiment_id |
| benchmark | --envs "1,2,4", --steps 2000, --template lane_following |
| learn-bench | --config req, --work-dir; runs benchmarks.phase6.benchmark_runner + convergence_report; exit 1 if regressed |

## sim_recorder
EpisodeRecorder(max_steps=5000) ring-drops oldest when full. start_recording(track_name, seed, env_config, env_version, scenario_name, observation_schema, action_schema, reward_config, fingerprint, scenario_config, simulator_version, protocol_version). record_step → frame {step,t,pos[3],yaw,speed,action[],reward,breakdown{},lat_offset,heading_err,collision} — NO observations recorded (telemetry only). stop_recording(termination_reason,episode_result); save_to_file JSON (gzip if .gz); load_from_file.
EpisodeReplayPlayer: load_recording, play/pause/toggle_play, seek (clamped), step_forward/step_backward, playback_speed stored field (UI drives it). Studio writes <data_root>/recordings/<name>/{episode.json,manifest.json{kind:"episode_recording",name,steps,track_name,simulator_version,created}}. Datasets at <data_root>/datasets/.

## agentRL — continual multi-track RL platform (PARTIALLY IMPLEMENTED)
Implemented: core/, obs/, act/, rewards/, envs/, memory/. MISSING: algos/, train/, eval/, checkpoints/, experiments/, NO training CLI. Library-style usage.
- core: AGENT_VERSION="0.1.0", CKPT_SCHEMA_V=1, OBS_SPEC_V=1, ACT_SPEC_V=1; seed_tree(base_seed)→{env,track,torch,mutator,eval,worker}; AgentConfig{algo_id,hidden_sizes(256,256),lr 3e-4,gamma 0.99,device cpu,extra}; TrainConfig{total_steps,eval_interval 10000,ckpt_interval 10000,num_envs 1,seed 42,run_dir agentRL/runs/default,eval_episodes 3,resume_from,extra}.
- obs: DEFAULT_CHANNELS widths speed1,velocity_body2,yaw_rate1,steering_angle1,distance_from_center1,heading_error1,distance_to_checkpoint1,lidar_ranges15. PRESETS: state8 (7 state ch → 8 dims), full23 (+lidar → 23), lidar15. ACTION_DIM=3. ObservationSpec{channel_names,frame_stack=1,prev_action=False,image=False}; input_dim=vector_dim*frame_stack+(3 if prev_action). ObsEncoder: frame deque, prev_action append, NotImplementedError on image.
- act: ActionAdapter(low,high): to_env clip[−1,1]→affine to bounds; from_env inverse; tracks prev env action; reset().
- rewards: reward_preset("drive_v1"): progress 2.0{max_step_delta_m 5}, speed 0.5{target 15}, centering 0.05{max_dist 6,linear}, heading 0.05, smooth_steer −0.02, checkpoint 5, completion 100, collision −50, off_road −25, reverse −1{thresh 100°}, time_penalty −0.12 (idle ≈ −0.02/step). termination_preset("term_v1"): term collision/off_road/wrong_direction(120°)/course_completion(1 lap); trunc max_steps 1500, checkpoint_timeout 20s. Registries: list_reward_presets→["drive_v1"], list_termination_presets→["term_v1"].
- envs: TrackRegistry.default() — FILE_TRACKS{oval,serpentine,smoke→tracks/*.sim.json} + gen_loop_0..5 (seeds 10000+i; tags train i<4 else holdout). track_gen: gen_oval(rx,ry,width,n_cps), gen_loop(seed,n_cps=10,base_r=55,radius_jitter=18,width=12). EnvFactory(reward="drive_v1",termination="term_v1",obs_spec=full23,sensor_names=None,max_duration_s=120) — clones project, injects obs channel subset, presets, minimal sensors, episode max_duration; build(track,seed)→build_env_from_dicts. ScenarioMutator(seed, n_obstacles(0,3), types cone/barrier/obstacle, lateral_frac(−0.4,0.4), s_range(0.15,0.95), min_gap 15m, spawn jitter ±1m/±15°, friction(0.85,1.1), noise(0.8,1.5)) — draw()→ScenarioDefinition on spline; deterministic per seed.
- memory/replay.py: ReplayBuffer(capacity,obs_dim,act_dim,seed) — done iff terminated and not truncated (truncation bootstrap). TrackRehearsalBuffer: per-track buffers + rehearsal_fraction mixing.
- Usage: TrackRegistry.default().load("oval") → EnvFactory(...).build(track, seed) → SimulationEnvironment.
- Deps: numpy only (no torch). tracks/ dir gitignored — file tracks need Studio to generate them.

## Build / tools / tests
- build/package_windows.py: `python build/package_windows.py` → PyInstaller --onedir --windowed --distpath dist --workpath build_temp, entry main.py → dist/AI_Environment_Simulator/AI_Environment_Simulator.exe; copies presets + README into bundle. hiddenimports: moderngl,glcontext,OpenGL,pygame,numpy,cv2,PIL,shapely,scipy,gymnasium + sim_* (NOTE: sim_experiment NOT included; glcontext/PIL/shapely/scipy/OpenGL unused in code). Excludes: torch,tensorflow,pandas,matplotlib,pytest,etc. → bundled app CANNOT run torch trainers.
- AI_Environment_Simulator.spec mirrors the script (console=False, upx=True).
- tools/ui_shots.py [width height] → assets/screenshots/qa/<W>x<H>_<name>.png (docstring stale saying docs/phase7_shots). ~35 shots.
- tools/smoke_interactions.py → ~37 interaction checks through real app event handling.
- tools/tcp_recording_e2e.py → 15 checks, TCP-driven recording E2E.
- tools/vehicle_dynamics/: calibrate.py, harness.py, maneuvers.py, run_validation.py (used by DYNAMICS tab + validation docs).
- tests/: 64 files, 481 test functions (7 files + conftest under tests/agent/). Run: `python -m pytest tests/ -q`. No markers. tests/agent/conftest needs tracks/*.sim.json (gitignored; generated by studio).
- Existing docs/: ~70 flat files; ~40 technical reference (ACTION_SCHEMA, OBSERVATION_SCHEMA, SENSOR_CONFIGURATION, REWARD_*, TERMINATION_SYSTEM, SCENARIO_SYSTEM, CLI_REFERENCE, EXPERIMENT_*, TRAINERS, etc.), ~22 phase reports/audits, UX refs. Untracked historically.
- No requirements.txt/pyproject anywhere. Real third-party imports: numpy, pygame, moderngl, cv2 (camera_sensor only), torch (trainers/agents/experiment scripts), gymnasium (gym_env w/ gym fallback). Installed env: Python 3.13.7, torch 2.10 CPU, gymnasium 1.2.1, moderngl 5.12, pygame 2.6.1.
- SIMULATOR_VERSION="4.0.0" (sim_version.py). TCP PROTOCOL_VERSION="2.1". WORKER_PROTOCOL_VERSION="1.0". MANIFEST_VERSION="1.0". TRAINER_CONTRACT_VERSION="1.0". DATASET_FORMAT="transitions_v1". SCHEMA_VERSION="2.0.0"/"3.0.0".
- Benchmarks committed: phase4_results.json (env throughput: 352 sps 1 env, 340 sps 2 envs aggregate, 332 sps 4 envs; sensor cost full_suite 342 sps vs state_only 1394 sps; metrics overhead ~0%); vision_results.json (84×84×3 frame 0.56 ms total, 1786 fps; env step vision 1736 sps vs vector-only 1783 sps); broadphase_results.json (spatial hash: 2.35× small track, 5.75× medium, 96-99% candidate reduction, 100% correctness); benchmarks/phase5_benchmarks.py, phase6/benchmark_runner.py (learning benchmark config eval_seeds [101,202,303], thresholds delta≥5/tol 10, timeout 1200 s), phase6/perf_runner.py (env_mode scaling inprocess|process|tcp|tcp_multi).
- experiments/ dir committed: run_ppo_experiment.py (standalone baseline PPO, out baseline_ppo/), run_ui_env_ppo.py (template-based, out baseline_ui_env/), exp_556fe0488d4db907/ real experiment dir w/ runs, exported/ copy.
- last_episode.json = legacy recording, replay tab fallback. data/, tracks/, logs/ gitignored runtime dirs.
