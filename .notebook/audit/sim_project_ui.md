---
noteId: "bf8b2520bf6711f1a29f1fbaabbd87c8"
tags: []

---

# sim_project + sim_ui Fact Sheet (audited 2026-10-03)

## Settings (`sim_project/settings.py`)
Storage: `<repo>/data/studio_settings.json` (app.py:68-78; legacy `<repo>/studio_settings.json` migrated via os.replace). Atomic save (tmp+os.replace). load() only keeps keys in DEFAULT_SETTINGS.

| Key | Type | Default | Controls |
|---|---|---|---|
| data_root | str | "" → <repo>/data | Root for recordings/ and datasets/ (settings.py:97-105); validate_data_root() write-probes |
| theme | str | "dark" | Palette id (dark, dark_ocean, dark_ember, light, light_solar); unknown → dark |
| ui_scale | float | 1.0 clamp [0.75,2.0] | Font/UI scale via Fonts(scale) |
| recent_files | list[str] | [], max 10 MRU | push_recent() filters missing paths |

Settings screen (HOME→SETTINGS): Theme palette cards grouped DARK/LIGHT, Data root Choose… (tkinter askdirectory → app.choose_data_root), UI scale buttons 0.9/1.0/1.15/1.3. recent_files not UI-editable.

## Project format (`sim_project/serializer.py`)
`*.sim.json` JSON indent=2. SCHEMA_VERSION="2.0.0" (+V3="3.0.0"). environment_version default "1.0.0" (semantic, auto-increment patch on fingerprint change at save).
Top keys: schema_version, environment_version, name, road_definition, vehicle_config, action_config, observation_schema, reward_config, termination_config, randomization_config, scenario_config, entities[], agent, episode_config, scenario_def, [curriculum], [experiment_config], fingerprint (SHA-256).

road_definition: {name, is_closed, control_points[{x,y,z,width,banking,friction}], boundary_config{left_type,right_type,wall_height,curb_width,curb_height,has_curbs,has_lane_markings}, spawn_point{x,y,z,yaw,initial_speed}, num_checkpoints, default_friction}.
agent: {agent_id, name, entity_type, entity_id, sensor_configs[], sensor_names[], observation_space, action_space, reward_function, termination_rules, spawn_config{spawn_mode,pos,yaw_deg,initial_speed,lateral_jitter_m,heading_jitter_deg}}.
scenario_def: {scenario_id, name, description, weather, time_of_day, ambient_light, surface_friction_mult, target_speed_override, time_limit_override, sensor_noise_mult, spawn_override, obstacle_overrides, randomization}.

Migration on load: schema_version default "1.0.0"; entities ← entities|obstacles|scenario_config.obstacles; missing agent→default vehicle agent; missing episode_config synthesized from termination/road; missing scenario_def→basic_lane_following; schema coerced to "2.0.0" unless already 2.0.0/3.0.0. Tolerant .get loading — no strict gate (validation is validator report, not load gate).

## Track library (`sim_project/library.py`)
Roots: `<repo>/tracks` writable + `<repo>/presets` readonly. .sim.json is source of truth; sidecar `.library.json` holds description/favorite/last_opened per filename.
TrackAsset: path, file_name, name, description, env_version, schema_version, point_count, entity_count, is_closed, length_m (polyline approx), modified, file_size, favorite, last_opened, broken+error, readonly.
Ops: scan(), save_project (unique slug → `<slug>.sim.json`, `_2`,`_3`), create, duplicate (name+" (copy)"), rename (project.name only — filename stays stable), delete (file+sidecar+.thumbs png), set_favorite, set_description, mark_opened, thumbs_dir()=`<root>/.thumbs`, recents(n=6).
Card context menu: Open, Favorite/Unfavorite, Rename…, Duplicate, Reveal in Folder, Delete…
Thumbnails: render_track_thumbnail 192×112 road polygon + centerline + direction chevron + spawn marker; cached keyed by mtime.

## Presets (`sim_project/presets/` + `<repo>/presets/`)
create_oval_circuit() → "Proving Ground Oval" oval 65×40 w12; create_serpentine_track() → "Alpine Serpentine Circuit" closed 10 CPs elevation ±4 m widths 10-14, 20 checkpoints; create_obstacle_challenge() → "Obstacle Evasion Proving Ground" oval + 3 obstacles. save_default_presets() writes oval_circuit/serpentine_track/obstacle_challenge .sim.json. main.py regenerates only if presets/ dir missing entirely. Disk also has custom_environment.sim.json (schema 2.0.0; others 1.0.0).

## Studio app (`sim_ui/app.py`)
SimulationStudioApp(width=1280, height=720, headless=False, port=8765).
Screens: studio_screen ∈ home|workspace; ws_tab ∈ EDIT|SIMULATE|REPLAY|DATA|DYNAMICS (init SIMULATE). open_project() → workspace+EDIT; ESC hierarchy dialog→tool→selection→home→quit.
TCP: studio ALWAYS hosts SimulationServer on 127.0.0.1:port at startup (no toggle); poll_and_process each frame; SIMULATE status bar shows "AI CONNECTED"/"AI listening".
Headless: skips graphics init; loop polls server + sleep(0.005). Interactive mode: max_episode_steps=0, max_seconds_without_checkpoint=0 (timeouts disabled while driving manually).
Undo: _project_snapshot() = project.to_dict() covers editor AND inspector edits; mark_dirty pushes snapshots; undo/redo.
Recording: start_recording(name,dest) captures schemas/fingerprint; stop → `<dest>/<name>/episode.json`+manifest.json{kind:"episode_recording",name,steps,track_name,simulator_version,created} → rec_summary dialog.
Training services: ExperimentManager/LocalTrainingOrchestrator/RunManager at `<repo>/experiments`; trn_* actions (create→ppo 20000 steps, ckpt 5000, eval seeds [0,1]×3 eps, seed 42; launch, batch(2 seeds), cancel, resume, eval, repro, export, dataset, compare); _poll_training_runs ~1 Hz.
CLI (main.py): --headless, --port 8765, --num-envs 1, --track oval|serpentine|obstacle|path, --width 1280, --height 720.

## Track editor (editor.py, editor_ui.py, edit_history.py)
Tools: tool ∈ select|draw; active_tool ∈ obstacle|barrier|cone|traffic_sign|traffic_light|spawn.
- LMB drag point; draw tool: LMB empty → append CP; click within 10px of segment → insert interpolated CP; click first CP of open track (≥3pts) → close loop; RMB on point → delete (if >3); Del key too.
- Entities: LMB drag move, RMB delete; placement via toolbar then auto-select.
- Spawn: click within 12px selects; spawn tool sets x/y.
- Pan MMB or RMB-drag; zoom wheel ×1.15 cursor-anchored clamp 0.05-60 (default 4.0 px/m).
- Snapping snap_enabled + snap_step=1.0 m (unsnapped rounds 0.1).
- frame_all/frame_selected/frame_bounds/frame_point (zoom clamp 0.05-35); _needs_frame on open.
- Toggles show_curvature/show_tangents/show_width_handles/show_grid (all default True).
- Rendering: 10m grid, boundary polylines, curvature-colored centerline, direction chevrons, checkpoint gates (start thicker, last open-route gate = FINISH badge), entity gizmos per type, spawn marker + vehicle footprint 4.5×1.8m + yaw arrow + label, selected-CP width preview + badge `P<i> (<w>m) Z:+ <Bank:°>`, Close Track hover affordance, placement banner.
- Toolbar 36px: Select, Draw Pt, Place ▾, Snap, Grid, Fit All, Fit Sel, Open/Closed toggle, View ▾, zoom −/+. OUTLINE rail 230px: CONTROL POINTS/ENTITIES/SPAWN/GATES. Inspector docked right 350px. Status bar: tool, coords, px/m, SNAP, dirty.
- EditHistory(limit=64): deep-copied full-document snapshots, no-op dedup, redo truncation.

Key bindings EDIT: Ctrl+S save, Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, A fit-all, F fit-selection, Del/Backspace delete, G grid, S snap, V select tool, P draw tool, Esc cancel→deselect.
App-level: Ctrl+S/Ctrl+O save/open; SIMULATE: R reset, C cycle camera (CHASE→HOOD→TOP_DOWN→ORBIT), TAB obs inspector; REPLAY: Space play/pause, ←/→ step; ESC close; Enter confirms new-track. Driving: steer A/D or ←/→ (target ±0.8, smoothed ×0.25); throttle W/↑ (+0.1, decay −0.15); brake S/↓/Space (+0.2, decay −0.2).

## Inspector (`inspector.py`)
Two-tier tabs: GEO [OVERVIEW, SCENE, TRACK, POINT, ENTITY]; RL [AGENT, SENSORS, OBS, ACTION, REWARD, TERM, SCENARIO, VALIDATE, TRAIN] (labels abbreviated). Bottom: Rebuild 3D, New/Save/Open. Row types: label, float/int ±, bool, enum, action, nav, chart, multichart.
- OVERVIEW: env summary + Training Readiness VALID/INVALID; Export Training; 3 template loaders.
- AGENT: id/name/entity labels, sensors bound, spawn initial_speed (0-60 step2), lateral_jitter_m (0-5 step0.2).
- SENSORS: enable toggle + EDIT per config; add RGB Camera/LiDAR/IMU/Vehicle State; camera params: rate, FOV 30-120°, w/h 32-512, mount xyz/yaw/pitch, noise, latency, "In Observations" (syncs image_channels); lidar: beams 3-64, FOV ≤360, range 5-200; imu: accel/gyro noise, bias drift; duplicate/remove.
- OBS: vector dim, flatten_vector, Leakage Guard, per-channel enable toggles.
- ACTION: space_type continuous/discrete; per-channel min/max, dead_zone ≤0.2 (clamp 0.3), rate_limit.
- REWARD: per-component enable + weight (−200..500).
- TERM: per-rule enable, TERM/TRUNC tags.
- SCENARIO: preset enum (6), weather, time_of_day, friction 0.2-2.0, ambient light 0.1-1.0, DR enable.
- VALIDATE: PASS/FAIL, counts, Re-Run, clickable issues → jump to owning tab; error badges on tabs.
- TRAIN: experiment list, create/launch/batch/cancel/resume/evaluate/repro/export-dataset/export/compare; run monitor (status, timesteps, episodes, curriculum, metrics, reward chart), batch progress, workers, dataset preview, comparison chart.
- SCENE: track info, entity list, +Box/+Barrier/+Cone/+Sign/+Light place actions.
- TRACK: is_closed, default_friction 0.1-2.5, num_checkpoints 4-64 step2, boundary enums, has_curbs, curb_width 0.2-2.0, wall_height 0.3-3.0.
- POINT: X/Y ±1000, width 4-40, elevation −50-100, banking ±30°, Delete Point (>3).
- ENTITY: pos x/y, yaw ±180°, collidable, delete.
- Readiness gating: validator re-runs after every property change; agent-space changes → runtime recompile.

## Workspace (`workspace_screen.py`)
Tabs EDIT|SIMULATE|REPLAY|DATA|DYNAMICS. Header: < Library, track name+dirty dot+version, tabs, Save, camera buttons (Chase/Hood/Top/Orbit) on SIMULATE/REPLAY.
- SIMULATE: 3D viewport + HUD telemetry (left), reward decomposition (right), camera PiPs, obs inspector; status bar Record/Stop + step count + AI status + hints.
- REPLAY: picker lists `<recordings>/*/episode.json` + legacy `last_episode.json`; player info chip, transport |< < > >> >|, speeds 0.5/1/2/4×, scrub bar, Close. Keys Space/←/→.
- HUD (hud.py): telemetry Speed km/h+m/s, Center Dev, Heading Err, Laps/Gates, Collision, steer/throttle/brake meters; reward inspector per-term bars; camera PiP; obs inspector two-column AI-obs vs ORACLE telemetry.
- DATA: scans <data_root>, experiments/, <repo>/data/experiments for manifest.json/episode.json(.gz); EPISODE RECORDINGS + TRAINING DATASETS sections; detail = inspect_dataset report + View Replay/Open Folder/Open Source/Delete.
- DYNAMICS: runs tools.vehicle_dynamics maneuver suite physics-only dt=1/60; speed selector 10/20/30 m/s, Run full suite, maneuver list; 14 metric rows, trajectory plot, sideslip/yaw/ay charts, Set A/B compare, Export → benchmarks/vehicle_dynamics/ui_<ts>/. NOTE: uses default VehicleConfig(), not the project's.

## Home screen (home_screen.py)
Left nav 190px: TRACKS (count badge), RECENT, TEMPLATES, DATASETS, EXPERIMENTS, SETTINGS; footer data_root.
TRACKS: + New Track, search, sort cycle modified→name→length, favorites first; cards 224×176 thumbnail/name/open-tag/pts·ent·length·m·v<ver>/description/★/age/Open/… menu.
Dialogs (dialogs.py): new_track (name + template pick, Enter, default template "empty"), confirm_delete, rename, confirm_quit, confirm_new/discard_new, record (name+dest browse), rec_summary (steps/duration/return/termination/path + View Replay/Record Again/Show in Folder), confirm_del_ds.

## Themes (theme.py)
PALETTES: dark "Midnight", dark_ocean "Ocean", dark_ember "Ember", light "Paper", light_solar "Solar". Default dark. Toggle via SETTINGS palette cards → set_theme persists settings.theme. Metrics: FONT_CAPTION 12…H1 20, PAD 12, RADIUS 6, TOOLBAR_H 36, RAIL_LEFT_W 240, RAIL_RIGHT_W 340.

## Caveats for docs
- vc_* vehicle-dynamics and sp_* spawn inspector handlers exist but no tab emits them — vehicle params are read-only in UI; edit via .sim.json or DYNAMICS tab (which uses DEFAULT config, not project's).
- hud.py legacy draw_top_bar/draw_bottom_bar/draw_editor_palette unused.
- presets regenerated only if whole presets/ dir missing.
- library rename() keeps filename stable (renames project.name only).
- Termination timeouts disabled in interactive studio (max_episode_steps=0, max_seconds_without_checkpoint=0).
