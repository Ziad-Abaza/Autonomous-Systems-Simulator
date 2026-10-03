---
noteId: "64cce690bf6811f1a29f1fbaabbd87c8"
tags: []

---

# Documentation Style Guide — READ FULLY BEFORE WRITING

You are writing documentation for the **Simulation Studio — Autonomous Systems Simulator**,
a Python 3D simulation + RL platform at `D:\coding\projects\Simulation`.
Platform version `4.0.0`, TCP protocol `2.1`, project schema `2.0.0`.

## Ground rules
1. **Facts come from the audit files** in `.notebook/audit/` (sim_core.md, sim_env.md,
   sim_net_client.md, sim_project_ui.md, sim_experiment.md) and — when you need more —
   from the actual source code. NEVER invent features, commands, flags, defaults, or
   benchmark numbers. If something is declared-but-unused in code (see audit "Caveats"
   sections), document it as such honestly.
2. **Every command must be real.** Verified commands:
   - `python main.py [--headless] [--port N] [--num-envs N] [--track oval|serpentine|obstacle|<path.sim.json>] [--width W] [--height H]`
   - `python -m sim_experiment.cli <subcommand> ...` (subcommands in audit file)
   - `python -m sim_client.agents.pid_driver [host] [port]` (positional args!)
   - `python -m sim_client.agents.random_agent` (no args)
   - `python -m sim_client.agents.ppo_train` (no args, rollout demo — NOT a real trainer)
   - `python -m pytest tests/ -q` (481 tests)
   - `python tools/ui_shots.py [W H]`, `python tools/smoke_interactions.py`, `python tools/tcp_recording_e2e.py`
   - `python build/package_windows.py`
   - `python experiments/run_ppo_experiment.py`, `python experiments/run_ui_env_ppo.py`
3. **Screenshots**: files live in `docs/screenshots/`. Reference with a RELATIVE path
   from your file's location (e.g. from `docs/user-guide/foo.md` → `../screenshots/x.png`;
   from `docs/getting-started/foo.md` → `../screenshots/x.png`). Alt text required.
   Only use filenames that exist — the inventory list is given in your task prompt.
   Do NOT invent screenshot filenames.
4. **Cross-links**: relative markdown links between docs (e.g. `../agents/tcp-protocol.md`).
   Every linked file must be in the tree map given in your prompt.
5. **Style**: GitHub-flavored Markdown. Concrete, technical, honest. Explain
   what/why/how with real field names, defaults, ranges, enum values, file paths
   (use backticks: `sim_env/environment.py`). Use tables for settings/parameters
   (| Name | Type | Default | Description |). Use ```mermaid diagrams where they
   genuinely help (they render on GitHub). No marketing fluff, no invented claims.
   Mark unimplemented/declared-but-unused features explicitly.
6. **Page structure**: `# Title` → one-line purpose → sections → where relevant a
   "See also" footer with relative links. Keep pages scannable.
7. **Terminology**: the product is "Simulation Studio" / "Autonomous Systems Simulator".
   The central concept: *Simulator = Environment, AI = Agent*. Version facts:
   SIMULATOR_VERSION 4.0.0; TCP protocol 2.1 (supports 2.0/2.1); worker protocol 1.0;
   manifest 1.0; trainer contract 1.0; dataset format transitions_v1; .sim.json
   schema 2.0.0 (3.0.0 recognized); agentRL version 0.1.0 (partial).
8. Do not create files outside your assigned list. Do not modify README.md or
   docs/README.md (handled separately). Do not edit code.
