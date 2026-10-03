"""
Standalone 3D AI Simulation Platform Launcher.
Supports interactive 3D visual studio mode and high-performance headless training mode.
"""

from __future__ import annotations
import sys
import os
import argparse

# In windowed executable mode, sys.stdout and sys.stderr may be None
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Ensure current directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_ui.app import SimulationStudioApp
from sim_project.presets import (
    create_oval_circuit,
    create_serpentine_track,
    create_obstacle_challenge,
    save_default_presets
)
from sim_project.serializer import EnvironmentProject


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone 3D AI Simulation Platform")
    parser.add_argument("--headless", action="store_true", help="Run in headless simulation mode without GUI")
    parser.add_argument("--port", type=int, default=8765, help="TCP port for external AI model connection (default: 8765)")
    parser.add_argument("--num-envs", type=int, default=1, help="Headless only: host N independent envs on --port via one multi-client server (default: 1)")
    parser.add_argument("--track", type=str, default=None, help="Open this track directly into the workspace: 'oval', 'serpentine', 'obstacle', or path to .sim.json (default: studio home)")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    return parser.parse_args()


def _run_multi_headless(port: int, num_envs: int, track: Optional[str]) -> int:
    """Single-process, multi-env headless server for env_mode='tcp_multi'."""
    import time
    from typing import Optional
    from sim_net.multi_server import SimServerMulti
    from sim_env.environment import SimulationEnvironment

    if track == "serpentine":
        proj = create_serpentine_track()
    elif track and os.path.exists(track):
        proj = EnvironmentProject.load(track)
    else:
        proj = create_oval_circuit()

    def make_env(i):
        return SimulationEnvironment(
            road_def=proj.road_def,
            agent=proj.agent,
            scenario_def=proj.scenario_def,
            seed=42 + i,
        )

    srv = SimServerMulti(
        env_factories=[(lambda i=i: make_env(i)) for i in range(num_envs)],
        host="127.0.0.1", port=port)
    if not srv.start():
        return 1
    print(f"[HeadlessMulti] {num_envs} envs listening on port {port}", flush=True)
    try:
        while True:
            srv.poll_and_process()
            time.sleep(0.0005)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


def main():
    args = parse_args()

    # Pre-generate preset files if not already created
    preset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    if not os.path.exists(preset_dir):
        save_default_presets(preset_dir)

    print("=" * 65, flush=True)
    print("      3D AI ENVIRONMENT SIMULATION PLATFORM", flush=True)
    print("      Domain: Autonomous Vehicle Reinforcement Learning", flush=True)
    print("=" * 65, flush=True)
    print(f"Mode: {'Headless Training' if args.headless else 'Interactive 3D Studio'}", flush=True)
    print(f"External AI Server Port: {args.port}", flush=True)
    print(f"Initial Environment: {args.track or 'studio home (oval)'}", flush=True)
    print("=" * 65, flush=True)

    if args.headless and args.num_envs > 1:
        sys.exit(_run_multi_headless(args.port, args.num_envs, args.track))

    app = SimulationStudioApp(
        width=args.width,
        height=args.height,
        headless=args.headless,
        port=args.port
    )

    # --track opens the requested environment directly into the workspace;
    # without it the studio starts on the home/library screen.
    if args.track:
        proj = None
        if args.track == "serpentine":
            proj = create_serpentine_track()
        elif args.track == "obstacle":
            proj = create_obstacle_challenge()
        elif args.track == "oval":
            proj = create_oval_circuit()
        elif os.path.exists(args.track):
            proj = EnvironmentProject.load(args.track)
            app.open_project(proj, path=args.track if not args.headless else None)
            proj = None
        if proj is not None:
            app.open_project(proj, None)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received. Shutting down...")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
