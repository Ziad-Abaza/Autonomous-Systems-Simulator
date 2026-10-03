"""
Standalone 3D AI Simulation Platform Launcher.
Supports interactive 3D visual studio mode and high-performance headless training mode.
"""

from __future__ import annotations
import sys
import os
import argparse

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
    parser.add_argument("--track", type=str, default="oval", help="Initial track: 'oval', 'serpentine', 'obstacle', or path to .sim.json")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    return parser.parse_args()


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
    print(f"Initial Environment: {args.track}", flush=True)
    print("=" * 65, flush=True)

    app = SimulationStudioApp(
        width=args.width,
        height=args.height,
        headless=args.headless,
        port=args.port
    )

    # Load requested track preset if specified
    if args.track == "serpentine":
        proj = create_serpentine_track()
        app.env.set_road_definition(proj.road_def)
        if not args.headless:
            app.renderer.load_track(app.env.track)
    elif args.track == "obstacle":
        proj = create_obstacle_challenge()
        app.env.set_road_definition(proj.road_def)
        # Add obstacles
        from sim_core.world.obstacle import Obstacle
        for obs_data in proj.scenario_config.obstacles:
            app.env.add_obstacle(Obstacle.from_dict(obs_data))
        if not args.headless:
            app.renderer.load_track(app.env.track)
    elif os.path.exists(args.track):
        proj = EnvironmentProject.load(args.track)
        app.env.set_road_definition(proj.road_def)
        if not args.headless:
            app.renderer.load_track(app.env.track)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received. Shutting down...")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
