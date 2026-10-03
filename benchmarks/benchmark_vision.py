"""
Vision Pipeline Performance Benchmark.
Profiles camera 3D render pass, GPU -> CPU pixel transfer (Legacy vs Pre-allocated read_into vs PBO),
procedural software rasterizer fallback, and end-to-end environment step latency.
"""

from __future__ import annotations
import math
import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl
from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition
from sim_core.track.mesh_generator import TrackMeshGenerator
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.sensors.camera_sensor import CameraSensor
from sim_render.renderer import SimulationRenderer3D
from sim_render.camera import SimulationCamera
from sim_render.offscreen import OffscreenFBO
from sim_env.environment import SimulationEnvironment
from sim_env.spaces import ObservationSchema


def run_vision_benchmark(iterations: int = 500) -> Dict[str, Any]:
    print("=" * 75)
    print(f"  RUNNING VISION PIPELINE LATENCY BENCHMARK ({iterations} frames)")
    print("=" * 75)

    ctx = moderngl.create_standalone_context()
    w, h = 84, 84

    # 1. Setup Models & Road
    road = RoadDefinition.create_default_oval()
    track = TrackMeshGenerator.generate(road)
    vehicle = VehicleModel(VehicleConfig())
    vehicle.reset(pos=Vec3(road.spawn_point.x, road.spawn_point.y, 0.1), yaw=road.spawn_point.yaw)

    renderer = SimulationRenderer3D(ctx)
    renderer.load_track(track)
    agent_cam = SimulationCamera()

    # Offscreen FBOs
    fbo_standard = OffscreenFBO(ctx, width=w, height=h, use_pbo=False)
    fbo_pbo = OffscreenFBO(ctx, width=w, height=h, use_pbo=True)

    # --- Benchmark 1: Procedural Perspective Road Rasterizer (Software) ---
    cam_sensor = CameraSensor(width=w, height=h)
    ctx_sensor = {
        'vehicle': vehicle,
        'track_queries': track.broadphase or track,  # dummy queries
    }
    from sim_core.track.track_queries import TrackSpatialQueries
    ctx_sensor['track_queries'] = TrackSpatialQueries(track)
    rng = np.random.default_rng(42)

    # Warmup
    for _ in range(50):
        _ = cam_sensor._synthesize_perspective_road(ctx_sensor)

    t0 = time.perf_counter()
    for _ in range(iterations):
        img_soft = cam_sensor._synthesize_perspective_road(ctx_sensor)
    t1 = time.perf_counter()
    soft_time_ms = (t1 - t0) / iterations * 1000.0

    # --- Benchmark 2: 3D OpenGL Scene Render Pass Time ---
    # Setup camera pose
    agent_cam.pos = Vec3(vehicle.state.pos.x, vehicle.state.pos.y, 1.2)
    agent_cam.target = Vec3(vehicle.state.pos.x + 20.0, vehicle.state.pos.y, 1.0)
    agent_cam.fov_degrees = 75.0

    fbo_standard.bind()
    # Warmup
    for _ in range(30):
        renderer.render_frame(
            vehicle=vehicle,
            track=track,
            obstacles=[],
            sensors=None,
            checkpoints=[],
            current_cp_idx=0,
            viewport_width=w,
            viewport_height=h,
            show_lidar_rays=False,
            show_trajectory=False,
            show_checkpoints=False,
            camera=agent_cam
        )
    ctx.finish()

    t0 = time.perf_counter()
    for _ in range(iterations):
        fbo_standard.bind()
        renderer.render_frame(
            vehicle=vehicle,
            track=track,
            obstacles=[],
            sensors=None,
            checkpoints=[],
            current_cp_idx=0,
            viewport_width=w,
            viewport_height=h,
            show_lidar_rays=False,
            show_trajectory=False,
            show_checkpoints=False,
            camera=agent_cam
        )
    ctx.finish()
    t1 = time.perf_counter()
    render_only_ms = (t1 - t0) / iterations * 1000.0

    # --- Benchmark 3: GPU -> CPU Transfer Paths ---
    # 3a. Legacy fbo.read()
    t0 = time.perf_counter()
    for _ in range(iterations):
        img_legacy = fbo_standard.read_rgb_legacy()
    t1 = time.perf_counter()
    transfer_legacy_ms = (t1 - t0) / iterations * 1000.0

    # 3b. Preallocated read_into()
    t0 = time.perf_counter()
    for _ in range(iterations):
        img_direct = fbo_standard.read_rgb()
    t1 = time.perf_counter()
    transfer_direct_ms = (t1 - t0) / iterations * 1000.0

    # 3c. Double-buffered PBO
    t0 = time.perf_counter()
    for _ in range(iterations):
        img_pbo = fbo_pbo.read_rgb()
    t1 = time.perf_counter()
    transfer_pbo_ms = (t1 - t0) / iterations * 1000.0

    # --- Benchmark 4: Full Environment Step Latency (Vector vs Vision) ---
    # Vector observation env
    schema_vec = ObservationSchema(include_camera_rgb=False)
    env_vec = SimulationEnvironment(observation_schema=schema_vec)
    env_vec.reset()
    t0 = time.perf_counter()
    for _ in range(iterations):
        env_vec.step([0.1, 0.5, 0.0])
    t1 = time.perf_counter()
    step_vec_ms = (t1 - t0) / iterations * 1000.0

    # Vision observation env (using procedural synthesis)
    schema_vis = ObservationSchema(include_camera_rgb=True)
    env_vis = SimulationEnvironment(observation_schema=schema_vis)
    env_vis.reset()
    t0 = time.perf_counter()
    for _ in range(iterations):
        env_vis.step([0.1, 0.5, 0.0])
    t1 = time.perf_counter()
    step_vis_ms = (t1 - t0) / iterations * 1000.0

    # Verification of shapes and dtypes
    assert img_soft.shape == (h, w, 3), f"Invalid soft shape: {img_soft.shape}"
    assert img_direct.shape == (h, w, 3), f"Invalid direct shape: {img_direct.shape}"
    assert img_direct.dtype == np.uint8, f"Invalid direct dtype: {img_direct.dtype}"

    results = {
        'frame_dimensions': f"{w}x{h}x3",
        'software_procedural_ms': round(soft_time_ms, 3),
        'software_fps': round(1000.0 / max(1e-6, soft_time_ms), 1),
        'opengl_3d_render_only_ms': round(render_only_ms, 3),
        'transfer_legacy_fbo_read_ms': round(transfer_legacy_ms, 3),
        'transfer_preallocated_read_into_ms': round(transfer_direct_ms, 3),
        'transfer_double_buffered_pbo_ms': round(transfer_pbo_ms, 3),
        'total_3d_frame_latency_ms': round(render_only_ms + transfer_direct_ms, 3),
        'total_3d_fps': round(1000.0 / max(1e-6, render_only_ms + transfer_direct_ms), 1),
        'env_step_vector_only_ms': round(step_vec_ms, 3),
        'env_step_vector_sps': round(1000.0 / max(1e-6, step_vec_ms), 1),
        'env_step_vision_ms': round(step_vis_ms, 3),
        'env_step_vision_sps': round(1000.0 / max(1e-6, step_vis_ms), 1),
        'transfer_speedup_vs_legacy': round(transfer_legacy_ms / max(1e-6, transfer_direct_ms), 2),
    }

    print("\n--- RESULTS ---")
    print(f"Software Procedural Camera:     {results['software_procedural_ms']:6.3f} ms  ({results['software_fps']} FPS)")
    print(f"OpenGL 3D Render Pass:          {results['opengl_3d_render_only_ms']:6.3f} ms")
    print(f"Transfer (Legacy fbo.read()):   {results['transfer_legacy_fbo_read_ms']:6.3f} ms")
    print(f"Transfer (Preallocated read):   {results['transfer_preallocated_read_into_ms']:6.3f} ms  ({results['transfer_speedup_vs_legacy']}x faster transfer)")
    print(f"Transfer (Double-buffered PBO): {results['transfer_double_buffered_pbo_ms']:6.3f} ms")
    print(f"Total 3D Camera Frame:          {results['total_3d_frame_latency_ms']:6.3f} ms  ({results['total_3d_fps']} FPS)")
    print(f"Full Env Step (Vector Obs):     {results['env_step_vector_only_ms']:6.3f} ms  ({results['env_step_vector_sps']} SPS)")
    print(f"Full Env Step (Vision Obs):     {results['env_step_vision_ms']:6.3f} ms  ({results['env_step_vision_sps']} SPS)")

    # Save to disk
    out_file = os.path.join("benchmarks", "vision_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Vision results written to {out_file}\n")

    fbo_standard.release()
    fbo_pbo.release()
    return results


if __name__ == "__main__":
    run_vision_benchmark()
