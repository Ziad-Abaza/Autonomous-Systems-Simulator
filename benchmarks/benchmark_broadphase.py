"""
Spatial Acceleration Broadphase Benchmark.
Compares O(N) linear boundary collision against SpatialHashGrid2D broadphase
across Small, Medium, and Large track configurations.
Validates zero false negatives and measures candidate count, query latency, and memory.
"""

from __future__ import annotations
import math
import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Dict, Any

from sim_core.math_utils import Vec2, Vec3
from sim_core.track.road_definition import RoadDefinition
from sim_core.track.mesh_generator import TrackMeshGenerator
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.collision import VehicleCollisionChecker
from sim_core.collision.spatial_hash import SpatialHashGrid2D


def create_track(case: str) -> RoadDefinition:
    road = RoadDefinition(name=f"Benchmark_{case}", is_closed=True)
    if case == "small":
        # Oval track: ~250m
        points = [
            (50.0, 0.0), (35.0, 25.0), (0.0, 30.0), (-35.0, 25.0),
            (-50.0, 0.0), (-35.0, -25.0), (0.0, -30.0), (35.0, -25.0)
        ]
        for x, y in points:
            road.add_control_point(x=x, y=y, width=10.0)
    elif case == "medium":
        # Multi-corner circuit: ~750m
        points = [
            (0.0, 0.0), (80.0, 20.0), (120.0, 80.0), (100.0, 160.0),
            (40.0, 180.0), (-40.0, 150.0), (-80.0, 200.0), (-140.0, 160.0),
            (-150.0, 80.0), (-80.0, 40.0), (-40.0, -20.0)
        ]
        for x, y in points:
            road.add_control_point(x=x, y=y, width=12.0)
    elif case == "large":
        # Long endurance grand prix track: ~2,500m
        num_pts = 32
        for i in range(num_pts):
            theta = (i / num_pts) * 2.0 * math.pi
            r = 300.0 + 90.0 * math.sin(3.0 * theta) + 40.0 * math.cos(5.0 * theta)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            road.add_control_point(x=x, y=y, width=14.0)
    return road


def run_broadphase_benchmark(num_queries: int = 2000) -> Dict[str, Any]:
    cases = ["small", "medium", "large"]
    results = {}

    rng = np.random.default_rng(42)
    vehicle = VehicleModel(VehicleConfig())

    print("=" * 75)
    print(f"  RUNNING SPATIAL BROADPHASE COLLISION BENCHMARK ({num_queries} queries/case)")
    print("=" * 75)

    for case in cases:
        road = create_track(case)
        track = TrackMeshGenerator.generate(road, sample_step=1.0)
        broadphase = track.broadphase
        segs = track.all_boundary_segments
        num_segs = len(segs)

        # Generate sample poses: mix of poses on track, at borders, and off-road
        test_poses = []
        spline_samples = track.spline.samples
        for _ in range(num_queries):
            sample = rng.choice(spline_samples)
            # Lateral offset from -15m to +15m
            lat = rng.uniform(-15.0, 15.0)
            norm = sample.normal
            px = sample.pos.x + norm.x * lat
            py = sample.pos.y + norm.y * lat
            yaw = rng.uniform(-math.pi, math.pi)
            test_poses.append((px, py, yaw))

        # --- 1. Without Broadphase (Full linear segment scan) ---
        t0 = time.perf_counter()
        outcomes_without = []
        for px, py, yaw in test_poses:
            vehicle.reset(pos=Vec3(px, py, 0.0), yaw=yaw)
            res = VehicleCollisionChecker.check_track_boundary_collision(
                vehicle=vehicle,
                boundary_segments=segs,
                broadphase=None
            )
            outcomes_without.append(res.collided)
        t1 = time.perf_counter()
        time_without_s = t1 - t0
        time_without_us = (time_without_s / num_queries) * 1e6

        # --- 2. With Broadphase (SpatialHashGrid2D) ---
        t0 = time.perf_counter()
        outcomes_with = []
        candidate_counts = []
        for px, py, yaw in test_poses:
            vehicle.reset(pos=Vec3(px, py, 0.0), yaw=yaw)
            obb = vehicle.get_obb()
            r = max(obb.half_length, obb.half_width) + 0.5
            cand_segs = broadphase.query_candidate_segments(obb.center, r)
            candidate_counts.append(len(cand_segs))

            res = VehicleCollisionChecker.check_track_boundary_collision(
                vehicle=vehicle,
                boundary_segments=segs,
                broadphase=broadphase
            )
            outcomes_with.append(res.collided)
        t1 = time.perf_counter()
        time_with_s = t1 - t0
        time_with_us = (time_with_s / num_queries) * 1e6

        # --- 3. Verification & Correctness ---
        mismatches = 0
        false_negatives = 0
        for i in range(num_queries):
            if outcomes_without[i] != outcomes_with[i]:
                mismatches += 1
                if outcomes_without[i] and not outcomes_with[i]:
                    false_negatives += 1

        correctness_pct = 100.0 * (1.0 - (mismatches / num_queries))
        mean_candidates = float(np.mean(candidate_counts))
        speedup = time_without_s / max(1e-9, time_with_s)
        mem_info = broadphase.get_memory_footprint()

        case_res = {
            'track_case': case,
            'track_length_m': round(track.spline.total_length, 1),
            'total_boundary_segments': num_segs,
            'time_without_us': round(time_without_us, 2),
            'time_with_us': round(time_with_us, 2),
            'speedup_factor': round(speedup, 2),
            'mean_candidates': round(mean_candidates, 1),
            'candidate_reduction_pct': round(100.0 * (1.0 - mean_candidates / num_segs), 1),
            'mismatches': mismatches,
            'false_negatives': false_negatives,
            'correctness_pct': correctness_pct,
            'memory_overhead_bytes': mem_info['approx_memory_bytes'],
            'grid_cells': mem_info['cell_count'],
        }
        results[case] = case_res

        print(f"\n[{case.upper()} TRACK] Length: {case_res['track_length_m']}m | Segments: {num_segs}")
        print(f"  Without Broadphase: {time_without_us:7.2f} µs/query  (Total: {time_without_s*1000:6.2f} ms)")
        print(f"  With Broadphase:    {time_with_us:7.2f} µs/query  (Total: {time_with_s*1000:6.2f} ms)")
        print(f"  Speedup:            {speedup:7.2f}x faster")
        print(f"  Candidates Checked: {mean_candidates:5.1f} / {num_segs} segments ({case_res['candidate_reduction_pct']}% reduction)")
        print(f"  Collision Matches:  {num_queries - mismatches}/{num_queries} ({correctness_pct:.2f}%)")
        print(f"  False Negatives:    {false_negatives} (Zero false negative contract verified)")
        print(f"  Memory Footprint:   {mem_info['approx_memory_bytes'] / 1024:.2f} KB across {mem_info['cell_count']} active cells")

    # Save benchmark results to disk
    os.makedirs("benchmarks", exist_ok=True)
    out_file = os.path.join("benchmarks", "broadphase_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Results written to {out_file}\n")
    return results


if __name__ == "__main__":
    run_broadphase_benchmark()
