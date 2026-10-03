"""
Closed-loop Autonomous Driving Agent using PID lateral lane tracking
and LiDAR obstacle/cornering speed modulation.
Demonstrates autonomous vehicle navigation connecting to the simulator.
"""

from __future__ import annotations
import time
import math
import numpy as np
from sim_client.client import SimulationClient


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        # Anti-windup
        self.integral = max(-1.0, min(1.0, self.integral))
        derivative = (error - self.prev_error) / max(1e-4, dt)
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.limits[0], min(self.limits[1], output))


def run_pid_agent(host: str = "127.0.0.1", port: int = 8765, max_steps: int = 1000):
    """
    Connects to the simulator and drives autonomously.
    """
    print(f"[PID Driver] Connecting to simulator at {host}:{port}...")
    client = SimulationClient(host=host, port=port)
    spec = client.connect()
    print(f"[PID Driver] Connected! Track: {spec['track_name']}, Length: {spec['track_length']:.1f}m, Physics: {spec['physics_hz']}Hz")

    # Lateral steering PID controller
    # Error is lateral offset + heading error lookahead + LiDAR barrier centering
    steer_pid = PIDController(kp=0.85, ki=0.005, kd=0.05, output_limits=(-1.0, 1.0))
    speed_pid = PIDController(kp=0.25, ki=0.01, kd=0.05, output_limits=(0.0, 1.0))

    obs, info = client.reset()
    print(f"[PID Driver] Environment reset. Starting autonomous drive loop for {max_steps} steps...")

    dt = spec.get('dt', 1.0 / 60.0)
    total_reward = 0.0
    episodes = 1

    for step in range(max_steps):
        # Extract features from info dictionary
        heading_err = info.get('heading_error', 0.0)
        lat_offset = info.get('lateral_offset', 0.0)
        speed = info.get('speed', 0.0)

        # Extract LiDAR ranges for obstacle and corner preview (15 rays spanning 180 degrees)
        if isinstance(obs, np.ndarray) and len(obs) >= 22:
            lidar_rays = obs[7:22] * 40.0  # de-normalize from [0, 1] to meters
            left_dist = float(np.mean(lidar_rays[10:]))
            right_dist = float(np.mean(lidar_rays[:5]))
            center_dist = float(np.min(lidar_rays[5:10]))
        else:
            left_dist = 6.0
            right_dist = 6.0
            center_dist = 40.0

        # LiDAR centering assist (detects proximity to track walls)
        lidar_bias = (right_dist - left_dist) / 12.0

        # Composite lookahead steering error
        steer_error = (lat_offset * 0.35) + (heading_err * 0.75) + (lidar_bias * 0.2)
        steer_cmd = steer_pid.update(steer_error, dt)

        # Dynamic target speed modulates on upcoming corners and forward clearance
        lidar_speed_factor = max(0.0, min(1.0, (center_dist - 10.0) / 18.0))
        turn_speed_factor = max(0.0, 1.0 - abs(heading_err) * 1.8)
        curvature_factor = min(lidar_speed_factor, turn_speed_factor)

        target_speed = 10.5 + 8.5 * curvature_factor
        speed_error = target_speed - speed

        if speed_error > 0:
            throttle_cmd = speed_pid.update(speed_error, dt)
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.0
            brake_cmd = min(0.7, -speed_error * 0.15)

        action = [float(steer_cmd), float(throttle_cmd), float(brake_cmd)]
        obs, reward, terminated, truncated, info = client.step(action)
        total_reward += reward

        if step % 60 == 0:
            print(f"Step {step:4d} | Speed: {speed:5.1f} m/s | LatErr: {lat_offset:5.2f}m | HeadErr: {math.degrees(heading_err):5.1f}° | Reward: {total_reward:6.1f} | Laps: {info.get('laps_completed', 0)}")

        if terminated or truncated:
            reason = info.get('termination_reason', 'unknown')
            print(f"[PID Driver] Episode {episodes} ended (Reason: {reason}). Total reward: {total_reward:.2f}. Resetting...")
            episodes += 1
            steer_pid.reset()
            speed_pid.reset()
            obs, info = client.reset()

    client.close()
    print(f"[PID Driver] Completed autonomous drive run of {max_steps} steps!")


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    run_pid_agent(host=host, port=port)
