import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_model import VehicleModel

def run(hz, script, T, v0):
    car = VehicleModel()
    car.reset(Vec3(0, 0, 0), 0.0, initial_speed=v0)
    dt = 1.0 / hz
    n = int(T * hz)
    for i in range(n):
        car.step(*script(i * dt), dt)
    s = car.state
    return (s.pos.x, s.pos.y, s.yaw, s.vel_body.x, s.vel_body.y, s.yaw_rate)

scenarios = {
    'mild_corner': (lambda t: (0.1, 0.4, 0.0), 4.0, 15.0),
    'throttle_in_corner': (lambda t: (0.15, 1.0 if t > 2 else 0.3, 0.0), 5.0, 15.0),
    'sine_steer': (lambda t: (0.15 * math.sin(2 * math.pi * 0.5 * t), 0.45, 0.0), 4.0, 20.0),
    'drift': (lambda t: (0.8 if t < 2 else 0.0, 1.0 if t < 2 else 0.4, 0.0), 5.0, 8.0),
}

for name, (script, T, v0) in scenarios.items():
    print(f'== {name} ==')
    res = {}
    for hz in (30, 60, 120, 240):
        res[hz] = run(hz, script, T, v0)
    ref = res[240]
    for hz in (30, 60, 120):
        d = res[hz]
        dx = abs(d[0] - ref[0]); dy = abs(d[1] - ref[1]); dyaw = abs(d[2] - ref[2])
        print(f'  {hz:3d}Hz vs 240Hz: dx={dx:.3f} dy={dy:.3f} dyaw={math.degrees(dyaw):.2f}deg')

# determinism: bitwise identical runs
r1 = run(60, scenarios['drift'][0], 5.0, 8.0)
r2 = run(60, scenarios['drift'][0], 5.0, 8.0)
print('determinism bitwise equal:', r1 == r2)
