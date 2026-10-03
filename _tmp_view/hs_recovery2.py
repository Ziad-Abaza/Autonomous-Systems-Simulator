"""Recovery with PROPER countersteer: steer_cmd sign that moves wheels
toward the velocity direction (desaturates front axle).

Convention: target_steer = -steer_cmd*0.58. Slide develops vy>0 (beta>0)
from positive steer_cmd. Front alpha = slip_angle - delta, so to null
alpha_f we need delta ~ +beta i.e. steer_cmd ~ -beta/0.58.
"""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_model import VehicleModel

def beta_of(car):
    st = car.state
    vx, vy = st.vel_body.x, st.vel_body.y
    return math.atan2(vy, max(0.1, vx)) if vx*vx+vy*vy > 1 else 0.0

for v in (40,):
    for cs in (0.0, -0.3, -0.5, -0.7):
        car = VehicleModel()
        dt = 1/60
        car.reset(Vec3(0,0,0), 0.0, initial_speed=v)
        thr = min(1.0, max(0.15, v/45.0))
        for _ in range(60):
            car.step(0.0, thr, 0.0, dt)
        # provoke: hold steer 0.10 for 1.5 s -> saturated slide
        for _ in range(int(1.5/dt)):
            car.step(0.10, thr, 0.0, dt)
        b_pk = math.degrees(beta_of(car))
        # recovery phase: countersteer cs for 4 s
        minb = 99.0
        recovered_t = None
        for i in range(int(4/dt)):
            car.step(cs, thr, 0.0, dt)
            b = beta_of(car)
            if recovered_t is None and abs(b) < math.radians(5):
                recovered_t = i*dt
        b_end = math.degrees(beta_of(car))
        wz = math.degrees(car.state.yaw_rate)
        print(f"v={v} cs={cs:+.2f} | beta_at_release={b_pk:5.1f} -> final={b_end:5.1f} "
              f"wz_end={wz:6.1f}dps recovered_in={recovered_t}")

print()
print("--- proportional countersteer: steer_cmd = -beta/0.58 (clip [-1,1]) ---")
for v in (30, 40):
    for provoke in (0.08, 0.10):
        car = VehicleModel()
        dt = 1/60
        car.reset(Vec3(0,0,0), 0.0, initial_speed=v)
        thr = min(1.0, max(0.15, v/45.0))
        for _ in range(60):
            car.step(0.0, thr, 0.0, dt)
        for _ in range(int(1.5/dt)):
            car.step(provoke, thr, 0.0, dt)
        b_pk = math.degrees(beta_of(car))
        recovered_t = None
        for i in range(int(4/dt)):
            b = beta_of(car)
            cs = max(-1.0, min(1.0, -b/0.58))
            car.step(cs, thr, 0.0, dt)
            if recovered_t is None and abs(b) < math.radians(5):
                recovered_t = i*dt
        b_end = math.degrees(beta_of(car))
        wz = math.degrees(car.state.yaw_rate)
        print(f"v={v} provoke={provoke:.2f} | beta_at_release={b_pk:5.1f} -> "
              f"final={b_end:5.1f} wz_end={wz:6.1f}dps recovered_in={recovered_t}")
