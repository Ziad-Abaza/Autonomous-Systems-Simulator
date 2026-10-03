"""High-speed recovery audit — does a slide recover on countersteer?"""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_model import VehicleModel

def state(car):
    st = car.state
    vx, vy = st.vel_body.x, st.vel_body.y
    beta = math.degrees(math.atan2(vy, max(0.1, vx))) if vx*vx+vy*vy > 1 else 0.0
    return beta, math.degrees(st.yaw_rate), vx, st.accel_body.y

# Scenario: 40 m/s, hold steer that saturates, then try to recover
for label, phases in [
    ("hold 0.10 then release",      [(1.5, 0.10), (3.0, 0.0)]),
    ("hold 0.10 then countersteer", [(1.5, 0.10), (3.0, -0.06)]),
    ("hold 0.08 then release",      [(1.5, 0.08), (3.0, 0.0)]),
    ("hold 0.08 then countersteer", [(1.5, 0.08), (3.0, -0.05)]),
    ("0.06 sustained",              [(6.0, 0.06)]),
    ("0.04 sustained",              [(6.0, 0.04)]),
]:
    for v in (30, 40):
        car = VehicleModel()
        dt = 1/60
        car.reset(Vec3(0, 0, 0), 0.0, initial_speed=v)
        thr = min(1.0, max(0.15, v/45.0))
        for _ in range(int(1.0/dt)):
            car.step(0.0, thr, 0.0, dt)
        pk = 0.0
        t = 0.0
        trace = []
        for dur, s in phases:
            for _ in range(int(dur/dt)):
                car.step(s, thr, 0.0, dt); t += dt
                b, wz, vx, ay = state(car)
                pk = max(pk, abs(b))
                trace.append((t, b, wz, vx))
            # print state at phase boundaries
        fb, fwz, fvx, _ = state(car)
        print(f"{label:32s} v={v} | peakBeta={pk:6.1f} finalBeta={fb:6.1f} "
              f"finalWz={fwz:6.1f}dps finalVx={fvx:5.1f}")
        # sample the trace
        samp = " ".join(f"t={tt:.1f}:b={bb:.0f}/w={ww:.0f}" for tt, bb, ww, _ in
                        [trace[int(len(trace)*k/8)] for k in range(8)])
        print("    ", samp)
    print()
