"""High-speed handling audit — reproduce the reported loss of grip.

Runs step-steer and constant-steer maneuvers at increasing speeds and
reports sideslip, yaw rate, lateral accel, and per-axle friction
utilization so we can see WHERE grip is lost and WHY.
"""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.vehicle.vehicle_model import VehicleModel

G = 9.81


def run(steer, target_speed, dur=6.0, dt=1/60, throttle=None, mu=1.0):
    car = VehicleModel()
    cfg = car.config
    if throttle is None:
        throttle = min(1.0, max(0.15, target_speed / cfg.top_speed))
    # pre-roll: full throttle 5 s to reach speed, zero steer
    car.reset(Vec3(0, 0, 0), 0.0, 0.0)
    for _ in range(int(5.0 / dt)):
        car.step(0.0, 1.0, 0.0, dt, mu)
    v0 = car.state.vel_body.x
    peak_beta = peak_wz = peak_ay = 0.0
    peak_util_f = peak_util_r = 0.0
    max_ay_over_g = 0.0
    beta_series = []
    for i in range(int(dur / dt)):
        car.step(steer, throttle, 0.0, dt, mu)
        st = car.state
        vx, vy = st.vel_body.x, st.vel_body.y
        beta = math.atan2(vy, max(0.1, vx)) if vx*vx+vy*vy > 1 else 0.0
        beta_series.append(beta)
        peak_beta = max(peak_beta, abs(beta))
        peak_wz = max(peak_wz, abs(st.yaw_rate))
        ay = st.accel_body.y
        peak_ay = max(peak_ay, abs(ay))
        max_ay_over_g = max(max_ay_over_g, abs(ay)/G)
        capf = cfg.tire_friction*mu*st.fz_front
        capr = cfg.tire_friction*mu*st.fz_rear
        peak_util_f = max(peak_util_f, math.hypot(st.fx_f, st.fy_f)/max(1, capf))
        peak_util_r = max(peak_util_r, math.hypot(st.fx_r, st.fy_r)/max(1, capr))
    tail = beta_series[-len(beta_series)//4:]
    steady_beta = sum(tail)/len(tail)
    return dict(v0=v0, vend=car.state.speed_total, peak_beta=math.degrees(peak_beta),
                steady_beta=math.degrees(steady_beta), peak_wz=math.degrees(peak_wz),
                peak_ay=peak_ay, max_ay_g=max_ay_over_g,
                util_f=peak_util_f, util_r=peak_util_r,
                final_yaw=math.degrees(car.state.yaw))


print(f"{'steer':>6} {'v_tgt':>5} {'v0':>5} {'vend':>5} {'pkBeta':>7} {'ssBeta':>7} "
      f"{'pkWz':>6} {'pkAy':>6} {'ay/g':>5} {'utF':>5} {'utR':>5} {'yawEnd':>8}")
for steer in (0.05, 0.10, 0.15, 0.25):
    for vt in (20, 25, 30, 35, 40):
        r = run(steer, vt)
        print(f"{steer:6.2f} {vt:5.0f} {r['v0']:5.1f} {r['vend']:5.1f} "
              f"{r['peak_beta']:7.1f} {r['steady_beta']:7.1f} {r['peak_wz']:6.1f} "
              f"{r['peak_ay']:6.2f} {r['max_ay_g']:5.2f} {r['util_f']:5.2f} {r['util_r']:5.2f} "
              f"{r['final_yaw']:8.1f}")
    print()

# disturbance recovery at high speed
print("--- hands-off recovery after 150 ms full-lock kick ---")
for vt in (20, 30, 40):
    car = VehicleModel()
    dt = 1/60
    car.reset(Vec3(0, 0, 0), 0.0, 0.0)
    for _ in range(int(5/dt)):
        car.step(0.0, 1.0, 0.0, dt)
    v0 = car.state.vel_body.x
    thr = min(1.0, max(0.15, vt/45.0))
    pk = 0.0
    for i in range(int(7/dt)):
        t = i*dt
        s = 1.0 if 1.0 <= t < 1.15 else 0.0
        car.step(s, thr, 0.0, dt)
        st = car.state
        beta = math.degrees(math.atan2(st.vel_body.y, max(0.1, st.vel_body.x)))
        pk = max(pk, abs(beta))
    fb = math.degrees(math.atan2(car.state.vel_body.y, max(0.1, car.state.vel_body.x)))
    print(f"v0={v0:.1f} thr={thr:.2f} peakBeta={pk:.1f}deg finalBeta={fb:.1f}deg "
          f"finalWz={math.degrees(car.state.yaw_rate):.1f}dps")
