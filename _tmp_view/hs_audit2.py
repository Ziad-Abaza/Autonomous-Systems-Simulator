"""High-speed audit v2 — spawn AT speed via initial_speed."""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_model import VehicleModel

G = 9.81


def run(steer, v, dur=6.0, dt=1/60, throttle=None, mu=1.0, settle_s=1.0):
    car = VehicleModel()
    cfg = car.config
    if throttle is None:
        throttle = min(1.0, max(0.15, v / cfg.top_speed))
    car.reset(Vec3(0, 0, 0), 0.0, initial_speed=v)
    # settle: straight-line at cruise throttle to establish equilibrium
    for _ in range(int(settle_s / dt)):
        car.step(0.0, throttle, 0.0, dt, mu)
    v0 = car.state.vel_body.x
    peak_beta = peak_wz = peak_ay = 0.0
    peak_util_f = peak_util_r = 0.0
    betas = []
    wzs = []
    for i in range(int(dur / dt)):
        car.step(steer, throttle, 0.0, dt, mu)
        st = car.state
        vx, vy = st.vel_body.x, st.vel_body.y
        beta = math.atan2(vy, max(0.1, vx)) if vx*vx+vy*vy > 1 else 0.0
        betas.append(beta); wzs.append(st.yaw_rate)
        peak_beta = max(peak_beta, abs(beta))
        peak_wz = max(peak_wz, abs(st.yaw_rate))
        peak_ay = max(peak_ay, abs(st.accel_body.y))
        capf = cfg.tire_friction*mu*st.fz_front
        capr = cfg.tire_friction*mu*st.fz_rear
        peak_util_f = max(peak_util_f, math.hypot(st.fx_f, st.fy_f)/max(1, capf))
        peak_util_r = max(peak_util_r, math.hypot(st.fx_r, st.fy_r)/max(1, capr))
    tail = betas[-len(betas)//4:]
    return dict(v0=v0, vend=car.state.speed_total,
                peak_beta=math.degrees(peak_beta), steady_beta=math.degrees(sum(tail)/len(tail)),
                peak_wz=math.degrees(peak_wz), peak_ay=peak_ay,
                util_f=peak_util_f, util_r=peak_util_r,
                final_yaw=math.degrees(car.state.yaw),
                wz_end=math.degrees(wzs[-1]))


print(f"{'steer':>6} {'v':>5} {'v0':>5} {'vend':>5} {'pkBeta':>7} {'ssBeta':>7} "
      f"{'pkWz':>6} {'pkAy':>6} {'utF':>5} {'utR':>5} {'yawEnd':>8} {'wzEnd':>7}")
for steer in (0.02, 0.05, 0.08, 0.10):
    for v in (20, 30, 40, 44):
        r = run(steer, v)
        print(f"{steer:6.2f} {v:5.0f} {r['v0']:5.1f} {r['vend']:5.1f} "
              f"{r['peak_beta']:7.1f} {r['steady_beta']:7.1f} {r['peak_wz']:6.1f} "
              f"{r['peak_ay']:6.2f} {r['util_f']:5.2f} {r['util_r']:5.2f} "
              f"{r['final_yaw']:8.1f} {r['wz_end']:7.1f}")
    print()
