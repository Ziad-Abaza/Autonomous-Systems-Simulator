"""Check suspected defect: sideways slide never decays once vx < 1.5 m/s
(kinematic blend keyed on |vx| only -> ay forced to 0 while vy persists)."""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_core.math_utils import Vec3
from sim_core.vehicle.vehicle_model import VehicleModel

car = VehicleModel()
dt = 1/60
car.reset(Vec3(0,0,0), 0.0, initial_speed=40.0)
# provoke a massive slide
for _ in range(int(1.5/dt)):
    car.step(0.25, 0.9, 0.0, dt)
# release, coast, watch vx/vy for 15 s
print("t    vx      vy      wz(deg)  |vy| hist")
t = 0.0
for i in range(int(15/dt)):
    car.step(0.0, 0.0, 0.0, dt)
    t += dt
    if i % 30 == 0:
        st = car.state
        print(f"{t:4.1f} {st.vel_body.x:6.2f} {st.vel_body.y:7.2f} {math.degrees(st.yaw_rate):7.1f}")
st = car.state
print(f"final: vx={st.vel_body.x:.2f} vy={st.vel_body.y:.2f} speed_total={st.speed_total:.2f}")
print(f"pos drifted to x={st.pos.x:.1f} y={st.pos.y:.1f}")
