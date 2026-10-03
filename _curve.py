import math
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.math_utils import Vec2, Vec3

# Direct curve check: sweep alpha via controlled states on _pacejka4
v = VehicleModel(VehicleConfig(tire_model="pacejka4"))
v.reset(Vec3(0,0,0.2), 0.0, 20.0)
st = v.state
st.accel_body = Vec2(0.0, 0.0)
g = 9.81
fz_f = 1200*g*0.48; fz_r = 1200*g*0.52
out = []
for vy in [1.0, 3.0, 6.0, 10.0, 15.0, 22.0]:
    st.alpha_w_eff = [0.0]*4
    for _ in range(300):
        st.accel_body = Vec2(0.0, 0.0)
        v._pacejka4(20.0, vy, 0.0, 20.0, fz_f, fz_r, fz_f, fz_r, 1.0,
                    80000.0, 85000.0, 0.0, 0.0, 1/240, 20.0)
    a_deg = math.degrees(abs(st.alpha_w_eff[0] - 0))
    out.append((round(a_deg,1), round(-st.fy_wheels[0])))
print("alpha_deg -> Fy_FL:", out)
peak = max(f for _,f in out)
print("post-peak decay:", out[-1][1] < peak, "| peak:", peak, "| tail:", out[-1][1])
