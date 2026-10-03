from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.vehicle_config import VehicleConfig
import math

# Pacejka4: constant moderate steer at 20 m/s — sanity vs bicycle
for tm in ("bicycle", "pacejka4"):
    v = VehicleModel(VehicleConfig(tire_model=tm))
    v.reset(type("P",(),{"x":0.0,"y":0.0,"z":0.0})(), 0.0, 20.0)
    for i in range(600):
        v.step(-0.10, 0.0, 0.0, 1/60)
    st = v.state
    print(tm, "| beta:", round(math.degrees(math.atan2(st.vel_body.y, max(st.vel_body.x,0.1))),2),
          "| wz:", round(st.yaw_rate,3), "| speed:", round(st.speed_total,1),
          "| fy_f:", round(st.fy_f), "| fz:", [round(f) for f in st.fz_wheels] if tm=="pacejka4" else "")
# post-peak decay probe: force at alpha 0.15 vs 0.5
v2 = VehicleModel(VehicleConfig(tire_model="pacejka4"))
v2.reset(type("P",(),{"x":0.0,"y":0.0,"z":0.0})(), 0.0, 20.0)
for i in range(900):
    v2.step(-0.25, 0.0, 0.0, 1/60)
st2 = v2.state
print("pacejka4 big-steer | beta:", round(math.degrees(math.atan2(st2.vel_body.y, max(st2.vel_body.x,0.1))),2),
      "| alpha_w:", [round(math.degrees(a),1) for a in st2.alpha_w_eff],
      "| fy_w:", [round(f) for f in st2.fy_wheels])
