import numpy as np
from sim_project.serializer import EnvironmentProject
from sim_experiment.headless import build_env_from_dicts
# 1) normal build still works
p = EnvironmentProject.load("presets/oval_circuit.sim.json"); d=p.to_dict()
env = build_env_from_dicts(d, d.get("scenario_def") or {}, seed=1); env.reset()
print("normal build OK, obs dim:", env.reset()[0].shape)
# 2) disable lidar -> build must refuse
d2 = p.to_dict()
for s in d2["agent"]["sensor_configs"]:
    if s["name"]=="lidar_rays": s["enabled"]=False
try:
    build_env_from_dicts(d2, d2.get("scenario_def") or {}, seed=1)
    print("FAIL: orphan channel tolerated")
except Exception as e:
    print("refused:", str(e)[:120])
# 3) disable lidar AND channel -> builds fine, no dead column
d3 = p.to_dict()
for s in d3["agent"]["sensor_configs"]:
    if s["name"]=="lidar_rays": s["enabled"]=False
d3["agent"]["observation_space"]["channels"]=[c for c in d3["agent"]["observation_space"]["channels"] if c["name"]!="lidar_ranges"]
env3=build_env_from_dicts(d3, d3.get("scenario_def") or {}, seed=1); o,_=env3.reset()
print("consistent build OK, obs dim:", o.shape)
