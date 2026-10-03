import numpy as np
from sim_env.templates import EnvironmentTemplateManager as M
from sim_experiment.headless import build_env_from_dicts
m=M()
for tid in ["empty","straight_sprint","slalom","basic_driving"]:
    p=m.create_project_from_template(tid); d=p.to_dict()
    env=build_env_from_dicts(d, d.get("scenario_def") or {}, seed=1)
    env.reset(seed=1)
    steps=0; info={}
    for i in range(2500):
        _,r,t,tr,info=env.step(np.array([0.0,0.45,0.0])); steps+=1
        if t or tr: break
    print("%-18s %4d steps -> %s" % (tid, steps, info.get("termination_reason")))
