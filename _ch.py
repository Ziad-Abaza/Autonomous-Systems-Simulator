import json
from sim_project.serializer import EnvironmentProject
from sim_experiment.headless import build_env_from_dicts
p = EnvironmentProject.load("presets/oval_circuit.sim.json"); d=p.to_dict()
# enable the image channel to see authored contract
for c in d["agent"]["observation_space"]["channels"]:
    print(c["name"], c.get("type"), c.get("shape"), c.get("enabled"), c.get("source"))
