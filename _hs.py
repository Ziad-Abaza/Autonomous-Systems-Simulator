from sim_project.serializer import EnvironmentProject
from sim_experiment.headless import build_env_from_dicts
p = EnvironmentProject.load("presets/oval_circuit.sim.json"); d=p.to_dict()
env = build_env_from_dicts(d, d.get("scenario_def") or {}, seed=1)
ag = env.agent
print("action export_schema:", ag.action_space.export_schema())
print("obs export_schema keys:", list(ag.observation_space.export_schema().keys())[:10])
print("obs export_schema:", ag.observation_space.export_schema())
obs,_ = env.reset(seed=1)
print("obs shape:", obs.shape, obs.dtype)
print("action_space def dims:", ag.action_space)
