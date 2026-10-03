from sim_env.environment import SimulationEnvironment
from sim_env.agent import AgentDefinition
env = SimulationEnvironment(agent=AgentDefinition.create_default_vehicle_agent())
env.reset(seed=1)
env.step([0,0.3,0])
samples = env.sensors.get_all_samples()
for name, s in samples.items():
    print(name, "->", sorted(s.keys()) if isinstance(s,dict) else type(s))
