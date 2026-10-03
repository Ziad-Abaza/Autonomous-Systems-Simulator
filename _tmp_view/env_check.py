import json, sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_core.track.road_definition import RoadDefinition
from sim_env.environment import SimulationEnvironment

d = json.load(open('tracks/lane_following_serpentine_circuit.sim.json'))
env = SimulationEnvironment(road_def=RoadDefinition.from_dict(d['road_definition']))
env.reset()
random.seed(1)
steps = 0
reason = 'running'
for i in range(1200):
    steer = 0.6 * math.sin(i / 40.0)
    obs, r, term, trunc, info = env.step([steer, 1.0, 0.0])
    steps += 1
    if term or trunc:
        reason = info.get('termination_reason', 'done')
        break
st = env.vehicle.state
beta = math.degrees(math.atan2(st.vel_body.y, max(0.1, st.vel_body.x)))
print(f'steps={steps} reason={reason} colliding={st.is_colliding}')
print(f'final: speed={st.speed_total:.1f} beta={beta:+.1f} pos.z={st.pos.z:.2f}')
print('all finite:', all(math.isfinite(v) for v in
      [st.pos.x, st.pos.y, st.vel_body.x, st.vel_body.y, st.yaw_rate]))
