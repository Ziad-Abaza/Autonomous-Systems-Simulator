"""Env-level sanity: surface friction regions, banking bias, collision response."""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
from sim_env.environment import SimulationEnvironment
from sim_core.track.road_definition import RoadDefinition
from sim_core.math_utils import Vec2

def drive_on(road, seconds=6.0, throttle=0.8):
    env = SimulationEnvironment(road_def=road)
    env.reset()
    dt = 1/60
    last = None
    for i in range(int(seconds/dt)):
        obs, r, term, trunc, info = env.step([0.0, throttle, 0.0])
        last = env.vehicle.state
        if term or trunc:
            break
    return env, last

# --- banking test: banked vs flat corner ---
# Straight road along +X, banked 15 deg: car should drift toward -normal.
road = RoadDefinition(name="banked", is_closed=False)
for i in range(12):
    road.add_control_point(x=i*30.0, y=0.0, z=0.0, width=14.0, banking=15.0)
road.spawn_point.x = 0.0; road.spawn_point.y = 0.0; road.spawn_point.yaw = 0.0
road.spawn_point.initial_speed = 15.0
env_b, st_b = drive_on(road, seconds=3.0)

road2 = RoadDefinition(name="flat", is_closed=False)
for i in range(12):
    road2.add_control_point(x=i*30.0, y=0.0, z=0.0, width=14.0, banking=0.0)
road2.spawn_point.x = 0.0; road2.spawn_point.y = 0.0; road2.spawn_point.yaw = 0.0
road2.spawn_point.initial_speed = 15.0
env_f, st_f = drive_on(road2, seconds=3.0)
print(f"banked 15deg: lateral drift y={st_b.pos.y:.2f} m (expect <0 = pushed right)")
print(f"flat        : lateral drift y={st_f.pos.y:.2f} m")

# --- grade test: uphill slows the car ---
road3 = RoadDefinition(name="hill", is_closed=False)
for i in range(12):
    road3.add_control_point(x=i*30.0, y=0.0, z=i*2.0, width=14.0, banking=0.0)
road3.spawn_point.x = 0.0; road3.spawn_point.y = 0.0; road3.spawn_point.yaw = 0.0
road3.spawn_point.initial_speed = 15.0
env_h, st_h = drive_on(road3, seconds=3.0)
print(f"uphill ~6.7%: final vx={st_h.vel_body.x:.2f} (flat was {st_f.vel_body.x:.2f})")

# --- collision: drive straight into a boundary wall ---
env = SimulationEnvironment()  # default oval
env.reset()
env.vehicle.reset(pos=__import__('sim_core.math_utils', fromlist=['Vec3']).Vec3(60,0,0.1), yaw=math.pi/2, initial_speed=20.0)
# aim it at the +X outer boundary: yaw 0 drives +x toward boundary
env.vehicle.state.yaw = 0.0
env.vehicle.state.vel_body = Vec2(20.0, 0.0)
import sim_core.math_utils as mu
env.vehicle.state.vel_world = mu.Vec2(20.0, 0.0)
pre_ke = 0.5*1200*(20.0**2)
hit = False
for i in range(300):
    obs, r, term, trunc, info = env.step([0.0, 0.5, 0.0])
    if env.vehicle.state.is_colliding or info.get('is_colliding'):
        hit = True
        break
    if term or trunc:
        break
st = env.vehicle.state
post_ke = 0.5*1200*(st.speed_total**2)
print(f"wall hit at step {i}: colliding={st.is_colliding} hit_flag={hit} "
      f"term={term} reason={info.get('termination_reason')} "
      f"speed={st.speed_total:.1f} KE {pre_ke:.0f}->{post_ke:.0f} J")

# --- off-road/curb friction: park the car beyond half-width and confirm mu drop ---
surf_road = env.track_queries.query_surface(mu.Vec2(60.0, 0.0))  # on road (spawn)
print("surface at spawn:", surf_road['region'], surf_road['friction_mult'])
# outside boundary
surf_off = env.track_queries.query_surface(mu.Vec2(60.0, 30.0))
print("surface outside :", surf_off['region'], surf_off['friction_mult'])
