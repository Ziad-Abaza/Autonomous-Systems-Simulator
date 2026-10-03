"""Reproduce: (a) parked-car creep from gravity bias, (b) wall depenetration
direction when the car is at/past the boundary, (c) escape self-heal."""
import math, sys
sys.path.insert(0, r"D:\coding\projects\Simulation")
import sim_core.math_utils as mu
from sim_env.environment import SimulationEnvironment
from sim_core.track.road_definition import RoadDefinition
from sim_core.math_utils import Vec2, Vec3
from sim_core.vehicle.vehicle_model import VehicleModel
from sim_core.vehicle.collision import VehicleCollisionChecker

# --- (a) parked on a 15-deg banked road, zero inputs: does it creep? ---
road = RoadDefinition(name="banked", is_closed=False)
for i in range(12):
    road.add_control_point(x=i*30.0, y=0.0, z=0.0, width=14.0, banking=15.0)
road.spawn_point.x = 0.0; road.spawn_point.y = 0.0; road.spawn_point.yaw = 0.0
env = SimulationEnvironment(road_def=road)
env.reset()
for i in range(int(5/ (1/60))):
    env.step([0.0, 0.0, 0.0])
st = env.vehicle.state
print(f"(a) parked 5s on 15deg bank: speed={st.speed_total:.4f} m/s "
      f"drift=({st.pos.x:+.2f},{st.pos.y:+.2f}) m  [expect ~0]")

# parked on a STEEP bank where tan(theta)>mu -> should slide
road2 = RoadDefinition(name="steep", is_closed=False)
for i in range(12):
    road2.add_control_point(x=i*30.0, y=0.0, z=0.0, width=14.0, banking=50.0)
road2.spawn_point.x = 0.0; road2.spawn_point.y = 0.0; road2.spawn_point.yaw = 0.0
env2 = SimulationEnvironment(road_def=road2)
env2.reset()
for i in range(int(5/ (1/60))):
    env2.step([0.0, 0.0, 0.0])
st = env2.vehicle.state
print(f"(a2) parked 5s on 50deg bank (tan=1.19>mu): speed={st.speed_total:.2f} "
      f"y drift={st.pos.y:+.2f} m  [expect slides]")

# --- (b) car straddling wall moving outward: pushed back inside? ---
env3 = SimulationEnvironment()
env3.reset()
ti = env3.track_queries.query_vehicle_pose(Vec2(60.0, 0.0), math.pi/2)
hw = ti['road_width'] / 2
# oval at (60,0): tangent +y, walls at x = 60 +/- (half_w+curb)
x_wall = 60.0 + hw + 0.5
env3.vehicle.reset(pos=Vec3(x_wall - 0.4, 0.0, 0.1), yaw=math.pi/2,
                   initial_speed=5.0)
env3.vehicle.state.vel_world = Vec2(5.0, 0.0)   # driving INTO the right wall
env3.vehicle.state.vel_body = Vec2(5.0, 0.0)
print(f"(b) wall at x={x_wall:.2f}; car straddling it moving outward:")
for i in range(45):
    env3.step([0.0, 0.0, 0.0])
    st = env3.vehicle.state
    if st.is_colliding or i % 10 == 0:
        print(f"    t={i/60:.2f} x={st.pos.x:.2f} colliding={st.is_colliding} "
              f"v=({st.vel_world.x:+.2f},{st.vel_world.y:+.2f})")
st = env3.vehicle.state
print(f"    final x={st.pos.x:.2f} (wall {x_wall:.2f}) "
      f"{'PUSHED IN' if st.pos.x < x_wall else 'EJECTED OUT'}")

# --- (c) car slightly past wall centre -> pulled back inside ---
env4 = SimulationEnvironment()
env4.reset()
env4.vehicle.reset(pos=Vec3(x_wall + 0.6, 0.0, 0.1), yaw=math.pi/2,
                   initial_speed=2.0)
env4.vehicle.state.vel_world = Vec2(0.0, 0.0)
env4.vehicle.state.vel_body = Vec2(0.0, 0.0)
env4.step([0.0, 0.0, 0.0])
st = env4.vehicle.state
print(f"(c) car started {0.6}m past wall centre: resolved to x={st.pos.x:.2f} "
      f"({'inside' if st.pos.x < x_wall else 'outside'})")
