"""
Environment Template System.
Generates fully configured declarative environments from templates:
- Empty Environment: Minimal open sandbox for scratch track building.
- Basic Driving: Simple oval proving ground for kinematic and throttle testing.
- Lane Following: Serpentine track with curvature, centering rewards, and LiDAR.
- Obstacle Avoidance: Circuit populated with cones, barriers, and evasion penalties.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from sim_core.track.road_definition import RoadDefinition, ControlPoint, SpawnPoint
from sim_core.vehicle.vehicle_config import VehicleConfig
from sim_core.world.entity import WorldEntity, create_entity
from sim_core.math_utils import Vec3
from sim_env.agent import AgentDefinition, AgentSpawnConfig
from sim_env.observation_designer import ObservationSpaceDefinition, ObservationChannelConfig, NormalizationType
from sim_env.action_designer import ActionSpaceDefinition
from sim_env.reward_designer import RewardFunctionDefinition, RewardComponentConfig
from sim_env.termination_designer import TerminationDefinition, TerminationRuleConfig
from sim_env.scenario_designer import ScenarioDefinition
from sim_env.episode_config import EpisodeConfiguration
from sim_project.serializer import EnvironmentProject


class EnvironmentTemplateManager:
    """
    Factory creating complete, editable EnvironmentProject instances from templates.
    """
    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        return [
            {"id": "empty", "name": "Empty Environment", "description": "Minimal sandbox with a short open route — start drawing your own track."},
            {"id": "basic_driving", "name": "Basic Driving", "description": "High-grip proving ground oval for basic throttle and speed tuning."},
            {"id": "straight_sprint", "name": "Straight Sprint", "description": "Open 300 m straight route — throttle, brake and top-speed tuning."},
            {"id": "hairpin", "name": "Hairpin Circuit", "description": "Closed circuit with a tight 180° hairpin for low-speed cornering."},
            {"id": "slalom", "name": "Cone Slalom", "description": "Open route threading offset cones — lateral control and precision."},
            {"id": "lane_following", "name": "Lane Following", "description": "Technical serpentine circuit with centering and heading rewards."},
            {"id": "obstacle_avoidance", "name": "Obstacle Avoidance", "description": "Circuit with cones and barriers demanding LiDAR collision avoidance."},
        ]

    @classmethod
    def create_project_from_template(cls, template_id: str) -> EnvironmentProject:
        template_id = template_id.lower().strip()

        if template_id == "empty":
            return cls._create_empty()
        elif template_id == "basic_driving":
            return cls._create_basic_driving()
        elif template_id == "lane_following":
            return cls._create_lane_following()
        elif template_id == "straight_sprint":
            return cls._create_straight_sprint()
        elif template_id == "hairpin":
            return cls._create_hairpin()
        elif template_id == "slalom":
            return cls._create_slalom()
        elif template_id == "obstacle_avoidance":
            return cls._create_obstacle_avoidance()
        else:
            return cls._create_basic_driving()

    @classmethod
    def _create_empty(cls) -> EnvironmentProject:
        road = RoadDefinition(name="Empty Sandbox Track", is_closed=False)
        road.add_control_point(0.0, 0.0, 0.0, 14.0)
        road.add_control_point(50.0, 0.0, 0.0, 14.0)
        road.add_control_point(100.0, 0.0, 0.0, 14.0)
        road.spawn_point = SpawnPoint(x=0.0, y=0.0, z=0.2, yaw=0.0, initial_speed=0.0)
        road.num_checkpoints = 4

        agent = AgentDefinition.create_default_vehicle_agent(agent_id="sandbox_agent")
        return EnvironmentProject(
            name="Empty Sandbox Environment",
            road_def=road,
            agent=agent
        )

    @classmethod
    def _create_basic_driving(cls) -> EnvironmentProject:
        road = RoadDefinition.create_default_oval(radius_x=65.0, radius_y=40.0, width=12.0)
        road.name = "Proving Ground Oval"
        agent = AgentDefinition.create_default_vehicle_agent(agent_id="oval_agent")
        return EnvironmentProject(
            name="Basic Driving Proving Ground",
            road_def=road,
            agent=agent
        )

    @classmethod
    def _create_lane_following(cls) -> EnvironmentProject:
        road = RoadDefinition(name="Serpentine Technical Circuit", is_closed=True)
        pts = [
            (0.0, -50.0, 0.0, 14.0),
            (50.0, -45.0, 2.0, 12.0),
            (80.0, -10.0, 4.0, 10.0),
            (65.0, 30.0, 3.0, 11.0),
            (25.0, 15.0, 1.5, 12.0),
            (0.0, 50.0, 0.0, 13.0),
            (-35.0, 60.0, -1.0, 14.0),
            (-70.0, 35.0, -2.0, 12.0),
            (-75.0, -15.0, -1.0, 11.0),
            (-40.0, -40.0, 0.0, 13.0),
        ]
        for x, y, z, w in pts:
            road.add_control_point(x=x, y=y, z=z, width=w, banking=0.0)
        road.spawn_point = SpawnPoint(x=0.0, y=-50.0, z=0.2, yaw=0.0, initial_speed=0.0)
        road.num_checkpoints = 20

        agent = AgentDefinition.create_default_vehicle_agent(agent_id="lane_follower_agent")
        # Boost centering and heading weights for lane following
        reward_fn = agent.reward_function
        c_center = reward_fn.get_component("centering")
        if c_center:
            c_center.weight = 1.0
        c_head = reward_fn.get_component("heading")
        if c_head:
            c_head.weight = 0.8

        return EnvironmentProject(
            name="Lane Following Serpentine Circuit",
            road_def=road,
            agent=agent
        )

    @classmethod
    def _create_straight_sprint(cls) -> EnvironmentProject:
        road = RoadDefinition(name="Sprint Straight", is_closed=False)
        pts = [
            (0.0, 0.0, 0.0, 14.0),
            (75.0, 0.0, 0.0, 14.0),
            (150.0, 2.0, 0.0, 14.0),
            (225.0, -2.0, 0.0, 14.0),
            (300.0, 0.0, 0.0, 14.0),
        ]
        for x, y, z, w in pts:
            road.add_control_point(x=x, y=y, z=z, width=w, banking=0.0)
        road.spawn_point = SpawnPoint(x=0.0, y=0.0, z=0.2, yaw=0.0,
                                      initial_speed=0.0)
        road.num_checkpoints = 6

        agent = AgentDefinition.create_default_vehicle_agent(
            agent_id="sprint_agent")
        return EnvironmentProject(
            name="Straight Sprint Route",
            road_def=road,
            agent=agent
        )

    @classmethod
    def _create_hairpin(cls) -> EnvironmentProject:
        road = RoadDefinition(name="Hairpin Circuit", is_closed=True)
        # Long straight down, tight 180° hairpin, straight back
        pts = [
            (0.0, 0.0, 0.0, 12.0),
            (60.0, 0.0, 0.0, 12.0),
            (110.0, 0.0, 0.0, 12.0),
            (135.0, 8.0, 0.0, 11.0),    # hairpin entry
            (138.0, 30.0, 0.5, 10.0),   # hairpin apex
            (120.0, 42.0, 0.5, 11.0),   # hairpin exit
            (100.0, 40.0, 0.0, 12.0),
            (50.0, 40.0, 0.0, 12.0),
            (0.0, 40.0, 0.0, 12.0),
            (-20.0, 20.0, 0.0, 12.0),
        ]
        for x, y, z, w in pts:
            road.add_control_point(x=x, y=y, z=z, width=w, banking=0.0)
        road.spawn_point = SpawnPoint(x=0.0, y=0.0, z=0.2, yaw=0.0,
                                      initial_speed=0.0)
        road.num_checkpoints = 16

        agent = AgentDefinition.create_default_vehicle_agent(
            agent_id="hairpin_agent")
        return EnvironmentProject(
            name="Hairpin Circuit",
            road_def=road,
            agent=agent
        )

    @classmethod
    def _create_slalom(cls) -> EnvironmentProject:
        road = RoadDefinition(name="Cone Slalom", is_closed=False)
        pts = [
            (0.0, 0.0, 0.0, 16.0),
            (40.0, 0.0, 0.0, 16.0),
            (80.0, 0.0, 0.0, 16.0),
            (120.0, 0.0, 0.0, 16.0),
            (160.0, 0.0, 0.0, 16.0),
            (200.0, 0.0, 0.0, 16.0),
        ]
        for x, y, z, w in pts:
            road.add_control_point(x=x, y=y, z=z, width=w, banking=0.0)
        road.spawn_point = SpawnPoint(x=0.0, y=0.0, z=0.2, yaw=0.0,
                                      initial_speed=0.0)
        road.num_checkpoints = 8

        agent = AgentDefinition.create_default_vehicle_agent(
            agent_id="slalom_agent")
        proj = EnvironmentProject(
            name="Cone Slalom Course",
            road_def=road,
            agent=agent
        )
        # Offset cones alternating around the centerline
        offsets = [-4.0, 4.0, -4.0, 4.0, -4.0]
        proj.entities = [
            create_entity("cone", pos=Vec3(40.0 + i * 32.0, offsets[i], 0.0),
                          yaw=0.0)
            for i in range(len(offsets))
        ]
        return proj

    @classmethod
    def _create_obstacle_avoidance(cls) -> EnvironmentProject:
        proj = cls._create_basic_driving()
        proj.name = "Obstacle Avoidance Challenge"
        proj.agent.agent_id = "obstacle_evasion_agent"

        # Add obstacles to scene
        entities: List[WorldEntity] = [
            create_entity("cone", pos=Vec3(45.0, 15.0, 0.0), yaw=0.0),
            create_entity("barrier", pos=Vec3(-40.0, 10.0, 0.0), yaw=0.3),
            create_entity("cone", pos=Vec3(0.0, 40.0, 0.0), yaw=0.0),
            create_entity("cone", pos=Vec3(5.0, 41.0, 0.0), yaw=0.0),
        ]
        proj.entities = entities

        # Heavily weight collision penalty
        c_col = proj.agent.reward_function.get_component("collision")
        if c_col:
            c_col.weight = -100.0

        return proj
