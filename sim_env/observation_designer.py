"""
Observation Space Designer and Validation System.
Provides composable, user-authorable observation channels,
deterministic schema export, strict isolation between Agent Perception,
Debug Telemetry, and Oracle Ground Truth, and a compiled runtime pipeline.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np


class ChannelCategory:
    AGENT_OBSERVATION = "agent_observation"      # Permitted in agent observation space
    DEBUG_TELEMETRY = "debug_telemetry"          # UI HUD / logging only, forbidden from observation
    ORACLE_GROUND_TRUTH = "oracle_ground_truth"  # Privileged state, strictly forbidden from observation


class NormalizationType:
    NONE = "none"
    SCALE = "scale"          # val / scale
    MIN_MAX = "min_max"      # (val - min) / (max - min)
    STANDARDIZED = "standardized"  # (val - mean) / std
    CLIP = "clip"            # np.clip(val, min, max)


@dataclass
class ObservationChannelConfig:
    """
    Specification for a single observation feature or sensor channel.
    """
    name: str
    channel_type: str = "scalar"      # "scalar", "vector", "image"
    shape: List[int] = field(default_factory=lambda: [1])
    dtype: str = "float32"            # "float32", "uint8"
    range_low: List[float] = field(default_factory=lambda: [-1.0])
    range_high: List[float] = field(default_factory=lambda: [1.0])
    normalization: str = NormalizationType.NONE
    norm_params: Dict[str, float] = field(default_factory=dict)
    source_sensor: str = "vehicle_state"
    source_key: str = "speed"
    category: str = ChannelCategory.AGENT_OBSERVATION
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationChannelConfig:
        return cls(
            name=str(data.get("name", "unnamed_obs")),
            channel_type=str(data.get("channel_type", "scalar")),
            shape=list(data.get("shape", [1])),
            dtype=str(data.get("dtype", "float32")),
            range_low=[float(v) for v in data.get("range_low", [-1.0])],
            range_high=[float(v) for v in data.get("range_high", [1.0])],
            normalization=str(data.get("normalization", NormalizationType.NONE)),
            norm_params={str(k): float(v) for k, v in data.get("norm_params", {}).items()},
            source_sensor=str(data.get("source_sensor", "vehicle_state")),
            source_key=str(data.get("source_key", "speed")),
            category=str(data.get("category", ChannelCategory.AGENT_OBSERVATION)),
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description", ""))
        )


@dataclass
class ObservationSpaceDefinition:
    """
    User-configurable observation space configuration.
    Maintains ordered list of channels, vector vs dict formatting,
    and schema validation.
    """
    channels: List[ObservationChannelConfig] = field(default_factory=list)
    flatten_vector: bool = True
    include_image_channel: bool = False
    # Named camera image channels: [{"name": sensor_name, "shape": [H,W,C]}].
    # `include_image_channel` remains as the legacy single-camera flag;
    # image_channels is the authoritative multi-camera contract.
    image_channels: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_default_space(cls) -> ObservationSpaceDefinition:
        """
        Creates the standard 23-dimensional normalized vehicle observation space:
        - Speed (1)
        - Body velocity [vx, vy] (2)
        - Yaw rate (1)
        - Steering angle (1)
        - Lateral error (1)
        - Heading error (1)
        - Distance to next checkpoint (1)
        - 15-beam planar LiDAR ranges (15)
        """
        channels = [
            ObservationChannelConfig(
                name="speed",
                channel_type="scalar",
                shape=[1],
                range_low=[0.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale": 45.0},
                source_sensor="vehicle_state",
                source_key="speed",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Longitudinal forward speed normalized by 45 m/s"
            ),
            ObservationChannelConfig(
                name="velocity_body",
                channel_type="vector",
                shape=[2],
                range_low=[-1.0, -1.0],
                range_high=[1.0, 1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale_x": 45.0, "scale_y": 10.0},
                source_sensor="vehicle_state",
                source_key="vel_body",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Body-frame velocities (vx / 45, vy / 10)"
            ),
            ObservationChannelConfig(
                name="yaw_rate",
                channel_type="scalar",
                shape=[1],
                range_low=[-1.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale": 3.0},
                source_sensor="vehicle_state",
                source_key="yaw_rate",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Angular yaw velocity normalized by 3.0 rad/s"
            ),
            ObservationChannelConfig(
                name="steering_angle",
                channel_type="scalar",
                shape=[1],
                range_low=[-1.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale": 0.6},
                source_sensor="vehicle_state",
                source_key="steering_angle",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Current front road wheel angle normalized by max steer (0.6 rad)"
            ),
            ObservationChannelConfig(
                name="distance_from_center",
                channel_type="scalar",
                shape=[1],
                range_low=[-1.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale_half_width": 1.0},
                source_sensor="vehicle_state",
                source_key="distance_from_center",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Lateral deviation from track centerline normalized to half road width"
            ),
            ObservationChannelConfig(
                name="heading_error",
                channel_type="scalar",
                shape=[1],
                range_low=[-1.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale": math.pi},
                source_sensor="vehicle_state",
                source_key="heading_error",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Angle between vehicle heading and track centerline tangent normalized by pi"
            ),
            ObservationChannelConfig(
                name="distance_to_checkpoint",
                channel_type="scalar",
                shape=[1],
                range_low=[0.0],
                range_high=[1.0],
                normalization=NormalizationType.SCALE,
                norm_params={"scale": 100.0},
                source_sensor="vehicle_state",
                source_key="distance_to_checkpoint",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="Euclidean distance to next checkpoint gate capped at 100 m"
            ),
            ObservationChannelConfig(
                name="lidar_ranges",
                channel_type="vector",
                shape=[15],
                range_low=[0.0] * 15,
                range_high=[1.0] * 15,
                normalization=NormalizationType.NONE,
                source_sensor="lidar_rays",
                source_key="ranges_norm",
                category=ChannelCategory.AGENT_OBSERVATION,
                description="15 normalized planar LiDAR range measurements [0: impact, 1: clear]"
            ),
        ]
        return cls(channels=channels, flatten_vector=True, include_image_channel=False)

    def image_channel_specs(self) -> List[Dict[str, Any]]:
        """Effective image channel list — legacy flag degrades to the
        single default camera for backward compatibility."""
        if self.image_channels:
            return self.image_channels
        if self.include_image_channel:
            return [{"name": "rgb_camera", "shape": [84, 84, 3]}]
        return []

    def add_channel(self, channel: ObservationChannelConfig) -> None:
        self.channels.append(channel)

    def remove_channel(self, name: str) -> bool:
        initial_len = len(self.channels)
        self.channels = [c for c in self.channels if c.name != name]
        return len(self.channels) < initial_len

    def enable_channel(self, name: str, enabled: bool = True) -> bool:
        for c in self.channels:
            if c.name == name:
                c.enabled = enabled
                return True
        return False

    def reorder_channels(self, ordered_names: List[str]) -> None:
        """Reorders channels to match specified name sequence."""
        new_channels = []
        name_map = {c.name: c for c in self.channels}
        for name in ordered_names:
            if name in name_map:
                new_channels.append(name_map.pop(name))
        # Append remaining channels
        new_channels.extend(name_map.values())
        self.channels = new_channels

    def get_active_channels(self) -> List[ObservationChannelConfig]:
        return [c for c in self.channels if c.enabled]

    def compute_vector_dim(self) -> int:
        dim = 0
        for c in self.get_active_channels():
            if c.channel_type in ("scalar", "vector"):
                dim += int(np.prod(c.shape))
        return dim

    def validate_no_leakage(self) -> Tuple[bool, List[str]]:
        """
        CRITICAL OBSERVATION RULE:
        Verifies that no DEBUG_TELEMETRY or ORACLE_GROUND_TRUTH channels
        are enabled inside the Agent Observation Space.
        """
        violations = []
        for c in self.get_active_channels():
            if c.category in (ChannelCategory.DEBUG_TELEMETRY, ChannelCategory.ORACLE_GROUND_TRUTH):
                violations.append(
                    f"Observation channel '{c.name}' has category '{c.category}'. "
                    f"Debug telemetry and oracle data must never enter the agent observation space."
                )
        return len(violations) == 0, violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "flatten_vector": self.flatten_vector,
            # legacy flag kept in sync for older consumers
            "include_image_channel": bool(self.image_channel_specs()),
            "image_channels": [dict(spec) for spec in self.image_channel_specs()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationSpaceDefinition:
        raw_channels = data.get("channels", [])
        channels = [ObservationChannelConfig.from_dict(c) for c in raw_channels]
        return cls(
            channels=channels,
            flatten_vector=bool(data.get("flatten_vector", True)),
            include_image_channel=bool(data.get("include_image_channel", False)),
            image_channels=[dict(s) for s in data.get("image_channels", [])]
        )

    def export_schema(self) -> Dict[str, Any]:
        """Exports machine-readable specification of observation space."""
        active = self.get_active_channels()
        vector_dim = self.compute_vector_dim()

        return {
            "flatten_vector": self.flatten_vector,
            "vector_dimension": vector_dim,
            "include_image": bool(self.image_channel_specs()),
            "image_channels": [dict(s) for s in self.image_channel_specs()],
            "num_channels": len(active),
            "channels": [
                {
                    "name": c.name,
                    "type": c.channel_type,
                    "shape": c.shape,
                    "dtype": c.dtype,
                    "range": {"low": c.range_low, "high": c.range_high},
                    "normalization": c.normalization,
                    "norm_params": c.norm_params,
                    "source": f"{c.source_sensor}:{c.source_key}",
                    "category": c.category,
                    "description": c.description,
                }
                for c in active
            ]
        }

    def compile_pipeline(self) -> CompiledObservationPipeline:
        """Compiles definition into zero-overhead runtime pipeline."""
        return CompiledObservationPipeline(self)


class CompiledObservationPipeline:
    """
    Zero-overhead runtime observation evaluator.
    Directly extracts and normalizes sensor values into preallocated buffers
    avoiding per-tick reflection or dictionary allocations.
    """
    def __init__(self, definition: ObservationSpaceDefinition):
        self.definition = definition
        self.flatten_vector = definition.flatten_vector
        # Effective named camera channels (legacy flag degrades to one)
        self.image_specs = definition.image_channel_specs()
        self.include_image = bool(self.image_specs)

        # Security assertion
        is_safe, violations = definition.validate_no_leakage()
        if not is_safe:
            raise ValueError(f"Observation security validation failed: {violations}")

        self.active_channels = definition.get_active_channels()
        self.total_dim = definition.compute_vector_dim()
        self.buffer = np.zeros(self.total_dim, dtype=np.float32)

    def build_observation(self, sensor_samples: Dict[str, Any]) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Populates observation vector from sensor samples.
        """
        if self.flatten_vector:
            idx = 0
            for c in self.active_channels:
                s_data = sensor_samples.get(c.source_sensor, {})
                raw_val = s_data.get(c.source_key, None) if isinstance(s_data, dict) else None

                if c.channel_type == "scalar":
                    if raw_val is None:
                        val = 0.0
                    elif isinstance(raw_val, (int, float, np.floating, np.integer)):
                        val = float(raw_val)
                    else:
                        val = 0.0

                    # Normalization
                    if c.normalization == NormalizationType.SCALE:
                        scale = c.norm_params.get("scale", 1.0)
                        if "scale_half_width" in c.norm_params:
                            scale = max(1.0, float(s_data.get("road_width", 12.0)) * 0.5)
                        val = val / max(1e-6, scale)
                    elif c.normalization == NormalizationType.MIN_MAX:
                        low = c.norm_params.get("min", 0.0)
                        high = c.norm_params.get("max", 1.0)
                        val = (val - low) / max(1e-6, high - low)
                    elif c.normalization == NormalizationType.CLIP:
                        low = c.norm_params.get("min", -1.0)
                        high = c.norm_params.get("max", 1.0)
                        val = min(high, max(low, val))

                    self.buffer[idx] = float(np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0))
                    idx += 1

                elif c.channel_type == "vector":
                    expected_len = int(np.prod(c.shape))
                    if raw_val is None:
                        vec = np.ones(expected_len, dtype=np.float32) if "lidar" in c.name else np.zeros(expected_len, dtype=np.float32)
                    elif isinstance(raw_val, np.ndarray):
                        vec = raw_val.astype(np.float32)
                    elif isinstance(raw_val, (list, tuple)):
                        vec = np.array(raw_val, dtype=np.float32)
                    else:
                        vec = np.zeros(expected_len, dtype=np.float32)

                    # Custom velocity normalization if needed
                    if c.name == "velocity_body" and "scale_x" in c.norm_params and vec.size >= 2:
                        vec = np.array([
                            vec[0] / max(1e-6, c.norm_params["scale_x"]),
                            vec[1] / max(1e-6, c.norm_params["scale_y"])
                        ], dtype=np.float32)

                    # Ensure exact size
                    if vec.size < expected_len:
                        padded = np.zeros(expected_len, dtype=np.float32)
                        padded[:vec.size] = vec
                        vec = padded
                    elif vec.size > expected_len:
                        vec = vec[:expected_len]

                    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
                    self.buffer[idx:idx + expected_len] = vec
                    idx += expected_len

            if self.include_image:
                images = self._extract_images(sensor_samples)
                if self.total_dim == 0:
                    return next(iter(images.values())) \
                        if len(images) == 1 else images
                out = {"vector": self.buffer.copy()}
                out.update(images)
                return out

            return self.buffer.copy()

        else:
            # Dictionary observation
            obs_dict: Dict[str, Any] = {}
            for c in self.active_channels:
                s_data = sensor_samples.get(c.source_sensor, {})
                raw_val = s_data.get(c.source_key, None) if isinstance(s_data, dict) else None

                if c.channel_type == "scalar":
                    val = float(raw_val) if raw_val is not None else 0.0
                    obs_dict[c.name] = float(np.nan_to_num(val, nan=0.0))
                elif c.channel_type == "vector":
                    expected_len = int(np.prod(c.shape))
                    if isinstance(raw_val, np.ndarray):
                        vec = raw_val.astype(np.float32)
                    elif isinstance(raw_val, (list, tuple)):
                        vec = np.array(raw_val, dtype=np.float32)
                    else:
                        vec = np.zeros(expected_len, dtype=np.float32)
                    obs_dict[c.name] = np.nan_to_num(vec, nan=0.0)

            if self.include_image:
                obs_dict.update(self._extract_images(sensor_samples))

            return obs_dict

    @staticmethod
    def _image_key(sensor_name: str) -> str:
        # "rgb_camera" keeps the legacy "image" contract key; every other
        # camera gets its own deterministic image_<name> key.
        return "image" if sensor_name == "rgb_camera" else f"image_{sensor_name}"

    def _extract_images(self, sensor_samples: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in self.image_specs:
            name = spec["name"]
            shape = spec.get("shape") or [84, 84, 3]
            sample = sensor_samples.get(name, None)
            img = sample if (isinstance(sample, np.ndarray)
                             and sample.ndim == 3) \
                else np.zeros(tuple(shape), dtype=np.uint8)
            out[self._image_key(name)] = img
        return out
