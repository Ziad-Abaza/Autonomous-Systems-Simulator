"""
Action Space Designer and Action Validation Engine.
Provides composable action channels, continuous and discrete spaces,
parameter scaling, dead zones, rate limiting, and defensive validation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np


class ActionType:
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class ActionChannelConfig:
    """
    Defines a single controllable channel (e.g. steering, throttle, brake).
    """
    name: str
    channel_type: str = ActionType.CONTINUOUS
    min_val: float = -1.0
    max_val: float = 1.0
    default_val: float = 0.0
    scaling: float = 1.0
    dead_zone: float = 0.0          # Absolute value below dead_zone is mapped to default_val
    rate_limit: float = 0.0         # Max change allowed per second (0.0 = unlimited)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionChannelConfig:
        return cls(
            name=str(data.get("name", "unnamed_channel")),
            channel_type=str(data.get("channel_type", ActionType.CONTINUOUS)),
            min_val=float(data.get("min_val", -1.0)),
            max_val=float(data.get("max_val", 1.0)),
            default_val=float(data.get("default_val", 0.0)),
            scaling=float(data.get("scaling", 1.0)),
            dead_zone=max(0.0, float(data.get("dead_zone", 0.0))),
            rate_limit=max(0.0, float(data.get("rate_limit", 0.0))),
            description=str(data.get("description", ""))
        )


@dataclass
class DiscreteActionOption:
    """
    Defines a discrete action mapping to target values for each channel.
    """
    name: str
    values: List[float]  # Target values for channels in order

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscreteActionOption:
        return cls(
            name=str(data.get("name", "action")),
            values=[float(v) for v in data.get("values", [])]
        )


@dataclass
class ActionSpaceDefinition:
    """
    User-configurable declarative action space definition.
    Can be continuous (multi-channel box) or discrete (indexed choices).
    """
    space_type: str = ActionType.CONTINUOUS
    channels: List[ActionChannelConfig] = field(default_factory=list)
    discrete_options: List[DiscreteActionOption] = field(default_factory=list)

    @classmethod
    def create_default_vehicle_action_space(cls, continuous: bool = True) -> ActionSpaceDefinition:
        """Standard automotive 3-channel action space: steering, throttle, brake."""
        channels = [
            ActionChannelConfig(
                name="steering",
                channel_type=ActionType.CONTINUOUS,
                min_val=-1.0,
                max_val=1.0,
                default_val=0.0,
                scaling=1.0,
                dead_zone=0.02,
                rate_limit=4.0,  # Max 4.0 units/sec
                description="Normalized steering command [-1: full left, +1: full right]"
            ),
            ActionChannelConfig(
                name="throttle",
                channel_type=ActionType.CONTINUOUS,
                min_val=0.0,
                max_val=1.0,
                default_val=0.0,
                scaling=1.0,
                dead_zone=0.01,
                rate_limit=5.0,
                description="Forward throttle command [0: idle, 1: full drive force]"
            ),
            ActionChannelConfig(
                name="brake",
                channel_type=ActionType.CONTINUOUS,
                min_val=0.0,
                max_val=1.0,
                default_val=0.0,
                scaling=1.0,
                dead_zone=0.01,
                rate_limit=6.0,
                description="Braking force command [0: released, 1: full brake]"
            ),
        ]

        discrete_opts = [
            DiscreteActionOption("Coast", [0.0, 0.0, 0.0]),
            DiscreteActionOption("Accelerate", [0.0, 0.7, 0.0]),
            DiscreteActionOption("Brake", [0.0, 0.0, 0.8]),
            DiscreteActionOption("Steer Left + Throttle", [-0.6, 0.4, 0.0]),
            DiscreteActionOption("Steer Right + Throttle", [0.6, 0.4, 0.0]),
        ]

        return cls(
            space_type=ActionType.CONTINUOUS if continuous else ActionType.DISCRETE,
            channels=channels,
            discrete_options=discrete_opts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "space_type": self.space_type,
            "channels": [c.to_dict() for c in self.channels],
            "discrete_options": [d.to_dict() for d in self.discrete_options]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionSpaceDefinition:
        space_type = str(data.get("space_type", ActionType.CONTINUOUS))
        raw_channels = data.get("channels", [])
        channels = [ActionChannelConfig.from_dict(c) for c in raw_channels]
        raw_discrete = data.get("discrete_options", [])
        discrete_opts = [DiscreteActionOption.from_dict(d) for d in raw_discrete]
        return cls(
            space_type=space_type,
            channels=channels,
            discrete_options=discrete_opts
        )

    def export_schema(self) -> Dict[str, Any]:
        """Returns machine-readable contract schema for external AI clients."""
        return {
            "space_type": self.space_type,
            "num_channels": len(self.channels),
            "channels": [
                {
                    "name": c.name,
                    "type": c.channel_type,
                    "min": c.min_val,
                    "max": c.max_val,
                    "default": c.default_val,
                    "scaling": c.scaling,
                    "dead_zone": c.dead_zone,
                    "rate_limit": c.rate_limit,
                    "description": c.description
                }
                for c in self.channels
            ],
            "num_discrete_actions": len(self.discrete_options) if self.space_type == ActionType.DISCRETE else 0,
            "discrete_actions": [
                {"name": d.name, "values": d.values}
                for d in self.discrete_options
            ] if self.space_type == ActionType.DISCRETE else []
        }

    def compile_decoder(self) -> CompiledActionDecoder:
        """Compiles authoring definition into an optimized runtime decoder."""
        return CompiledActionDecoder(self)


class CompiledActionDecoder:
    """
    Optimized runtime action decoder executing 60 Hz rate limiting,
    deadzone mapping, scaling, and safe clamping without UI or reflection overhead.
    """
    def __init__(self, definition: ActionSpaceDefinition):
        self.space_type = definition.space_type
        self.num_channels = len(definition.channels)

        # Vectorized arrays for fast execution
        self.channel_names = [c.name for c in definition.channels]
        self.mins = np.array([c.min_val for c in definition.channels], dtype=np.float32)
        self.maxs = np.array([c.max_val for c in definition.channels], dtype=np.float32)
        self.defaults = np.array([c.default_val for c in definition.channels], dtype=np.float32)
        self.scalings = np.array([c.scaling for c in definition.channels], dtype=np.float32)
        self.dead_zones = np.array([c.dead_zone for c in definition.channels], dtype=np.float32)
        self.rate_limits = np.array([c.rate_limit for c in definition.channels], dtype=np.float32)

        # Discrete table
        self.discrete_table: List[np.ndarray] = [
            np.array(opt.values, dtype=np.float32) for opt in definition.discrete_options
        ]
        if not self.discrete_table:
            # Fallback single zero action
            self.discrete_table = [self.defaults.copy()]

        # State tracking for rate limiting
        self.prev_action = self.defaults.copy()

    def reset(self) -> None:
        self.prev_action = self.defaults.copy()

    def validate_action(self, raw_action: Any) -> Tuple[bool, str]:
        """
        Defensive validation checking for None, empty, NaN, Inf, wrong types,
        out-of-bounds discrete actions, or missing continuous dimensions.
        """
        if raw_action is None:
            return False, "Action cannot be None"

        if self.space_type == ActionType.DISCRETE:
            try:
                if isinstance(raw_action, (np.ndarray, list, tuple)):
                    if len(raw_action) == 0:
                        return False, "Discrete action container is empty"
                    val = raw_action[0]
                else:
                    val = raw_action

                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return False, f"Discrete action cannot be NaN or Infinity: {val}"

                idx = int(val)
                if idx < 0 or idx >= len(self.discrete_table):
                    return False, f"Discrete action index {idx} out of range [0, {len(self.discrete_table) - 1}]"
                return True, ""
            except (ValueError, TypeError, OverflowError) as e:
                return False, f"Invalid discrete action value: {e}"
        else:
            try:
                act = np.asarray(raw_action, dtype=np.float32).flatten()
                if act.size == 0:
                    return False, "Continuous action array is empty"
                if np.any(np.isnan(act)):
                    return False, "Continuous action contains NaN"
                if np.any(np.isinf(act)):
                    return False, "Continuous action contains Infinity"
                if act.size < self.num_channels:
                    return False, f"Continuous action size {act.size} is less than required channels {self.num_channels}"
                return True, ""
            except Exception as e:
                return False, f"Continuous action parse error: {e}"

    def decode(self, raw_action: Any, dt: float = 1.0 / 60.0) -> np.ndarray:
        """
        Decodes raw action into safely clamped, rate-limited, deadzoned channel values.
        Guaranteed to never return NaN or Inf, even with malformed input.
        """
        if self.space_type == ActionType.DISCRETE:
            idx = 0
            try:
                if isinstance(raw_action, (np.ndarray, list, tuple)):
                    val = raw_action[0] if len(raw_action) > 0 else 0
                else:
                    val = raw_action
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        val = 0
                idx = int(val)
            except Exception:
                idx = 0

            idx = max(0, min(idx, len(self.discrete_table) - 1))
            target = self.discrete_table[idx].copy()
        else:
            try:
                act = np.asarray(raw_action, dtype=np.float32).flatten()
            except Exception:
                act = self.defaults.copy()

            if act.size == 0:
                target = self.defaults.copy()
            else:
                # Sanitize NaN/Inf
                act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
                # Pad or slice to num_channels
                if act.size < self.num_channels:
                    padded = self.defaults.copy()
                    padded[:act.size] = act
                    target = padded
                else:
                    target = act[:self.num_channels].copy()

        # Apply scaling
        scaled = target * self.scalings

        # Apply deadzone per channel
        for i in range(self.num_channels):
            dz = self.dead_zones[i]
            if dz > 0.0:
                dev = scaled[i] - self.defaults[i]
                if abs(dev) < dz:
                    scaled[i] = self.defaults[i]

        # Apply rate limiting for continuous actions
        if self.space_type == ActionType.CONTINUOUS and dt > 0.0:
            for i in range(self.num_channels):
                rl = self.rate_limits[i]
                if rl > 0.0:
                    max_delta = rl * dt
                    delta = scaled[i] - self.prev_action[i]
                    if delta > max_delta:
                        scaled[i] = self.prev_action[i] + max_delta
                    elif delta < -max_delta:
                        scaled[i] = self.prev_action[i] - max_delta

        # Final clamping to configured bounds
        clamped = np.clip(scaled, self.mins, self.maxs)
        self.prev_action = clamped.copy()
        return clamped
