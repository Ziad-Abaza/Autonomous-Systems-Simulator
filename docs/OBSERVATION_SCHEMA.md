---
noteId: "2ece0e40beda11f1a29f1fbaabbd87c8"
tags: []

---

# Observation Schema & Designer Specification

**Google DeepMind Advanced Agentic Coding — Phase 3 Specification**  
**Component:** Perception & Observation Pipeline  
**Source Implementation:** [`sim_env/observation_designer.py`](file:///D:/coding/projects/Simulation/sim_env/observation_designer.py)  

---

## 1. Critical Observation Security Rule

> [!CAUTION]
> **CRITICAL OBSERVATION RULE:**  
> The system strictly distinguishes **Agent Observation** from **Debug Telemetry** and **Oracle Ground Truth**.  
> Debug telemetry and oracle data must NEVER enter the agent observation space under any circumstance.

In reinforcement learning, "observation leakage" occurs when privileged simulator internal states (such as exact global coordinates of obstacles, unoccluded enemy positions, perfect collision vectors, or evaluation metrics) inadvertently pollute the agent's observation vector. This produces agents that learn degenerate "cheating" policies that collapse when deployed in real environments.

### Channel Categories
Every potential telemetry channel is classified into a strict enumeration:
```python
class ChannelCategory:
    AGENT_OBSERVATION = "agent_observation"      # Observable by agent sensors in realistic conditions
    DEBUG_TELEMETRY = "debug_telemetry"          # Simulator internal state for UI/HUD displays
    ORACLE_GROUND_TRUTH = "oracle_ground_truth"  # Privileged global ground truth used only by evaluator/reward
```

### Automated Leakage Gatekeeper
The observation pipeline enforces zero leakage at compile time:
```python
def validate_no_leakage(self) -> Tuple[bool, List[str]]:
    violations = []
    for c in self.get_active_channels():
        if c.category in (ChannelCategory.DEBUG_TELEMETRY, ChannelCategory.ORACLE_GROUND_TRUTH):
            violations.append(
                f"Observation channel '{c.name}' has privileged category '{c.category}'. "
                f"Debug telemetry and oracle data must never enter the agent observation space."
            )
    return len(violations) == 0, violations
```
If an agent observation space definition contains a channel categorized as `DEBUG_TELEMETRY` or `ORACLE_GROUND_TRUTH`, `validate_no_leakage()` fails, and attempting to compile the runtime pipeline raises a `ValueError`.

---

## 2. Machine-Readable Observation Schema

Each observation channel is defined by `ObservationChannelConfig`:

```python
@dataclass
class ObservationChannelConfig:
    name: str                           # Unique string identifier
    channel_type: str                   # "scalar", "vector", "image"
    shape: List[int]                    # Dimensionality, e.g. [1], [15], [84, 84, 3]
    dtype: str = "float32"              # "float32", "uint8"
    range_low: List[float]              # Minimum possible/allowed values
    range_high: List[float]             # Maximum possible/allowed values
    normalization: str = "none"         # "none", "scale", "min_max", "clip"
    norm_params: Dict[str, Any]         # Parameters for normalizer (e.g. {"scale": 45.0})
    source_sensor: str = ""             # Sensor providing the raw signal ("vehicle_state", "lidar_rays")
    source_key: str = ""                # Dictionary key in sensor sample dict
    category: str = "agent_observation" # ChannelCategory
    enabled: bool = True                # Toggled via UI
    description: str = ""               # Human-readable documentation
```

### JSON Representation Example
```json
{
  "name": "speed",
  "channel_type": "scalar",
  "shape": [1],
  "dtype": "float32",
  "range_low": [0.0],
  "range_high": [50.0],
  "normalization": "scale",
  "norm_params": { "scale": 45.0 },
  "source_sensor": "vehicle_state",
  "source_key": "speed",
  "category": "agent_observation",
  "enabled": true,
  "description": "Forward vehicle speed normalized by 45 m/s"
}
```

---

## 3. Standard Observation Channels

The default vehicle agent includes 8 standard channels:

| Channel Name | Type | Shape | Raw Range | Normalization | Normalized Range | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `speed` | Scalar | `[1]` | $[0, 50]$ m/s | Scale ($45.0$) | $[0.0, 1.11]$ | `vehicle_state.speed` |
| `velocity_body` | Vector | `[2]` | $[-10, 50] \times [-10, 10]$ | Scale ($v_x: 45, v_y: 10$) | $[-1.0, 1.0]$ | `vehicle_state.vel_body` |
| `yaw_rate` | Scalar | `[1]` | $[-3, 3]$ rad/s | Scale ($3.0$) | $[-1.0, 1.0]$ | `vehicle_state.yaw_rate` |
| `steering_angle` | Scalar | `[1]` | $[-0.6, 0.6]$ rad | Scale ($0.6$) | $[-1.0, 1.0]$ | `vehicle_state.steering_angle` |
| `lateral_error` | Scalar | `[1]` | $[-10, 10]$ m | Scale (half road width) | $[-1.0, 1.0]$ | `vehicle_state.distance_from_center` |
| `heading_error` | Scalar | `[1]` | $[-\pi, \pi]$ rad | Scale ($\pi$) | $[-1.0, 1.0]$ | `vehicle_state.heading_error` |
| `dist_checkpoint`| Scalar | `[1]` | $[0, 100]$ m | Scale ($100.0$) | $[0.0, 1.0]$ | `vehicle_state.distance_to_checkpoint` |
| `lidar_ranges` | Vector | `[15]` | $[0, 50]$ m | Built-in norm | $[0.0, 1.0]$ | `lidar_rays.ranges_norm` |

**Total Vector Dimension:** $1 + 2 + 1 + 1 + 1 + 1 + 1 + 15 = 23$ features.

---

## 4. Normalization Modes

The pipeline provides 4 deterministic normalization models:
1. **`none`**: Passes raw sensor signal unchanged.
2. **`scale`**: Multiplies by $\frac{1}{\text{scale}}$, with dynamic adaptive scaling support (e.g. `scale_half_width` using live local road width).
3. **`min_max`**: Maps $[min, max] \rightarrow [0, 1]$ via $\frac{x - min}{max - min}$.
4. **`clip`**: Clamps value strictly into $[min, max]$.

All transformations apply `np.nan_to_num(val, nan=0.0, posinf=1.0, neginf=-1.0)` to guarantee numeric safety.

---

## 5. Zero-Overhead Compiled Runtime (`CompiledObservationPipeline`)

The authoring model compiles into a high-performance execution engine:
- Pre-indexes sensor names and dictionary keys during initialization.
- Pre-allocates a contiguous 1D `np.float32` array of size `total_dim`.
- Directly writes normalized scalar and vector slices into memory buffers without allocating intermediate Python objects during `step()`.
- Eliminates per-step string formatting, schema parsing, and dynamic reflection.
