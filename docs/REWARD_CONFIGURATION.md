---
noteId: "02ab7340bed811f1a29f1fbaabbd87c8"
tags: []

---

# Reward Engine Configuration & Component Decomposition

This document details the configuration, mathematical formulations, runtime decomposition, and authoring validation of the **Reward Engine** (`sim_env/reward_engine.py`).

---

## 1. Design Overview

The reward system calculates continuous step feedback and discrete event rewards for reinforcement learning policies. 

Key design principles:
1. **Zero Silent Contract Breaks**: All weights and bounds are strictly validated against negative parameters or illegal values.
2. **Transparent Decomposition**: Total step reward is accompanied by a full component dictionary (`breakdown`), allowing training algorithms and diagnostic HUDs to monitor individual reward contributions.
3. **Smooth Dense Feedback**: Provides dense shaping terms (progress, centering, alignment) balanced by safety penalties (collision, off-road).

---

## 2. Mathematical Formulations

At each physics step ($dt = 1/60\text{ s}$), the reward engine calculates:

$$R_{\text{total}} = R_{\text{progress}} + R_{\text{centering}} + R_{\text{speed}} + R_{\text{heading}} + R_{\text{smoothness}} + R_{\text{checkpoint}} + R_{\text{lap}} + R_{\text{collision}} + R_{\text{off\_road}} + R_{\text{backward}}$$

### 1. Forward Progress Reward ($R_{\text{progress}}$)
Measures displacement $\Delta s$ along the track spline centerline:
$$R_{\text{progress}} = w_{\text{progress}} \cdot \Delta s$$
Discontinuous teleportation ($\Delta s > 5.0\text{ m}$) is rejected to prevent physics glitches from generating reward spikes.

### 2. Centering Reward ($R_{\text{centering}}$)
Linearly rewards the vehicle for remaining centered between track boundaries:
$$R_{\text{centering}} = w_{\text{centering}} \cdot \left(1.0 - \min\left(1.0, \frac{|d_{\text{lat}}|}{0.5 \cdot W_{\text{road}}}\right)\right)$$

### 3. Target Speed Reward ($R_{\text{speed}}$)
Encourages driving near the configurable target velocity:
$$R_{\text{speed}} = w_{\text{speed}} \cdot \min\left(1.0, \max\left(0.0, \frac{v_{\text{lon}}}{v_{\text{target}}}\right)\right)$$

### 4. Heading Alignment Reward ($R_{\text{heading}}$)
Encourages vehicle forward orientation to match the local track tangent:
$$R_{\text{heading}} = w_{\text{heading}} \cdot \cos(\theta_{\text{error}})$$

### 5. Action Smoothness Penalty ($R_{\text{smoothness}}$)
Penalizes high-frequency steering chatter:
$$R_{\text{smoothness}} = - w_{\text{smoothness}} \cdot (\delta_t - \delta_{t-1})^2$$

### 6. Event Penalties and Bonuses
- **Checkpoint Bonus**: $+ B_{\text{cp}}$ awarded once upon crossing each sequential gate.
- **Lap Completion Bonus**: $+ B_{\text{lap}}$ awarded upon completing full circuit traversal.
- **Collision Penalty**: $- P_{\text{collision}}$ applied on contact with barrier or obstacle.
- **Off-Road Penalty**: $- P_{\text{off\_road}}$ applied when wheels leave track surface.
- **Backward Driving Penalty**: $- P_{\text{backward}}$ applied when heading error $> 108^\circ$ or $\Delta s < -0.05\text{ m}$.

---

## 3. Configuration & Parameter Validation

`RewardConfig` enforces valid parameters to prevent degenerate training behavior:

```python
from sim_env.reward_engine import RewardConfig

cfg = RewardConfig(
    weight_progress=1.0,
    weight_centering=0.5,
    weight_speed=0.2,
    target_speed=20.0,
    weight_heading=0.3,
    weight_action_smoothness=0.05,
    checkpoint_bonus=10.0,
    lap_completion_bonus=100.0,
    collision_penalty=50.0,
    off_road_penalty=25.0,
    backward_penalty=1.0
)

# Throws ValueError if any weight or penalty is negative, or target_speed <= 0
cfg.validate()
```

---

## 4. Runtime Reward Breakdown Inspector

During interactive simulation or evaluation, the studio UI renders a live component breakdown:

```
REWARD DECOMPOSITION (Step 412)
----------------------------------------
Progress (+0.33m)      :  +0.332
Centering (0.12m off)  :  +0.480
Speed (19.4 / 20.0 m/s):  +0.194
Heading (1.2 deg err)  :  +0.299
Smoothness (Δδ 0.01)   :  -0.001
Checkpoint Passed      :  +0.000
Collision              :  +0.000
Off-Road               :  +0.000
Backward               :  +0.000
----------------------------------------
Total Step Reward      :  +1.304
Accumulated Episode    : +542.850
```
