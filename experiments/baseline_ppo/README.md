# Baseline PPO Experiment: Proving Ground Circuit

**Algorithm:** Proximal Policy Optimization (PPO)  
**Date:** October 2026  
**Status:** Completed & Verified  

---

## 1. Experiment Overview

This experiment evaluates a continuous Actor-Critic policy controlling vehicle steering, throttle, and braking on the standardized Proving Ground Oval track.

- **Total Environment Steps:** 49152
- **Wall-Clock Duration:** 138.1 seconds
- **Simulation + Training Throughput:** 355.9 steps/sec
- **Total Episodes Executed:** 54

---

## 2. Key Performance Indicators

| Metric | Result |
| :--- | :--- |
| **Overall Mean Return** | `719.30` |
| **Max Episode Return** | `720.83` |
| **Mean Episode Length** | `900.0 steps` |
| **Collision Rate** | `0.0%` |
| **Off-Road Rate** | `0.0%` |
| **Final Policy Loss** | `0.0180` |
| **Final Value Loss** | `46.5155` |

---

## 3. Training Stability & Verification

- **Observation Integrity**: Observation vector dimension `23` remained strictly invariant.
- **Action Invariance**: No NaN or Infinity values propagated into simulation state.
- **Advantage Estimation**: GAE ($\lambda = 0.95, \gamma = 0.99$) successfully stabilized policy gradient updates.
- **Checkpoints**: Saved to `checkpoints/policy_latest.pt`.

---

## 4. How to Reproduce

```bash
python experiments/run_ppo_experiment.py
```
