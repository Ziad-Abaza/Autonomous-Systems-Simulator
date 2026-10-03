import json, os, sys
sys.path.insert(0, ".")
import numpy as np
from agentRL.core.config import TrainConfig, AgentConfig
from agentRL.train.trainer import OffPolicyTrainer
from agentRL.checkpoints.io import read_run_state, load_agent
from agentRL.algos.sac import SACAgent
from agentRL.obs.spec import ObservationSpec, PRESETS
from agentRL.envs.factory import EnvFactory
from agentRL.envs.track_registry import TrackRegistry

base = "docs/master_audit/evidence/agent/resume_fidelity"
os.makedirs(base, exist_ok=True)
factory = EnvFactory(obs_spec=ObservationSpec(channel_names=PRESETS["state8"]))
spec = TrackRegistry.default().load("smoke")

def _agent(seed=7):
    s = ObservationSpec(channel_names=PRESETS["state8"])
    return SACAgent(obs_spec=s,
                    act_space={"low": np.array([-1.0,0.0,0.0]), "high": np.array([1.0,1.0,1.0])},
                    cfg=AgentConfig(algo_id="sac", hidden_sizes=(64,64), lr=3e-4, gamma=0.99,
                                    extra={"seed": seed, "warmup": 32, "batch": 16, "random_steps": 16}))

cfg1 = TrainConfig(total_steps=300, eval_interval=10**9, ckpt_interval=300,
                   num_envs=1, seed=42, eval_episodes=1, run_dir=f"{base}/leg1")
OffPolicyTrainer(factory, spec, _agent(), cfg1).train()
st1 = read_run_state(f"{base}/leg1"); print("leg1 timestep:", st1["timestep"])

cfg2 = TrainConfig(total_steps=600, eval_interval=10**9, ckpt_interval=600,
                   num_envs=1, seed=42, eval_episodes=1, run_dir=f"{base}/leg2",
                   resume_from=f"{base}/leg1/checkpoints/latest.pt")
OffPolicyTrainer(factory, spec, _agent(), cfg2).train()
st2 = read_run_state(f"{base}/leg2"); print("leg2 timestep:", st2["timestep"])
ag = load_agent(f"{base}/leg2/checkpoints/latest.pt"); print("loaded:", ag.algo_id)
rows = [json.loads(l) for l in open(f"{base}/leg2/metrics.jsonl")]
print("leg2 metrics rows:", len(rows), "| seq monotonic:", [r["seq"] for r in rows]==sorted(r["seq"] for r in rows))
