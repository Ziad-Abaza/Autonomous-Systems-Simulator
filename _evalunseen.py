import sys, json
sys.path.insert(0, ".")
from agentRL.checkpoints.io import load_agent
from agentRL.envs.factory import EnvFactory
from agentRL.envs.track_registry import TrackRegistry
from agentRL.eval.matrix import EvalMatrix
from agentRL.obs.spec import ObservationSpec, PRESETS

factory = EnvFactory(reward="drive_v1", termination="term_v1", initial_speed=8.0,
                     obs_spec=ObservationSpec(channel_names=PRESETS["full23"]))
reg = TrackRegistry.default()
agent = load_agent("agentRL/experiments/runs/E001b/sac/checkpoints/step_50000.pt")
print("policy:", agent.algo_id)
tracks = [reg.load("oval"), reg.load("gen_loop_4"), reg.load("gen_loop_5")]
grid = EvalMatrix(factory).run({"e001b_50k": agent}, tracks,
                               out_dir="docs/master_audit/evidence/agent/eval_unseen",
                               seeds=(42,43), max_steps=1500)
for t, c in grid["e001b_50k"].items():
    print("%-12s ret=%8.2f speed=%.2f coll=%.2f len=%d fail=%s" % (
        t, c["mean_return"], c["mean_speed"], c["collision_rate"],
        c["mean_length"], c.get("failure_classes")))
