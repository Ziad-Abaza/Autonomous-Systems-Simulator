import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentRL.envs.factory import EnvFactory
from agentRL.envs.track_registry import TrackRegistry
from agentRL.baselines.pd_driver import PDPolicy
from agentRL.eval.evaluate import evaluate_policy

reg = TrackRegistry.default()
factory = EnvFactory()
out = {}
for tid in ["serpentine", "oval", "gen_loop_0"]:
    try:
        res = evaluate_policy(factory, reg.load(tid), PDPolicy(),
                              seeds=(42, 43, 44), max_steps=1500)
        out[tid] = res["aggregate"]
        ep_summary = [(e["length"], e["failure_class"], round(e["mean_speed"], 2))
                      for e in res["per_episode"]]
        print(f"== {tid} ==")
        print(json.dumps(res["aggregate"], indent=1))
        print("episodes:", ep_summary)
    except Exception as e:
        print(f"== {tid} == FAILED: {type(e).__name__}: {e}")

with open("_tmp_view/rl_baseline_repaired.json", "w") as f:
    json.dump(out, f, indent=2)
