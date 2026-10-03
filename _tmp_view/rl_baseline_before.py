import json, sys, os, subprocess, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load pre-repair vehicle model + config from git HEAD and pre-seed
# sys.modules so SimulationEnvironment picks them up.
for mod_name, path in [
    ("sim_core.vehicle.vehicle_config", "sim_core/vehicle/vehicle_config.py"),
    ("sim_core.vehicle.vehicle_model", "sim_core/vehicle/vehicle_model.py"),
]:
    src = subprocess.check_output(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True)
    mod = types.ModuleType(mod_name)
    mod.__dict__["__file__"] = f"<HEAD:{path}>"
    sys.modules[mod_name] = mod
    exec(compile(src, mod.__dict__["__file__"], "exec"), mod.__dict__)

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

with open("_tmp_view/rl_baseline_before.json", "w") as f:
    json.dump(out, f, indent=2)
