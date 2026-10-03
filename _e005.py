import sys, time
sys.path.insert(0, ".")
from agentRL.experiments.matrix import run_experiment
res = run_experiment("E005", run_dir="docs/master_audit/evidence/agent/E005_reduced",
                     steps_override=4000, eval_episodes=1)
import json
print(json.dumps(res, default=str)[:800])
