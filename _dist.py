from sim_client import SimulationClient
import numpy as np
c = SimulationClient(port=9789)
spec = c.connect()
print("handshake:", spec["contract_source"], "vector_dim:", spec["vector_dim"],
      "track:", spec["track_name"], "protocol:", spec["protocol_version"])
obs, info = c.reset(seed=1)
print("reset obs:", np.asarray(obs).shape)
tot = 0.0
for i in range(30):
    obs, r, t, tr, info = c.step([0.0, 0.5, 0.0]); tot += r
    if t or tr: break
print("30 steps done, return=%.2f, reason=%s" % (tot, info.get("termination_reason")))
c.close()
