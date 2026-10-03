"""
Verification test for autonomous PID closed-loop driving.
"""

import subprocess
import sys
import time
from sim_client.agents.pid_driver import run_pid_agent


def test_pid_autonomous_driving():
    port = 8792
    cmd = [sys.executable, "-u", "main.py", "--headless", "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    time.sleep(1.5)

    try:
        # Run PID agent for 150 steps (~2.5s of simulated driving)
        run_pid_agent(host="127.0.0.1", port=port, max_steps=150)
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=2.0)
        except Exception:
            proc.kill()
