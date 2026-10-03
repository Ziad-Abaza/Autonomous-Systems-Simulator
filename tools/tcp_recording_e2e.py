"""
End-to-end TCP external-AI recording test.

Boots the real app, connects a SimGymEnv client over TCP, drives the
environment via STEP messages while recording, then verifies:
  - episode.json + manifest.json written under the recordings dir
  - frames contain the external actions, rewards, and state
  - the saved episode loads back into the replay player

    python tools/tcp_recording_e2e.py
"""
import json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
from sim_ui.app import SimulationStudioApp

app = SimulationStudioApp(width=1280, height=720, headless=False, port=8792)
checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# pump the server + recorder the way the app loop does (server only
# answers while polled — the test client is synchronous/blocking)
_stop_pump = threading.Event()

def pump_worker():
    while not _stop_pump.is_set():
        app.server.poll_and_process()
        app._record_step_if_needed()
        time.sleep(0.001)

pump_thread = threading.Thread(target=pump_worker, daemon=True)


try:
    # ---- start recording through the real dialog path ----
    app.ws_tab = "SIMULATE"
    app.start_recording("tcp_e2e_episode", app.record_dir)
    check("recording started", app.recorder.is_recording)

    # ---- external client drives the env over TCP ----
    pump_thread.start()
    from sim_client.gym_env import SimGymEnv
    client = SimGymEnv(host="127.0.0.1", port=8792)
    obs, info = client.reset(seed=7)

    steps_sent = 25
    for i in range(steps_sent):
        client.step([0.1, 0.5, 0.0])
    time.sleep(0.1)  # let the pump drain the last steps

    check("env advanced via TCP steps",
          app.env.current_step >= steps_sent)
    check("frames recorded for external steps",
          len(app.recorder.frames) >= steps_sent)

    fr = app.recorder.frames[-1]
    check("frame has action", "action" in fr and len(fr["action"]) >= 2)
    check("frame has reward", "reward" in fr)
    check("frame has position", "pos" in fr and len(fr["pos"]) == 3)

    client.close()

    # ---- stop → files on disk ----
    app.stop_recording()
    ep_dir = os.path.join(app.record_dir, "tcp_e2e_episode")
    ep_file = os.path.join(ep_dir, "episode.json")
    man_file = os.path.join(ep_dir, "manifest.json")
    check("episode.json written", os.path.exists(ep_file))
    check("manifest.json written", os.path.exists(man_file))
    if os.path.exists(man_file):
        man = json.load(open(man_file, encoding="utf-8"))
        check("manifest has step count", man.get("steps", 0) >= steps_sent)
        check("manifest names the track", bool(man.get("track_name")))
    check("summary dialog queued", app.active_dialog == "rec_summary")
    check("summary has next-step info",
          app.recording_summary.get("steps", 0) >= steps_sent)

    # ---- saved episode loads into the replay player ----
    app.load_replay(ep_file)
    check("replay loaded from saved file",
          app.replay_player.total_frames >= steps_sent)
    check("ws switched to REPLAY", app.ws_tab == "REPLAY")
    app.replay_player.seek(10)
    app._apply_replay_frame()
    st = app.env.vehicle.state
    check("replay applied pose",
          abs(st.pos.x) + abs(st.pos.y) > 0.001)
finally:
    _stop_pump.set()
    try:
        client.close()
    except Exception:
        pass
    app.cleanup()

failed = [n for n, ok in checks if not ok]
print(f"\n{len(checks)-len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
