"""Headless TCP recording regression test.

Root cause fixed: `_record_step_if_needed` ran only inside the
`if not self.headless:` branch of the studio loop, so a headless
`main.py --headless` process recorded zero frames even while external
TCP clients stepped the env. The hook now runs in the headless branch,
and `--record DIR` + cleanup-flush make the whole path usable end-to-end.
"""
import threading
import time
import numpy as np
import pytest


def test_headless_records_external_tcp_steps(tmp_path):
    from sim_ui.app import SimulationStudioApp
    from sim_client import SimulationClient

    app = SimulationStudioApp(headless=True, port=9897)
    assert app.server_started
    app.start_recording("e2e_ep", str(tmp_path))

    # Drive the app loop headlessly: same calls the real loop makes.
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            app.server.poll_and_process()
            app._record_step_if_needed()
            time.sleep(0.001)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    time.sleep(0.1)

    try:
        client = SimulationClient(host="127.0.0.1", port=9897)
        spec = client.connect()
        assert spec["vector_dim"] > 0
        client.reset(seed=5)
        steps_done = 0
        for _ in range(30):
            obs, r, term, trunc, info = client.step([0.0, 0.4, 0.0])
            steps_done += 1
            if term or trunc:
                break
        time.sleep(0.2)  # let the headless loop drain the recorded steps
    finally:
        stop.set()
        t.join(timeout=3)

    assert app.recorder.is_recording
    frames = app.recorder.frames
    assert len(frames) >= steps_done, \
        f"recorded {len(frames)} frames for {steps_done} external steps"
    # frames carry real telemetry, not placeholders
    assert any(f["speed"] > 0 for f in frames)
    assert all(f.get("action") is not None for f in frames)

    app.stop_recording()
    import json, os
    ep = tmp_path / "e2e_ep" / "episode.json"
    assert ep.exists()
    data = json.loads(ep.read_text())
    assert data["metadata"]["track_name"]
    assert len(data["frames"]) >= steps_done

    # reload + replay through the real player
    from sim_recorder.recorder import EpisodeRecorder
    rec = EpisodeRecorder.load_from_file(str(ep))
    from sim_recorder.replay import EpisodeReplayPlayer
    rp = EpisodeReplayPlayer()
    rp.load_recording(rec)
    assert rp.total_frames >= steps_done


def test_recording_does_not_alter_simulation(tmp_path):
    """Recording on vs off must produce identical step results."""
    from sim_ui.app import SimulationStudioApp
    app = SimulationStudioApp(headless=True, port=9898)
    assert app.server_started

    env = app.env
    env.reset(seed=9)
    ref = [env.step([0.0, 0.3, 0.0]) for _ in range(20)]

    app.start_recording("nondet_check", str(tmp_path))
    env.reset(seed=9)
    rec = [env.step([0.0, 0.3, 0.0]) for _ in range(20)]
    app.stop_recording()

    for a, b in zip(ref, rec):
        np.testing.assert_allclose(a[0], b[0])
        assert a[1:] == b[1:]


def test_cleanup_flushes_active_recording(tmp_path):
    from sim_ui.app import SimulationStudioApp
    app = SimulationStudioApp(headless=True, port=0)
    app.start_recording("flush_ep", str(tmp_path))
    app.env.step([0.0, 0.3, 0.0])
    app._last_recorded_step = -1
    app._record_step_if_needed()
    app.cleanup()
    assert (tmp_path / "flush_ep" / "episode.json").exists()
