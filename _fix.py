import re
src = open("tests/test_track_semantics.py").read()
src = src.replace("""    laps0 = env.laps_completed
    cps0 = env.checkpoints_passed""", """    laps0 = env.checkpoint_tracker.laps_completed
    cps0 = env.checkpoint_tracker.total_checkpoints_passed""")
src = src.replace("""    assert env.laps_completed == laps0
    assert env.checkpoints_passed == cps0""", """    assert env.checkpoint_tracker.laps_completed == laps0
    assert env.checkpoint_tracker.total_checkpoints_passed == cps0""")
src = src.replace("""    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, True, 1, 1, False)
    assert (t, tr) == (True, False)
    # closed track, lap count reached -> terminates
    ev.reset()
    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, True, 1, 1, True)
    assert (t, tr) == (True, False)
    # closed track, partial lap -> no completion
    ev.reset()
    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, True, 0, 1, True)
    assert (t, tr) == (False, False)""", """    # (dt, is_colliding, is_on_road, heading_error, cp_passed, laps, is_closed)
    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, 1, False)
    assert (t, tr) == (True, False)
    # closed track, lap count reached -> terminates
    ev.reset()
    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, 1, True)
    assert (t, tr) == (True, False)
    # closed track, partial lap -> no completion
    ev.reset()
    t, tr, _ = ev.evaluate(0.016, False, True, 0.0, False, 0, True)
    assert (t, tr) == (False, False)""")
src = src.replace("assert env.laps_completed == 1", "assert env.checkpoint_tracker.laps_completed == 1")
open("tests/test_track_semantics.py","w").write(src)
