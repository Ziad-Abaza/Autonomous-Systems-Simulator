"""ObservationSpec + ObsEncoder tests."""
from __future__ import annotations

import numpy as np
import pytest


def test_spec_dims():
    from agentRL.obs.spec import ObservationSpec, PRESETS

    s8 = ObservationSpec(channel_names=PRESETS["state8"])
    assert s8.vector_dim == 8
    assert s8.input_dim == 8

    full = ObservationSpec(channel_names=PRESETS["full23"])
    assert full.vector_dim == 23
    assert full.input_dim == 23

    pa = ObservationSpec(channel_names=PRESETS["state8"], prev_action=True)
    assert pa.input_dim == 8 + 3

    fs = ObservationSpec(channel_names=PRESETS["state8"], frame_stack=4)
    assert fs.input_dim == 32

    both = ObservationSpec(channel_names=PRESETS["state8"], frame_stack=2,
                           prev_action=True)
    assert both.input_dim == 16 + 3


def test_spec_roundtrip():
    from agentRL.obs.spec import ObservationSpec, PRESETS

    spec = ObservationSpec(channel_names=PRESETS["state8"], frame_stack=3,
                           prev_action=True)
    assert ObservationSpec.from_dict(spec.to_dict()) == spec


def test_channel_subset_validation():
    from agentRL.obs.spec import ObservationSpec

    with pytest.raises(ValueError):
        ObservationSpec(channel_names=("speed", "oracle_telemetry_x"))
    with pytest.raises(ValueError):
        ObservationSpec(channel_names=())


def test_encoder_stack_order():
    from agentRL.obs.spec import ObservationSpec, PRESETS
    from agentRL.obs.encoder import ObsEncoder

    enc = ObsEncoder(ObservationSpec(channel_names=PRESETS["state8"],
                                     frame_stack=3))
    enc.reset()
    a = np.arange(8, dtype=np.float32)
    b = a + 100
    c = a + 200
    enc.encode(a)
    enc.encode(b)
    out = enc.encode(c)
    # oldest -> newest concatenation
    assert np.array_equal(out, np.concatenate([a, b, c]))


def test_encoder_prev_action():
    from agentRL.obs.spec import ObservationSpec, PRESETS
    from agentRL.obs.encoder import ObsEncoder

    enc = ObsEncoder(ObservationSpec(channel_names=PRESETS["state8"],
                                     prev_action=True))
    enc.reset()
    obs = np.ones(8, dtype=np.float32)
    act = np.array([0.5, 0.2, 0.0], dtype=np.float32)
    out = enc.encode(obs, prev_action=act)
    assert out.shape == (11,)
    assert np.array_equal(out[8:], act)


def test_image_reserved():
    from agentRL.obs.spec import ObservationSpec, PRESETS
    from agentRL.obs.encoder import ObsEncoder

    enc = ObsEncoder(ObservationSpec(channel_names=PRESETS["state8"],
                                     image=True))
    with pytest.raises(NotImplementedError):
        enc.encode(np.zeros(8, dtype=np.float32))
