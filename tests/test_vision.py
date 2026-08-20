import numpy as np
import pytest

from coppelia_rl.sim_interface.vision import VisionSensor


def test_read_enables_explicit_handling(fake_sim):
    """Explicit handling is a runtime flag, not persisted into a saved scene
    (confirmed live: XmlDefinedEnv.reset() reloads the scene from disk every
    episode, which reverts it) - so it must be re-asserted on every read,
    not just once when the VisionSensor wrapper is constructed."""
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)
    assert not fake_sim.getExplicitHandling(handle) & 1

    vision.get_rgb()

    assert fake_sim.getExplicitHandling(handle) & 1


def test_explicit_handling_reasserted_after_scene_reload(fake_sim):
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)
    vision.get_rgb()

    # Simulate a scene reload reverting the runtime flag, as observed live.
    fake_sim.objects[handle]["explicit_handling"] = 0

    vision.get_rgb()  # would raise if explicit handling weren't reasserted


def test_get_rgb_shape_and_dtype(fake_sim):
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)

    rgb = vision.get_rgb()

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8


def test_get_depth_shape_dtype_and_range(fake_sim):
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)

    depth = vision.get_depth()

    assert depth.shape == (4, 4)
    assert depth.dtype == np.float32
    assert np.all((depth >= 0.0) & (depth <= 1.0))


def test_get_rgb_calls_handle_vision_sensor_first(fake_sim):
    """Regression guard: sim.getVisionSensorImg's own manual page says the
    returned data "doesn't make sense" without handleVisionSensor being
    called first - FakeSim enforces the same ordering, consuming a
    single-use flag, so removing that call would fail this test."""
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)

    vision.get_rgb()  # would raise if handleVisionSensor wasn't called

    calls = [name for name, _ in fake_sim.calls if name == "handleVisionSensor"]
    assert len(calls) == 1


def test_get_depth_calls_handle_vision_sensor_first(fake_sim):
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)

    vision.get_depth()  # would raise if handleVisionSensor wasn't called


def test_reading_without_explicit_handling_raises(fake_sim):
    """Confirms FakeSim mirrors the real error ("object not tagged for
    explicit handling") when handleVisionSensor is called on a sensor that
    was never enabled for it - constructing through VisionSensor always
    enables it, so this drives FakeSim directly to prove the guard exists."""
    handle = fake_sim._new_handle("vision")

    with pytest.raises(RuntimeError):
        fake_sim.handleVisionSensor(handle)


def test_reading_twice_requires_handling_each_time(fake_sim):
    handle = fake_sim._new_handle("vision")
    vision = VisionSensor(fake_sim, handle)

    vision.get_rgb()
    vision.get_rgb()  # each read re-handles - the flag is single-use, not sticky
