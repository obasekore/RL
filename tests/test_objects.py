import numpy as np
import pytest

from coppelia_rl.sim_interface.objects import Joint, Sensor, SceneObject, Signal


def test_scene_object_position_roundtrip(fake_sim):
    handle = fake_sim._new_handle("dummy")
    obj = SceneObject(fake_sim, handle)

    obj.set_position([1.0, 2.0, 3.0])
    np.testing.assert_allclose(obj.get_position(), [1.0, 2.0, 3.0])


def test_scene_object_pose_roundtrip(fake_sim):
    handle = fake_sim._new_handle("dummy")
    obj = SceneObject(fake_sim, handle)

    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    obj.set_pose(pose)
    np.testing.assert_allclose(obj.get_pose(), pose)


def test_scene_object_set_name_registers_alias(fake_sim):
    handle = fake_sim._new_handle("dummy")
    obj = SceneObject(fake_sim, handle)

    obj.set_name("MyDummy")

    assert fake_sim.getObject("/MyDummy") == handle


def test_joint_target_velocity_and_readback(fake_sim):
    handle = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    joint = Joint(fake_sim, handle)

    joint.set_target_velocity(0.5)
    fake_sim.step()

    assert joint.get_joint_velocity() == 0.5
    assert joint.get_joint_position() == 0.5 * 0.05


def test_joint_set_control_mode(fake_sim):
    handle = fake_sim._new_handle("joint")
    joint = Joint(fake_sim, handle)

    joint.set_control_mode("velocity")

    assert fake_sim.objects[handle][fake_sim.jointintparam_dynctrlmode] == fake_sim.jointdynctrl_velocity


def test_joint_set_control_mode_rejects_unknown_mode(fake_sim):
    handle = fake_sim._new_handle("joint")
    joint = Joint(fake_sim, handle)

    with pytest.raises(ValueError):
        joint.set_control_mode("sideways")


def test_joint_get_interval(fake_sim):
    handle = fake_sim._new_handle("joint")
    joint = Joint(fake_sim, handle)

    cyclic, lo, rng = joint.get_interval()

    assert cyclic is False
    assert lo == -3.14159
    assert rng == 6.28318


def test_sensor_read_proximity(fake_sim):
    handle = fake_sim._new_handle("sensor")
    sensor = Sensor(fake_sim, handle)

    detected, distance, point, obj_handle, normal = sensor.read_proximity()

    assert detected is True
    assert distance == 0.5
    assert point.shape == (3,)
    assert normal.shape == (3,)


def test_sensor_read_force(fake_sim):
    handle = fake_sim._new_handle("sensor")
    sensor = Sensor(fake_sim, handle)

    available, force, torque = sensor.read_force()

    assert available is True
    np.testing.assert_allclose(force, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(torque, [0.0, 0.0, 0.0])


def test_signal_int32_roundtrip(fake_sim):
    signal = Signal(fake_sim, "episode_success")

    signal.set_int32(1)

    assert signal.get_int32() == 1

    signal.clear()

    assert signal.get_int32() is None
