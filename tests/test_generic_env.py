from pathlib import Path

import numpy as np

from coppelia_rl.env_schema.generic_env import XmlDefinedEnv
from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    ObservationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
)
from coppelia_rl.sim_interface.client import SimClient

_SCENE_PATH = Path("unused.ttt")


def _client(fake_sim):
    return SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)


def _make_env(fake_sim, **kwargs) -> XmlDefinedEnv:
    spec = EnvSpec(
        name="test",
        step_dt=0.05,
        scene_path=_SCENE_PATH,
        observations=kwargs.get("observations", []),
        actions=kwargs.get("actions", ActionGroupSpec(action_type="continuous", entries=[])),
        reward_terms=kwargs.get("reward_terms", []),
        termination_conditions=kwargs.get("termination_conditions", [TerminationConditionSpec(kind="max_steps", value=100)]),
        domain_randomization=kwargs.get("domain_randomization"),
    )
    return XmlDefinedEnv(spec, _client(fake_sim))


# -- observations -------------------------------------------------------------


def test_joint_and_object_observations(fake_sim):
    joint = fake_sim._new_handle("joint", joint_position=0.3, joint_velocity=0.1)
    fake_sim.setObjectAlias(joint, "J1")
    a = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(a, "A")
    fake_sim.objects[a]["position"] = np.array([1.0, 2.0, 3.0])
    b = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(b, "B")

    env = _make_env(
        fake_sim,
        observations=[
            ObservationSpec(kind="joint_position", key="j1_pos", ref="J1"),
            ObservationSpec(kind="joint_velocity", key="j1_vel", ref="J1"),
            ObservationSpec(kind="object_position", key="a_pos", ref="A", frame="world"),
            ObservationSpec(kind="object_pose", key="a_pose", ref="A", frame="world"),
        ],
    )
    assert set(env.observation_space.spaces.keys()) == {"j1_pos", "j1_vel", "a_pos", "a_pose"}

    obs, info = env.reset()
    assert obs["j1_pos"].shape == (1,)
    assert obs["j1_vel"].shape == (1,)
    np.testing.assert_allclose(obs["a_pos"], [1.0, 2.0, 3.0])
    assert obs["a_pose"].shape == (7,)
    env.close()


def test_camera_observation(fake_sim):
    cam = fake_sim._new_handle("vision", resolution=(4, 4))
    fake_sim.setObjectAlias(cam, "Cam1")

    env = _make_env(
        fake_sim,
        observations=[ObservationSpec(kind="camera", key="cam", ref="Cam1", resolution=(4, 4), channels="rgb")],
    )
    assert env.observation_space["cam"].shape == (4, 4, 3)

    obs, info = env.reset()
    assert obs["cam"].shape == (4, 4, 3)
    assert obs["cam"].dtype == np.uint8
    env.close()


def test_custom_observation(fake_sim):
    env = _make_env(
        fake_sim,
        observations=[
            ObservationSpec(kind="custom", key="custom_obs", callable="tests.custom_fixtures:constant_observation")
        ],
    )
    obs, info = env.reset()
    np.testing.assert_allclose(obs["custom_obs"], [0.5, -0.5])
    env.close()


# -- actions --------------------------------------------------------------------


def test_continuous_joint_velocity_action(fake_sim):
    joint = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    fake_sim.setObjectAlias(joint, "J1")

    env = _make_env(
        fake_sim,
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="J1", ref="J1", value_range=(-1.0, 1.0))],
        ),
    )
    assert env.action_space.shape == (1,)
    env.reset()
    env.step(np.array([0.5], dtype=np.float32))
    assert fake_sim.objects[joint]["joint_target_velocity"] == 0.5
    # Regression guard: confirmed live that setJointTargetVelocity() is
    # silently near-inert unless the joint is actually in "velocity" dynamic
    # control mode - stock models like UR5.ttm default to "position" mode.
    assert fake_sim.objects[joint][fake_sim.jointintparam_dynctrlmode] == fake_sim.jointdynctrl_velocity
    env.close()


def test_joint_control_mode_reasserted_after_scene_reload(fake_sim):
    """Regression guard for a bug found live: reset() reloads the scene from
    disk on every episode, which - like vision sensors' explicit-handling
    flag in milestone 6 - silently reverts the joint's dynamic control mode
    if it's only ever set once at env construction. The mode must be
    reapplied on every reset(), not just the first."""
    joint = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    fake_sim.setObjectAlias(joint, "J1")

    env = _make_env(
        fake_sim,
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="J1", ref="J1", value_range=(-1.0, 1.0))],
        ),
    )
    env.reset()
    assert fake_sim.objects[joint][fake_sim.jointintparam_dynctrlmode] == fake_sim.jointdynctrl_velocity

    # Simulate what a real scene reload does: revert the runtime-only flag.
    fake_sim.objects[joint][fake_sim.jointintparam_dynctrlmode] = fake_sim.jointdynctrl_position

    env.reset()

    assert fake_sim.objects[joint][fake_sim.jointintparam_dynctrlmode] == fake_sim.jointdynctrl_velocity
    env.close()


def test_continuous_joint_position_action_sets_position_control_mode(fake_sim):
    joint = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    fake_sim.setObjectAlias(joint, "J1")

    env = _make_env(
        fake_sim,
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_position", key="J1", ref="J1", value_range=(-1.0, 1.0))],
        ),
    )
    env.reset()

    assert fake_sim.objects[joint][fake_sim.jointintparam_dynctrlmode] == fake_sim.jointdynctrl_position
    env.close()


def test_continuous_custom_action(fake_sim):
    from tests.custom_fixtures import recording_action

    env = _make_env(
        fake_sim,
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="custom", key="custom_action", callable="tests.custom_fixtures:recording_action")],
        ),
    )
    env.reset()
    env.step(np.array([1.5], dtype=np.float32))
    assert recording_action.last_value == np.float32(1.5)
    env.close()


def test_discrete_event_action(fake_sim):
    env = _make_env(
        fake_sim,
        actions=ActionGroupSpec(
            action_type="discrete",
            entries=[
                ActionEntrySpec(kind="event", key="open", signal="gripper_open"),
                ActionEntrySpec(kind="event", key="close", signal="gripper_close"),
            ],
        ),
    )
    assert env.action_space.n == 2
    env.reset()
    env.step(1)
    assert fake_sim._int32_signals["gripper_open"] == 0
    assert fake_sim._int32_signals["gripper_close"] == 1
    env.close()


# -- reward -----------------------------------------------------------------------


def test_distance_reward(fake_sim):
    a = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(a, "A")
    b = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(b, "B")
    fake_sim.objects[b]["position"] = np.array([3.0, 4.0, 0.0])

    env = _make_env(fake_sim, reward_terms=[RewardTermSpec(kind="distance", weight=-1.0, from_ref="A", to_ref="B")])
    env.reset()
    _, reward, _, _, _ = env.step(np.array([]))
    assert reward == -5.0
    env.close()


def test_collision_reward(fake_sim):
    s1 = fake_sim._new_handle("shape")
    fake_sim.setObjectAlias(s1, "S1")
    s2 = fake_sim._new_handle("shape")
    fake_sim.setObjectAlias(s2, "S2")
    fake_sim.collisions[frozenset({s1, s2})] = True

    env = _make_env(fake_sim, reward_terms=[RewardTermSpec(kind="collision", weight=5.0, object1="S1", object2="S2")])
    env.reset()
    _, reward, _, _, _ = env.step(np.array([]))
    assert reward == 5.0
    env.close()


def test_signal_event_reward_is_one_shot(fake_sim):
    env = _make_env(fake_sim, reward_terms=[RewardTermSpec(kind="signal_event", weight=10.0, signal="grasp")])
    env.reset()
    fake_sim.setInt32Signal("grasp", 1)
    _, reward1, _, _, _ = env.step(np.array([]))
    assert reward1 == 10.0
    _, reward2, _, _, _ = env.step(np.array([]))
    assert reward2 == 0.0
    env.close()


def test_custom_reward(fake_sim):
    env = _make_env(fake_sim, reward_terms=[RewardTermSpec(kind="custom", weight=2.0, callable="tests.custom_fixtures:custom_reward")])
    env.reset()
    _, reward, _, _, _ = env.step(np.array([]))
    assert reward == 6.0
    env.close()


# -- termination ------------------------------------------------------------------


def test_max_steps_truncates(fake_sim):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="max_steps", value=2)])
    env.reset()
    _, _, terminated1, truncated1, _ = env.step(np.array([]))
    assert not terminated1 and not truncated1
    _, _, terminated2, truncated2, _ = env.step(np.array([]))
    assert not terminated2 and truncated2
    env.close()


def test_signal_termination_terminates_and_is_cleared_on_reset(fake_sim):
    env = _make_env(fake_sim, termination_conditions=[TerminationConditionSpec(kind="signal", name="task_done")])
    env.reset()
    fake_sim.setInt32Signal("task_done", 1)
    _, _, terminated, truncated, _ = env.step(np.array([]))
    assert terminated and not truncated
    assert fake_sim._int32_signals["task_done"] == 0

    # A stale truthy signal from a prior episode must not leak into the next one.
    fake_sim.setInt32Signal("task_done", 1)
    env.reset()
    assert fake_sim._int32_signals["task_done"] == 0
    env.close()


def test_custom_termination(fake_sim):
    env = _make_env(
        fake_sim, termination_conditions=[TerminationConditionSpec(kind="custom", callable="tests.custom_fixtures:custom_termination")]
    )
    env.reset()
    _, _, terminated1, _, _ = env.step(np.array([]))
    assert not terminated1
    _, _, terminated2, _, _ = env.step(np.array([]))
    assert terminated2
    env.close()


def test_client_property_exposes_sim_client_for_escape_hatch(fake_sim):
    env = _make_env(fake_sim)
    assert env.client is env._client


# -- domain randomization ("enabled" just by the spec block being present) --------------


def test_domain_randomization_resamples_on_reset(fake_sim):
    from coppelia_rl.env_schema.spec import DomainRandomizationSpec, RangeRandomizationSpec

    cube = fake_sim._new_handle("shape")
    fake_sim.setObjectAlias(cube, "cube")

    env = _make_env(
        fake_sim,
        domain_randomization=DomainRandomizationSpec(
            masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))]
        ),
    )
    env.reset()

    assert 0.1 <= fake_sim.objects[cube]["mass"] <= 0.2
    env.close()


def test_domain_randomization_disabled_when_spec_absent(fake_sim):
    env = _make_env(fake_sim)
    assert env._randomizer is None
    env.close()


def test_domain_randomization_resample_on_once_does_not_reresample(fake_sim):
    from coppelia_rl.env_schema.spec import DomainRandomizationSpec, RangeRandomizationSpec

    cube = fake_sim._new_handle("shape")
    fake_sim.setObjectAlias(cube, "cube")

    env = _make_env(
        fake_sim,
        domain_randomization=DomainRandomizationSpec(
            masses=[RangeRandomizationSpec(ref="cube", value_range=(0.1, 0.2))], resample_on="once"
        ),
    )
    env.reset()
    first_mass = fake_sim.objects[cube]["mass"]
    env.reset()
    second_mass = fake_sim.objects[cube]["mass"]

    assert first_mass == second_mass
    env.close()
