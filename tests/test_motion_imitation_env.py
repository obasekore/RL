from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from coppelia_rl.env_schema import motion_imitation as motion_imitation_module  # noqa: E402
from coppelia_rl.env_schema.generic_env import XmlDefinedEnv  # noqa: E402
from coppelia_rl.env_schema.spec import (  # noqa: E402
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    MotionImitationSpec,
    ObservationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
)
from coppelia_rl.sim_interface.client import SimClient  # noqa: E402
from motion_imitation_fixtures import (  # noqa: E402
    FL_HIP_PITCH_DELTA,
    QUADRUPED_DOF_NAMES,
    QUADRUPED_FEET,
    QUADRUPED_REST_RAD,
    QUADRUPED_SKELETON,
    build_fake_sim_scene_for_quadruped,
    build_synthetic_quadruped_clip,
)

_SCENE_PATH = Path("unused.ttt")
_FRAME_RATE = 20.0
_STEP_DT = 1.0 / _FRAME_RATE


def _client(fake_sim) -> SimClient:
    return SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)


def _make_env(
    fake_sim,
    clip_dir: Path,
    *,
    rsi: bool = True,
    reward_terms=None,
    termination_conditions=None,
    step_dt: float = _STEP_DT,
) -> XmlDefinedEnv:
    spec = EnvSpec(
        name="quad_walk",
        step_dt=step_dt,
        scene_path=_SCENE_PATH,
        observations=[ObservationSpec(kind="joint_position", key="fl_hip_pitch", ref="FL_hip_pitch")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[
                ActionEntrySpec(kind="joint_position", key="FL_hip_pitch", ref="FL_hip_pitch", value_range=(-3.0, 3.0))
            ],
        ),
        reward_terms=reward_terms if reward_terms is not None else [],
        termination_conditions=termination_conditions if termination_conditions is not None else [],
        motion_imitation=MotionImitationSpec(clip_dir=clip_dir, rsi=rsi),
    )
    return XmlDefinedEnv(spec, _client(fake_sim))


def _set_live_pose(fake_sim, handles: dict[str, int], angles: np.ndarray) -> None:
    for name, angle in zip(QUADRUPED_DOF_NAMES, angles):
        fake_sim.objects[handles[name]]["joint_position"] = float(angle)


# -- construction -----------------------------------------------------------------


def test_construction_wires_clip_and_skeleton_from_motion_imitation_block(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)

    env = _make_env(fake_sim, clip_dir)

    assert env._motion_imitation is not None
    assert env._motion_imitation.skeleton.name == QUADRUPED_SKELETON.name
    assert env._motion_imitation.dof_names == QUADRUPED_DOF_NAMES
    assert env._motion_imitation.n_frames == 4
    env.close()


def test_construction_raises_when_step_dt_does_not_match_clip_frame_rate(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, frame_rate=_FRAME_RATE)

    with pytest.raises(ValueError):
        _make_env(fake_sim, clip_dir, step_dt=0.1)


def test_construction_raises_when_clip_validation_status_is_error(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, validation_status="error")

    with pytest.raises(ValueError):
        _make_env(fake_sim, clip_dir)


def test_pose_tracking_requires_motion_imitation_block(fake_sim):
    build_fake_sim_scene_for_quadruped(fake_sim)
    spec = EnvSpec(
        name="t",
        step_dt=_STEP_DT,
        scene_path=_SCENE_PATH,
        observations=[],
        actions=ActionGroupSpec(action_type="continuous", entries=[]),
        reward_terms=[RewardTermSpec(kind="pose_tracking", weight=1.0)],
        termination_conditions=[],
    )
    with pytest.raises(ValueError):
        XmlDefinedEnv(spec, _client(fake_sim))


def test_fall_detection_requires_motion_imitation_block(fake_sim):
    build_fake_sim_scene_for_quadruped(fake_sim)
    spec = EnvSpec(
        name="t",
        step_dt=_STEP_DT,
        scene_path=_SCENE_PATH,
        observations=[],
        actions=ActionGroupSpec(action_type="continuous", entries=[]),
        reward_terms=[],
        termination_conditions=[TerminationConditionSpec(kind="fall_detection")],
    )
    with pytest.raises(ValueError):
        XmlDefinedEnv(spec, _client(fake_sim))


# -- RSI ------------------------------------------------------------------------


def test_rsi_reset_calls_instantaneous_joint_setter_with_sampled_frame_values(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=False)  # frame fixed to 0 -> deterministic

    env.reset(seed=0)

    fl_hip_pitch_handle = handles["FL_hip_pitch"]
    assert fake_sim.objects[fl_hip_pitch_handle]["joint_position"] == pytest.approx(QUADRUPED_REST_RAD[0])
    assert "joint_target_position" not in fake_sim.objects[fl_hip_pitch_handle]
    env.close()


def test_rsi_reset_sets_root_pose_from_reference_frame(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=False)

    env.reset(seed=0)

    root_handle = handles[QUADRUPED_SKELETON.root_bone]
    np.testing.assert_allclose(fake_sim.objects[root_handle]["pose"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    env.close()


def test_rsi_reset_samples_different_frames_across_multiple_resets(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4)
    env = _make_env(fake_sim, clip_dir, rsi=True)

    seen_frames = set()
    for seed in range(8):
        env.reset(seed=seed)
        seen_frames.add(env._motion_imitation.frame_index)
    env.close()

    assert len(seen_frames) >= 2


def test_rsi_disabled_always_resets_to_frame_zero(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4)
    env = _make_env(fake_sim, clip_dir, rsi=False)

    for seed in range(5):
        env.reset(seed=seed)
        assert env._motion_imitation.frame_index == 0
    env.close()


def test_rsi_disabled_still_clears_runtime_state_each_episode(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4, loop="ping_pong")
    env = _make_env(fake_sim, clip_dir, rsi=False)

    env.reset(seed=0)
    env._motion_imitation.direction = -1
    env._motion_imitation.prev_ee_world = {"FL_foot": np.array([1.0, 2.0, 3.0])}

    env.reset(seed=1)

    assert env._motion_imitation.direction == 1
    assert env._motion_imitation.prev_ee_world == {}
    env.close()


# -- rewards ----------------------------------------------------------------------
#
# These call the reward functions directly against env._motion_imitation rather than
# through env.step() - step() advances frame_index *before* computing reward (so that
# a step's resulting live state is compared against the *next* reference frame), which
# makes "read the current reference, set live state to match, then step()" racy against
# which frame index the reward actually gets computed against. Calling the reward
# function directly against a known frame_index sidesteps that entirely and tests the
# formula itself more directly.


def test_pose_tracking_reward_near_max_when_live_joints_equal_reference_frame(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=True)
    env.reset(seed=0)  # RSI sets live joints + root pose to exactly match the sampled frame

    reward = motion_imitation_module.pose_tracking_reward(env._motion_imitation)

    assert reward == pytest.approx(1.0, abs=1e-6)
    env.close()


def test_pose_tracking_reward_lower_when_live_joints_differ_from_reference_frame(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=True)
    env.reset(seed=0)
    mi = env._motion_imitation
    _set_live_pose(fake_sim, handles, mi.reference_dof_angles() + 1.0)

    reward = motion_imitation_module.pose_tracking_reward(mi)

    assert reward < 0.5
    env.close()


def test_velocity_tracking_reward_computed_against_reference_joint_velocities(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)
    mi = env._motion_imitation

    for name, velocity in zip(QUADRUPED_DOF_NAMES, mi.reference_dof_velocities()):
        fake_sim.objects[handles[name]]["joint_velocity"] = float(velocity)
    reward_matching = motion_imitation_module.velocity_tracking_reward(mi)

    for name in QUADRUPED_DOF_NAMES:
        fake_sim.objects[handles[name]]["joint_velocity"] = 10.0
    reward_mismatched = motion_imitation_module.velocity_tracking_reward(mi)

    assert reward_matching == pytest.approx(1.0, abs=1e-6)
    assert reward_matching > reward_mismatched
    env.close()


def test_end_effector_tracking_reward_uses_fk_of_reference_joint_angles(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)
    mi = env._motion_imitation
    reference_ee = mi.reference_end_effector_positions_root_relative()

    for name in QUADRUPED_FEET:
        fake_sim.objects[handles[name]]["position"] = reference_ee[name].copy()
    reward_matching = motion_imitation_module.end_effector_tracking_reward(mi)

    for name in QUADRUPED_FEET:
        fake_sim.objects[handles[name]]["position"] = reference_ee[name] + 1.0
    reward_mismatched = motion_imitation_module.end_effector_tracking_reward(mi)

    assert reward_matching == pytest.approx(1.0, abs=1e-6)
    assert reward_mismatched < reward_matching
    env.close()


def test_contact_matching_reward_neutral_during_reference_swing_frames(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4)
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)
    mi = env._motion_imitation
    mi.frame_index = 1  # FL_foot in swing per the fixture (fl_contact = [True, False, False, False])

    motion_imitation_module.contact_matching_reward(mi, _STEP_DT)  # establishes prev_ee_world
    fake_sim.objects[handles["FL_foot"]]["position"] = np.array([5.0, 5.0, 5.0])  # large jump, should be ignored
    reward = motion_imitation_module.contact_matching_reward(mi, _STEP_DT)

    assert reward == pytest.approx(1.0, abs=1e-6)
    env.close()


def test_contact_matching_reward_penalizes_slip_during_reference_stance_frames(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4)
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)
    mi = env._motion_imitation  # frame_index=0: both FL_foot and RL_foot are in stance per the fixture

    motion_imitation_module.contact_matching_reward(mi, _STEP_DT)  # establishes prev_ee_world
    fake_sim.objects[handles["RL_foot"]]["position"] = np.array([5.0, 5.0, 5.0])  # large slip
    reward = motion_imitation_module.contact_matching_reward(mi, _STEP_DT)

    assert reward < 1.0
    env.close()


# -- phase advancement --------------------------------------------------------------


def test_frame_advances_by_one_per_step(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4, loop="one_shot")
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)
    assert env._motion_imitation.frame_index == 0

    env.step(np.array([0.0], dtype=np.float32))
    assert env._motion_imitation.frame_index == 1
    env.close()


def test_cyclic_clip_wraps_frame_index_at_boundary(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4, loop="cyclic")
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)

    for _ in range(4):
        env.step(np.array([0.0], dtype=np.float32))

    assert env._motion_imitation.frame_index == 0
    env.close()


def test_one_shot_clip_truncates_naturally_on_exhaustion(fake_sim, tmp_path):
    n_frames = 4
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=n_frames, loop="one_shot")
    env = _make_env(fake_sim, clip_dir, rsi=False, termination_conditions=[])
    env.reset(seed=0)

    # frame_index starts at 0; advance_frame() only sets exhausted once frame_index+1
    # would reach n_frames, i.e. after n_frames steps (0->1->2->3, then the 4th step
    # detects 3+1>=4 and stops advancing further).
    terminated = truncated = False
    for _ in range(n_frames):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

    assert truncated is True
    assert terminated is False
    assert env._motion_imitation.exhausted is True
    env.close()


def test_ping_pong_clip_reverses_direction_at_boundary(fake_sim, tmp_path):
    build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path, n_frames=4, loop="ping_pong")
    env = _make_env(fake_sim, clip_dir, rsi=False)
    env.reset(seed=0)

    frames = [env._motion_imitation.frame_index]
    for _ in range(5):
        env.step(np.array([0.0], dtype=np.float32))
        frames.append(env._motion_imitation.frame_index)

    assert frames == [0, 1, 2, 3, 2, 1]
    env.close()


# -- fall detection -----------------------------------------------------------------


def test_fall_detection_terminates_on_low_root_height(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(
        fake_sim, clip_dir, rsi=False, termination_conditions=[TerminationConditionSpec(kind="fall_detection")]
    )
    env.reset(seed=0)
    root_handle = handles[QUADRUPED_SKELETON.root_bone]
    fake_sim.objects[root_handle]["position"] = np.array([0.0, 0.0, 0.0])  # below min_root_height=0.12

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    env.close()


def test_fall_detection_terminates_on_excessive_tilt(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(
        fake_sim, clip_dir, rsi=False, termination_conditions=[TerminationConditionSpec(kind="fall_detection")]
    )
    env.reset(seed=0)
    root_handle = handles[QUADRUPED_SKELETON.root_bone]
    fake_sim.objects[root_handle]["position"] = np.array([0.0, 0.0, 0.5])  # above min height
    # 90 degree rotation about the x axis: [qw,qx,qy,qz] = [cos45, sin45, 0, 0] -> tilts
    # the z-up axis onto -y, an 90 degree tilt, exceeding max_tilt_deg=60.
    c, s = 0.70710678, 0.70710678
    fake_sim.objects[root_handle]["pose"] = np.array([0.0, 0.0, 0.5, s, 0.0, 0.0, c])  # native xyzw: [x,y,z,qx,qy,qz,qw]

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

    assert terminated is True
    env.close()


def test_fall_detection_does_not_terminate_when_within_thresholds(fake_sim, tmp_path):
    handles = build_fake_sim_scene_for_quadruped(fake_sim)
    clip_dir = build_synthetic_quadruped_clip(tmp_path)
    env = _make_env(
        fake_sim, clip_dir, rsi=False, termination_conditions=[TerminationConditionSpec(kind="fall_detection")]
    )
    env.reset(seed=0)
    root_handle = handles[QUADRUPED_SKELETON.root_bone]
    fake_sim.objects[root_handle]["position"] = np.array([0.0, 0.0, 0.3])  # above min_root_height

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
    env.close()
