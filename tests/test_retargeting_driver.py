from pathlib import Path

import numpy as np
import pytest

# coppelia_rl.retargeting.driver imports coppelia_rl.motion.clip, which transitively
# imports coppelia_rl.motion.tracks_hdf5 (h5py at module level, unconditional) even for
# the array-level retarget_tracks/check_joint_velocities functions that don't touch
# HDF5 themselves - so the whole file needs the guard, not just the Clip-level test.
pytest.importorskip("h5py")

from coppelia_rl.motion.clip import Clip, open_clip, save_clip  # noqa: E402
from coppelia_rl.motion.schema import ChannelsBlock, ClipHeader, ClipMeta, Provenance, SkeletonBinding  # noqa: E402
from coppelia_rl.motion.tracks_hdf5 import ContactTrack, EndEffectorTrack, TracksData  # noqa: E402
from coppelia_rl.retargeting.driver import check_joint_velocities, retarget_clip, retarget_tracks  # noqa: E402
from coppelia_rl.retargeting.fk import dof_rest_angles_rad, effector_positions  # noqa: E402
from coppelia_rl.skeleton.parser import dof_joint_names, ordered_dof_names, parse_skeleton_yaml  # noqa: E402

_QUADRUPED_PATH = Path(__file__).resolve().parents[1] / "skeletons" / "quadruped_generic.yaml"
_QUADRUPED = parse_skeleton_yaml(_QUADRUPED_PATH)
_QUADRUPED_DOF_NAMES = ordered_dof_names(_QUADRUPED)
_QUADRUPED_JOINT_NAMES = dof_joint_names(_QUADRUPED)


def _zero_angles(n_frames: int) -> np.ndarray:
    return np.zeros((n_frames, len(_QUADRUPED_DOF_NAMES)))


# -- warm start -----------------------------------------------------------------


def test_warm_start_reduces_iterations_after_first_frame():
    rest_rad = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    perturbed = rest_rad.copy()
    perturbed[0] += np.radians(15)
    target = effector_positions(_QUADRUPED, _QUADRUPED_DOF_NAMES, perturbed, ["FL_foot"])[0]

    n_frames = 3
    end_effector_targets = {"FL_foot": np.tile(target, (n_frames, 1))}
    contact = {"FL_foot": np.zeros(n_frames, dtype=bool)}

    result = retarget_tracks(
        _QUADRUPED, _QUADRUPED_DOF_NAMES, n_frames, end_effector_targets, contact, frame_rate=30.0
    )

    assert result.iterations_per_frame[0] > 1
    assert result.iterations_per_frame[1] == 1
    assert result.iterations_per_frame[2] == 1


# -- joint velocity post-check -----------------------------------------------------


def test_velocity_check_flags_exceedance():
    angles = _zero_angles(3)
    idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    # Jump at frame 0->1 (30 rad/s, far above FL_hip's max_velocity=6.0), then hold -
    # so only one transition (diff index 0) exceeds, not two.
    angles[1, idx] = 1.0
    angles[2, idx] = 1.0

    issues = check_joint_velocities(_QUADRUPED, _QUADRUPED_DOF_NAMES, _QUADRUPED_JOINT_NAMES, angles, frame_rate=30.0)

    assert len(issues) == 1
    assert issues[0].type == "joint_velocity_exceeded"
    assert issues[0].joint == "FL_hip"
    assert issues[0].frame_range == (0, 1)
    assert issues[0].severity == "warning"


def test_velocity_check_within_limit_no_issue():
    angles = _zero_angles(3)
    idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    angles[1, idx] = 0.1  # 3 rad/s < max_velocity=6.0

    issues = check_joint_velocities(_QUADRUPED, _QUADRUPED_DOF_NAMES, _QUADRUPED_JOINT_NAMES, angles, frame_rate=30.0)

    assert issues == []


def test_velocity_check_multiaxis_joint_single_issue_not_per_axis():
    angles = _zero_angles(3)
    pitch_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    roll_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_roll")
    angles[1, pitch_idx] = 1.0  # both axes of FL_hip exceed at the same transition
    angles[1, roll_idx] = 1.0

    issues = check_joint_velocities(_QUADRUPED, _QUADRUPED_DOF_NAMES, _QUADRUPED_JOINT_NAMES, angles, frame_rate=30.0)

    fl_hip_issues = [i for i in issues if i.joint == "FL_hip"]
    assert len(fl_hip_issues) == 1


def test_velocity_check_short_clip_no_crash():
    assert check_joint_velocities(_QUADRUPED, _QUADRUPED_DOF_NAMES, _QUADRUPED_JOINT_NAMES, _zero_angles(1), 30.0) == []


# -- ik_target_unreachable end-to-end via retarget_tracks --------------------------


def test_ik_target_unreachable_end_to_end():
    rest_rad = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    reachable = effector_positions(_QUADRUPED, _QUADRUPED_DOF_NAMES, rest_rad, ["FL_foot"])[0]

    n_frames = 4
    curve = np.tile(reachable, (n_frames, 1))
    curve[2] = np.array([10.0, 10.0, 10.0])  # unreachable at frame 2 only
    end_effector_targets = {"FL_foot": curve}
    contact = {"FL_foot": np.zeros(n_frames, dtype=bool)}

    result = retarget_tracks(
        _QUADRUPED, _QUADRUPED_DOF_NAMES, n_frames, end_effector_targets, contact, frame_rate=30.0, max_iterations=50
    )

    unreachable_issues = [i for i in result.validation_issues if i.type == "ik_target_unreachable"]
    assert len(unreachable_issues) == 1
    assert unreachable_issues[0].joint == "FL_foot"
    assert unreachable_issues[0].severity == "error"
    assert unreachable_issues[0].frame_range == (2, 2)
    assert result.joint_angles_rad.shape == (n_frames, len(_QUADRUPED_DOF_NAMES))
    assert not np.isnan(result.joint_angles_rad).any()


# -- retarget_clip integration -------------------------------------------------------


def test_retarget_clip_integration(tmp_path):
    skeleton = _QUADRUPED
    dof_names = _QUADRUPED_DOF_NAMES
    n_frames = 3
    frame_rate = 10.0
    duration_s = n_frames / frame_rate

    rest_rad = dof_rest_angles_rad(skeleton, dof_names)
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    rest_positions = effector_positions(skeleton, dof_names, rest_rad, feet)

    root_pose = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (n_frames, 1)).astype(np.float32)
    end_effectors = [
        EndEffectorTrack(name=bone, target=np.tile(pos, (n_frames, 1)).astype(np.float32), frame="root_relative")
        for bone, pos in zip(feet, rest_positions)
    ]
    contact = [
        ContactTrack(name="FL_foot", contact=np.ones(n_frames, dtype=bool)),
        ContactTrack(name="RL_foot", contact=np.ones(n_frames, dtype=bool)),
    ]

    header = ClipHeader(
        clip=ClipMeta(name="standing", format_version=1, frame_rate=frame_rate, duration_s=duration_s, loop="cyclic"),
        provenance=Provenance(source_type="create"),
        skeleton=SkeletonBinding(target=_QUADRUPED_PATH.resolve(), morphology_class="quadruped"),
        channels=ChannelsBlock(
            root_pose="authored",
            joint_angles="derived",
            end_effector_targets={"FL_foot": "derived", "RL_foot": "authored"},
        ),
    )
    tracks = TracksData(
        root_pose=root_pose,
        joint_angles=np.zeros((n_frames, len(dof_names)), dtype=np.float32),
        joint_angles_names=dof_names,
        contact=contact,
        phase_variable=np.linspace(0.0, 1.0, n_frames, dtype=np.float32),
        end_effectors=end_effectors,
    )

    clip_dir = tmp_path / "standing"
    save_clip(clip_dir, Clip(header=header, tracks=tracks), skeleton=skeleton)
    opened = open_clip(clip_dir, skeleton=skeleton)

    retargeted = retarget_clip(opened, skeleton)

    assert retargeted.header.channels.joint_angles == "derived"
    assert retargeted.header.validation.status == "ok"
    assert retargeted.tracks.joint_angles.shape == (n_frames, len(dof_names))
    assert np.allclose(retargeted.tracks.joint_angles[0], rest_rad, atol=0.05)

    # authored curve is solved-against but left untouched in the output
    rl_original = next(ee for ee in opened.tracks.end_effectors if ee.name == "RL_foot")
    rl_retargeted = next(ee for ee in retargeted.tracks.end_effectors if ee.name == "RL_foot")
    np.testing.assert_array_equal(rl_retargeted.target, rl_original.target)

    # derived curve reflects the contact lock (constant across the all-True contact run)
    fl_retargeted = next(ee for ee in retargeted.tracks.end_effectors if ee.name == "FL_foot")
    assert np.allclose(fl_retargeted.target, fl_retargeted.target[0])

    # solved output round-trips through tracks.h5 cleanly
    save_clip(tmp_path / "standing_retargeted", retargeted, skeleton=skeleton)


def test_world_frame_end_effector_converted_correctly():
    skeleton = _QUADRUPED
    dof_names = _QUADRUPED_DOF_NAMES
    n_frames = 2
    frame_rate = 10.0
    duration_s = n_frames / frame_rate

    rest_rad = dof_rest_angles_rad(skeleton, dof_names)
    fl_local = effector_positions(skeleton, dof_names, rest_rad, ["FL_foot"])[0]

    root_translation = np.array([1.0, 2.0, 0.5])
    root_pose = np.tile(
        np.array([*root_translation, 1.0, 0.0, 0.0, 0.0]), (n_frames, 1)
    ).astype(np.float32)
    fl_world = fl_local + root_translation  # identity rotation, so world = local + translation

    end_effectors = [EndEffectorTrack(name="FL_foot", target=np.tile(fl_world, (n_frames, 1)).astype(np.float32), frame="world")]
    contact = [ContactTrack(name="FL_foot", contact=np.zeros(n_frames, dtype=bool))]

    header = ClipHeader(
        clip=ClipMeta(name="reach", format_version=1, frame_rate=frame_rate, duration_s=duration_s, loop="one_shot"),
        provenance=Provenance(source_type="create"),
        skeleton=SkeletonBinding(target=_QUADRUPED_PATH.resolve(), morphology_class="quadruped"),
        channels=ChannelsBlock(root_pose="authored", joint_angles="derived"),
    )
    tracks = TracksData(
        root_pose=root_pose,
        joint_angles=np.zeros((n_frames, len(dof_names)), dtype=np.float32),
        joint_angles_names=dof_names,
        contact=contact,
        phase_variable=np.zeros(n_frames, dtype=np.float32),
        end_effectors=end_effectors,
    )

    retargeted = retarget_clip(Clip(header=header, tracks=tracks), skeleton)

    solved_theta = retargeted.tracks.joint_angles[0]
    solved_local_pos = effector_positions(skeleton, dof_names, solved_theta, ["FL_foot"])[0]
    solved_world_pos = solved_local_pos + root_translation
    assert np.allclose(solved_world_pos, fl_world, atol=1e-3)

    fl_retargeted = next(ee for ee in retargeted.tracks.end_effectors if ee.name == "FL_foot")
    assert fl_retargeted.frame == "world"
