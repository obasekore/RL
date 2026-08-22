from pathlib import Path

import numpy as np

from coppelia_rl.retargeting.fk import (
    dof_bounds_rad,
    dof_max_velocity_rad_per_s,
    dof_rest_angles_rad,
    effector_positions,
    end_effector_position,
    forward_kinematics,
    root_relative_to_world,
    world_to_root_relative,
)
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml

_SKELETONS_DIR = Path(__file__).resolve().parents[1] / "skeletons"
_QUADRUPED = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
_QUADRUPED_DOF_NAMES = ordered_dof_names(_QUADRUPED)
_BIPED = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")


def test_root_transform_is_identity():
    transforms = forward_kinematics(_QUADRUPED, {})
    assert np.allclose(transforms["torso"], np.eye(4))


def test_straight_chain_position_at_zero_angles():
    # FL_upper -> FL_lower -> FL_foot, all offsets along -z once past FL_upper's
    # own [0.20,0.10,-0.05] offset from torso; at theta=0 no rotation is applied,
    # so world position is just the cumulative sum of offsets (+ contact_geometry.offset).
    transforms = forward_kinematics(_QUADRUPED, {})
    expected = np.array([0.20, 0.10, -0.05 - 0.18 - 0.18]) + np.array([0.0, 0.0, -0.01])
    pos = end_effector_position(_QUADRUPED, transforms, "FL_foot")
    assert np.allclose(pos, expected)


def test_end_effector_uses_contact_geometry_offset_when_present():
    transforms = forward_kinematics(_QUADRUPED, {})
    bone_origin = transforms["FL_foot"][:3, 3]
    pos = end_effector_position(_QUADRUPED, transforms, "FL_foot")
    assert not np.allclose(pos, bone_origin)  # contact_geometry.offset shifts it
    assert np.allclose(pos, bone_origin + np.array([0.0, 0.0, -0.01]))


def test_end_effector_falls_back_to_bone_origin_when_no_contact_geometry():
    transforms = forward_kinematics(_BIPED, {})
    bone_origin = transforms["L_hand"][:3, 3]
    pos = end_effector_position(_BIPED, transforms, "L_hand")
    assert np.allclose(pos, bone_origin)


def test_roll_rotates_about_forward_axis():
    # A bone whose joint has a single 'roll' axis, with forward_axis=x: rotating it
    # should not move a point offset purely along x (rotation about x leaves the
    # x-axis itself fixed), and forward_kinematics should honor a nonzero angle.
    transforms_zero = forward_kinematics(_QUADRUPED, {})
    transforms_rolled = forward_kinematics(_QUADRUPED, {"FL_hip_roll": np.radians(30)})
    pos_zero = end_effector_position(_QUADRUPED, transforms_zero, "FL_foot")
    pos_rolled = end_effector_position(_QUADRUPED, transforms_rolled, "FL_foot")
    assert not np.allclose(pos_zero, pos_rolled)


def test_dof_bounds_rad_matches_hand_computed_conversion():
    bounds = dof_bounds_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    fl_hip_pitch_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    assert np.allclose(bounds[fl_hip_pitch_idx], (np.radians(-45), np.radians(90)))
    fl_knee_idx = _QUADRUPED_DOF_NAMES.index("FL_knee")
    assert np.allclose(bounds[fl_knee_idx], (np.radians(-10), np.radians(150)))


def test_dof_rest_angles_rad_matches_hand_computed_conversion():
    rest = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    fl_hip_pitch_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    assert np.isclose(rest[fl_hip_pitch_idx], np.radians(20))
    fl_hip_roll_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_roll")
    assert np.isclose(rest[fl_hip_roll_idx], 0.0)


def test_dof_max_velocity_unconverted_and_inf_when_unset():
    max_vel = dof_max_velocity_rad_per_s(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    fl_hip_pitch_idx = _QUADRUPED_DOF_NAMES.index("FL_hip_pitch")
    assert max_vel[fl_hip_pitch_idx] == 6.0  # rad/s, no conversion
    fl_knee_idx = _QUADRUPED_DOF_NAMES.index("FL_knee")
    assert max_vel[fl_knee_idx] == 8.0

    # synthetic joint with no max_velocity declared -> inf
    from coppelia_rl.skeleton.schema import BoneSpec, JointSpec, SkeletonSpec

    minimal = SkeletonSpec(
        name="t",
        morphology_class="custom",
        up_axis="z",
        forward_axis="x",
        root_bone="root",
        bones=[
            BoneSpec(name="root", parent=None, is_root=True),
            BoneSpec(
                name="eff",
                parent="root",
                joint=JointSpec(name="eff_joint", type="revolute_1dof", axes=["pitch"], limits={"pitch": (-10, 10)}),
                is_end_effector=True,
            ),
        ],
        end_effectors=["eff"],
    )
    dof_names = ordered_dof_names(minimal)
    assert dof_max_velocity_rad_per_s(minimal, dof_names)[0] == np.inf


def test_world_root_relative_round_trip():
    rng = np.random.default_rng(0)
    n = 5
    positions = rng.random((n, 3))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    root_pose = np.concatenate([positions, quats], axis=1)

    world_curve = rng.random((n, 3))
    local = world_to_root_relative(world_curve, root_pose)
    back = root_relative_to_world(local, root_pose)
    assert np.allclose(back, world_curve, atol=1e-8)


def test_effector_positions_matches_end_effector_position():
    theta = dof_rest_angles_rad(_QUADRUPED, _QUADRUPED_DOF_NAMES)
    bone_names = ["FL_foot", "RR_foot"]
    batch = effector_positions(_QUADRUPED, _QUADRUPED_DOF_NAMES, theta, bone_names)

    transforms = forward_kinematics(_QUADRUPED, dict(zip(_QUADRUPED_DOF_NAMES, theta)))
    expected = np.array([end_effector_position(_QUADRUPED, transforms, name) for name in bone_names])
    assert np.allclose(batch, expected)
