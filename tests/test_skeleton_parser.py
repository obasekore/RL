from pathlib import Path

import pytest

from coppelia_rl.skeleton.parser import (
    SkeletonValidationError,
    dof_joint_names,
    ordered_dof_names,
    parse_skeleton_yaml,
)

_SKELETONS_DIR = Path(__file__).resolve().parents[1] / "skeletons"

_QUADRUPED_DOF_NAMES = [
    "FL_hip_pitch", "FL_hip_roll", "FL_knee",
    "FR_hip_pitch", "FR_hip_roll", "FR_knee",
    "RL_hip_pitch", "RL_hip_roll", "RL_knee",
    "RR_hip_pitch", "RR_hip_roll", "RR_knee",
]

_BIPED_DOF_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "spine_joint_pitch", "spine_joint_yaw",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
]

_QUADRUPED_DOF_JOINT_NAMES = [
    "FL_hip", "FL_hip", "FL_knee",
    "FR_hip", "FR_hip", "FR_knee",
    "RL_hip", "RL_hip", "RL_knee",
    "RR_hip", "RR_hip", "RR_knee",
]

_BIPED_DOF_JOINT_NAMES = [
    "L_hip", "L_hip", "L_hip", "L_knee", "L_ankle", "L_ankle",
    "R_hip", "R_hip", "R_hip", "R_knee", "R_ankle", "R_ankle",
    "spine_joint", "spine_joint",
    "L_shoulder", "L_shoulder", "L_shoulder", "L_elbow",
    "R_shoulder", "R_shoulder", "R_shoulder", "R_elbow",
]

_MINIMAL_VALID = """
skeleton:
  name: t
  morphology_class: custom
  up_axis: z
  forward_axis: x

root:
  bone: root_bone

bones:
  - name: root_bone
    parent: null
    is_root: true
  - name: eff
    parent: root_bone
    joint:
      name: eff_joint
      type: revolute_1dof
      axes: [pitch]
      limits:
        pitch: [-10, 10]
      max_velocity: 1.0
    is_end_effector: true
    contact_capable: true
    contact_geometry:
      type: sphere
      radius: 0.01

end_effectors: [eff]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "skeleton.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# -- happy path ---------------------------------------------------------------


def test_quadruped_yaml_parses():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    assert spec.name == "quadruped_generic"
    assert spec.morphology_class == "quadruped"
    assert spec.root_bone == "torso"
    assert len(spec.bones) == 13
    assert spec.end_effectors == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert spec.gait_defaults.limb_groups["front"] == ["FL_foot", "FR_foot"]
    assert spec.gait_defaults.presets["trot"].duty_factor == 0.5
    assert spec.fall_detection.min_root_height == 0.12
    assert spec.fall_detection.max_tilt_deg == 60


def test_biped_yaml_parses():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
    assert spec.name == "biped_generic"
    assert spec.morphology_class == "biped"
    assert spec.root_bone == "pelvis"
    assert len(spec.bones) == 14
    assert spec.end_effectors == ["L_foot", "R_foot", "L_hand", "R_hand"]
    assert spec.gait_defaults.limb_groups["legs"] == ["L_foot", "R_foot"]
    assert spec.gait_defaults.presets["walk"].duty_factor == 0.6
    assert spec.fall_detection.min_root_height == 0.5
    assert spec.fall_detection.max_tilt_deg == 45


def test_bone_offset_parses():
    torso = next(b for b in parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml").bones if b.name == "torso")
    assert torso.offset == (0.0, 0.0, 0.0)  # root default, unset in the fixture

    quadruped = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    fl_upper = next(b for b in quadruped.bones if b.name == "FL_upper")
    assert fl_upper.offset == (0.20, 0.10, -0.05)
    fl_lower = next(b for b in quadruped.bones if b.name == "FL_lower")
    assert fl_lower.offset == (0.0, 0.0, -0.18)


def test_rest_angle_parses():
    quadruped = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    fl_upper = next(b for b in quadruped.bones if b.name == "FL_upper")
    assert fl_upper.joint.rest_angle == {"pitch": 20.0, "roll": 0.0}

    biped = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
    l_hand = next(b for b in biped.bones if b.name == "L_hand")
    assert l_hand.joint.rest_angle == {}  # fixed joint, no rest_angle declared


def test_dof_joint_names_quadruped():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    assert dof_joint_names(spec) == _QUADRUPED_DOF_JOINT_NAMES


def test_dof_joint_names_biped():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
    assert dof_joint_names(spec) == _BIPED_DOF_JOINT_NAMES


def test_joint_axes_length_matches_type():
    quadruped = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    fl_upper = next(b for b in quadruped.bones if b.name == "FL_upper")
    assert fl_upper.joint.type == "revolute_2dof"
    assert fl_upper.joint.axes == ["pitch", "roll"]

    biped = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
    l_upper_leg = next(b for b in biped.bones if b.name == "L_upper_leg")
    assert l_upper_leg.joint.type == "revolute_3dof"
    assert l_upper_leg.joint.axes == ["pitch", "roll", "yaw"]


def test_ordered_dof_names_quadruped():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "quadruped_generic.yaml")
    assert ordered_dof_names(spec) == _QUADRUPED_DOF_NAMES


def test_ordered_dof_names_biped():
    spec = parse_skeleton_yaml(_SKELETONS_DIR / "biped_generic.yaml")
    assert ordered_dof_names(spec) == _BIPED_DOF_NAMES


def test_minimal_valid_parses(tmp_path):
    spec = parse_skeleton_yaml(_write(tmp_path, _MINIMAL_VALID))
    assert spec.root_bone == "root_bone"
    assert ordered_dof_names(spec) == ["eff_joint"]


# -- validation failures --------------------------------------------------------


def test_missing_parent_ref_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("parent: root_bone", "parent: ghost")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_no_root_bone_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("is_root: true", "is_root: false")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_two_root_bones_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "  - name: eff\n    parent: root_bone",
        "  - name: eff\n    parent: root_bone\n    is_root: true",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_root_bone_is_root_mismatch_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("root:\n  bone: root_bone", "root:\n  bone: eff")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_duplicate_bone_name_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "end_effectors: [eff]",
        "  - name: root_bone\n    parent: null\n\nend_effectors: [eff]",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_end_effector_unknown_bone_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("end_effectors: [eff]", "end_effectors: [ghost]")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_end_effector_not_marked_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("is_end_effector: true", "is_end_effector: false")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_axes_type_mismatch_raises(tmp_path):
    bad = _MINIMAL_VALID.replace("axes: [pitch]", "axes: [pitch, roll]")
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_fixed_joint_declaring_limits_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "      type: revolute_1dof\n      axes: [pitch]\n      limits:\n        pitch: [-10, 10]",
        "      type: fixed\n      limits:\n        pitch: [-10, 10]",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_contact_capable_without_geometry_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "    contact_geometry:\n      type: sphere\n      radius: 0.01\n",
        "",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_gait_preset_phase_offset_unknown_bone_raises(tmp_path):
    bad = _MINIMAL_VALID + """
gait_defaults:
  presets:
    p1:
      phase_offset: {ghost: 0.0}
      duty_factor: 0.5
"""
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_gait_limb_group_unknown_bone_raises(tmp_path):
    bad = _MINIMAL_VALID + """
gait_defaults:
  limb_groups:
    g1: [ghost]
"""
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_rest_angle_axis_not_in_axes_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "      limits:\n        pitch: [-10, 10]",
        "      limits:\n        pitch: [-10, 10]\n      rest_angle:\n        roll: 0",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_fixed_joint_declaring_rest_angle_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "      type: revolute_1dof\n      axes: [pitch]\n      limits:\n        pitch: [-10, 10]",
        "      type: fixed\n      rest_angle:\n        pitch: 0",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))


def test_rest_angle_outside_limits_raises(tmp_path):
    bad = _MINIMAL_VALID.replace(
        "      limits:\n        pitch: [-10, 10]",
        "      limits:\n        pitch: [-10, 10]\n      rest_angle:\n        pitch: 20",
    )
    with pytest.raises(SkeletonValidationError):
        parse_skeleton_yaml(_write(tmp_path, bad))
