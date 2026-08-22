from pathlib import Path

import pytest

from coppelia_rl.motion.clip_yaml_parser import ClipYamlValidationError, parse_clip_yaml
from coppelia_rl.motion.clip_yaml_serializer import save_clip_yaml

_QUADRUPED_PATH = Path(__file__).resolve().parents[1] / "skeletons" / "quadruped_generic.yaml"


def _clip_yaml_text(skeleton_target: str) -> str:
    return f"""
clip:
  name: walk_cycle_01
  format_version: 1
  frame_rate: 50.0
  duration_s: 4.7
  loop: cyclic

provenance:
  source_type: create
  source_ref: null
  authoring_tool: motion_retargeting_studio
  created_at: 2026-08-20T00:00:00Z
  notes: "trot gait, synthesized from root spline, RL foot manually keyed"

skeleton:
  target: {skeleton_target}
  morphology_class: quadruped
  retargeted_from: null

channels:
  root_pose: authored
  joint_angles: derived
  end_effector_targets:
    FL_foot: derived
    FR_foot: derived
    RL_foot: authored
    RR_foot: derived
  contact_state: authored

validation:
  status: warning
  issues:
    - type: joint_velocity_exceeded
      joint: FL_hip
      frame_range: [58, 71]
      severity: warning
      message: "exceeds actuator limit near sharp turn (control point 3)"
"""


def _write_clip_yaml(dir_path: Path, skeleton_target: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "clip.yaml"
    path.write_text(_clip_yaml_text(skeleton_target), encoding="utf-8")
    return path


# -- happy path ---------------------------------------------------------------


def test_full_clip_yaml_parses(tmp_path):
    header = parse_clip_yaml(_write_clip_yaml(tmp_path, _QUADRUPED_PATH.as_posix()))

    assert header.clip.name == "walk_cycle_01"
    assert header.clip.frame_rate == 50.0
    assert header.clip.duration_s == 4.7
    assert header.clip.loop == "cyclic"
    assert header.provenance.source_type == "create"
    assert header.provenance.authoring_tool == "motion_retargeting_studio"
    assert header.skeleton.target == _QUADRUPED_PATH.resolve()
    assert header.skeleton.morphology_class == "quadruped"
    assert header.skeleton.retargeted_from is None
    assert header.channels.root_pose == "authored"
    assert header.channels.joint_angles == "derived"
    assert header.channels.end_effector_targets == {
        "FL_foot": "derived",
        "FR_foot": "derived",
        "RL_foot": "authored",
        "RR_foot": "derived",
    }
    assert header.channels.contact_state == "authored"
    assert header.validation.status == "warning"
    assert header.validation.issues[0].type == "joint_velocity_exceeded"
    assert header.validation.issues[0].joint == "FL_hip"
    assert header.validation.issues[0].frame_range == (58, 71)
    assert header.validation.issues[0].severity == "warning"


def test_created_at_datetime_coercion_normalized_to_string(tmp_path):
    """PyYAML's safe_load implicitly parses an unquoted ISO8601 timestamp like
    `2026-08-20T00:00:00Z` (Doc 2's own clip.yaml example) into a
    datetime.datetime, not a str - parse_clip_yaml must normalize it back."""
    header = parse_clip_yaml(_write_clip_yaml(tmp_path, _QUADRUPED_PATH.as_posix()))
    assert isinstance(header.provenance.created_at, str)
    assert header.provenance.created_at == "2026-08-20T00:00:00Z"


def test_skeleton_target_resolved_relative_to_clip_yaml_file(tmp_path):
    clip_dir = tmp_path / "clips" / "walk_01"
    path = _write_clip_yaml(clip_dir, "../../skeletons/quadruped_generic.yaml")
    header = parse_clip_yaml(path)
    assert header.skeleton.target == (clip_dir / "../../skeletons/quadruped_generic.yaml").resolve()


def test_round_trip(tmp_path):
    header = parse_clip_yaml(_write_clip_yaml(tmp_path / "src", _QUADRUPED_PATH.as_posix()))

    out_path = tmp_path / "roundtrip" / "clip.yaml"
    out_path.parent.mkdir(parents=True)
    save_clip_yaml(header, out_path)
    reparsed = parse_clip_yaml(out_path)

    assert reparsed == header


def test_round_trip_minimal_create_mode(tmp_path):
    text = f"""
clip:
  name: c
  format_version: 1
  frame_rate: 30.0
  duration_s: 1.0
  loop: one_shot

provenance:
  source_type: create

skeleton:
  target: {_QUADRUPED_PATH.as_posix()}
  morphology_class: quadruped

channels:
  root_pose: authored
  joint_angles: derived
"""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    path = src_dir / "clip.yaml"
    path.write_text(text, encoding="utf-8")
    header = parse_clip_yaml(path)

    assert header.provenance.source_ref is None
    assert header.skeleton.retargeted_from is None
    assert header.validation.status == "ok"
    assert header.validation.issues == []

    out_path = tmp_path / "roundtrip" / "clip.yaml"
    out_path.parent.mkdir(parents=True)
    save_clip_yaml(header, out_path)
    reparsed = parse_clip_yaml(out_path)

    assert reparsed == header


# -- validation failures --------------------------------------------------------


def test_bad_loop_raises(tmp_path):
    text = _clip_yaml_text(_QUADRUPED_PATH.as_posix()).replace("loop: cyclic", "loop: bogus")
    path = tmp_path / "clip.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ClipYamlValidationError):
        parse_clip_yaml(path)


def test_bad_channel_value_raises(tmp_path):
    text = _clip_yaml_text(_QUADRUPED_PATH.as_posix()).replace("root_pose: authored", "root_pose: bogus")
    path = tmp_path / "clip.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ClipYamlValidationError):
        parse_clip_yaml(path)


def test_bad_validation_status_raises(tmp_path):
    text = _clip_yaml_text(_QUADRUPED_PATH.as_posix()).replace("status: warning", "status: bogus")
    path = tmp_path / "clip.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ClipYamlValidationError):
        parse_clip_yaml(path)


def test_missing_required_field_raises(tmp_path):
    text = _clip_yaml_text(_QUADRUPED_PATH.as_posix()).replace("  loop: cyclic\n", "")
    path = tmp_path / "clip.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ClipYamlValidationError):
        parse_clip_yaml(path)
