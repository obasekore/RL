from pathlib import Path

import pytest

from coppelia_rl.env_schema.parser import EnvXmlValidationError, parse_env_xml

_TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"

_MINIMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rl_env name="{name}" step_dt="0.05" {scene_attr}>
  <observations><joint_position ref="j1"/></observations>
  <actions type="{action_type}"><joint_velocity ref="j1" range="[-1,1]"/></actions>
  <reward><term type="distance" from="a" to="b" weight="-1.0"/></reward>
  <termination><condition type="max_steps" value="10"/></termination>
</rl_env>
"""


def test_reach_xml_parses():
    spec = parse_env_xml(_TASKS_DIR / "reach.xml")
    assert spec.name == "reach"
    assert spec.actions.action_type == "continuous"
    assert len(spec.actions.entries) == 6
    assert len(spec.observations) == 14
    assert spec.reward_terms[0].kind == "distance"
    assert spec.termination_conditions[0].kind == "max_steps"
    assert spec.scene_path.name == "reach.ttt"
    assert spec.scene_path.is_absolute()


def test_pick_and_place_xml_parses():
    spec = parse_env_xml(_TASKS_DIR / "pick_and_place.xml")
    assert spec.name == "pick_and_place"
    camera_obs = [o for o in spec.observations if o.kind == "camera"]
    assert len(camera_obs) == 1
    assert camera_obs[0].resolution == (128, 128)
    assert camera_obs[0].channels == "rgb"
    assert {t.kind for t in spec.reward_terms} == {"distance", "signal_event"}
    assert {c.kind for c in spec.termination_conditions} == {"signal", "max_steps"}


def test_mobile_nav_xml_parses():
    spec = parse_env_xml(_TASKS_DIR / "mobile_nav.xml")
    assert spec.name == "mobile_nav"
    assert len(spec.actions.entries) == 2
    custom_conditions = [c for c in spec.termination_conditions if c.kind == "custom"]
    assert len(custom_conditions) == 1
    assert custom_conditions[0].callable == "tasks.mobile_nav_task:reached_goal"


def test_scene_path_resolved_relative_to_xml_file():
    spec = parse_env_xml(_TASKS_DIR / "reach.xml")
    assert spec.scene_path == (_TASKS_DIR / ".." / "scenes" / "reach.ttt").resolve()


def test_range_attribute_parsed_to_float_tuple():
    spec = parse_env_xml(_TASKS_DIR / "mobile_nav.xml")
    assert spec.actions.entries[0].value_range == (-5.0, 5.0)


def test_missing_required_attribute_fails_schema_validation(tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(_MINIMAL_XML.format(name="bad", scene_attr="", action_type="continuous"))
    with pytest.raises(EnvXmlValidationError):
        parse_env_xml(xml_path)


def test_invalid_enum_value_fails_schema_validation(tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(
        _MINIMAL_XML.format(name="bad", scene_attr='scene="x.ttt"', action_type="not_a_real_type")
    )
    with pytest.raises(EnvXmlValidationError):
        parse_env_xml(xml_path)


def test_continuous_element_inside_discrete_actions_rejected(tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<rl_env name="bad" step_dt="0.05" scene="x.ttt">
  <observations><joint_position ref="j1"/></observations>
  <actions type="discrete"><joint_velocity ref="j1" range="[-1,1]"/></actions>
  <reward><term type="distance" from="a" to="b" weight="-1.0"/></reward>
  <termination><condition type="max_steps" value="10"/></termination>
</rl_env>
"""
    )
    with pytest.raises(EnvXmlValidationError):
        parse_env_xml(xml_path)


# -- motion_imitation -----------------------------------------------------------

_MOTION_IMITATION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rl_env name="quad_walk" step_dt="0.05" scene="x.ttt">
  <observations><joint_position ref="FL_hip_pitch"/></observations>
  <actions type="continuous"><joint_position ref="FL_hip_pitch" range="[-1,1]"/></actions>
  <reward>
    <term type="pose_tracking" weight="0.65"/>
    <term type="velocity_tracking" weight="0.1"/>
    <term type="end_effector_tracking" weight="0.15"/>
    <term type="contact_matching" weight="0.1"/>
  </reward>
  <termination>
    <condition type="fall_detection"/>
    <condition type="max_steps" value="500"/>
  </termination>
  <motion_imitation clip_dir="clips/walk_cycle_01"{rsi_attr}/>
</rl_env>
"""


def test_motion_imitation_element_parses_clip_dir_and_rsi(tmp_path):
    xml_path = tmp_path / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=' rsi="true"'))
    spec = parse_env_xml(xml_path)
    assert spec.motion_imitation is not None
    assert spec.motion_imitation.clip_dir == (tmp_path / "clips" / "walk_cycle_01").resolve()
    assert spec.motion_imitation.rsi is True


def test_motion_imitation_clip_dir_resolved_relative_to_xml_file(tmp_path):
    nested = tmp_path / "envs"
    nested.mkdir()
    xml_path = nested / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=""))
    spec = parse_env_xml(xml_path)
    assert spec.motion_imitation.clip_dir == (nested / "clips" / "walk_cycle_01").resolve()


def test_motion_imitation_rsi_defaults_to_true_when_omitted(tmp_path):
    xml_path = tmp_path / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=""))
    spec = parse_env_xml(xml_path)
    assert spec.motion_imitation.rsi is True


def test_motion_imitation_rsi_false_parses(tmp_path):
    xml_path = tmp_path / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=' rsi="false"'))
    spec = parse_env_xml(xml_path)
    assert spec.motion_imitation.rsi is False


def test_no_motion_imitation_block_parses_as_none(tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(_MINIMAL_XML.format(name="t", scene_attr='scene="x.ttt"', action_type="continuous"))
    spec = parse_env_xml(xml_path)
    assert spec.motion_imitation is None


def test_pose_velocity_ee_contact_reward_term_kinds_parse_with_only_type_and_weight(tmp_path):
    xml_path = tmp_path / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=""))
    spec = parse_env_xml(xml_path)
    kinds = {t.kind: t.weight for t in spec.reward_terms}
    assert kinds == {
        "pose_tracking": 0.65,
        "velocity_tracking": 0.1,
        "end_effector_tracking": 0.15,
        "contact_matching": 0.1,
    }


def test_fall_detection_termination_condition_parses(tmp_path):
    xml_path = tmp_path / "quad_walk.xml"
    xml_path.write_text(_MOTION_IMITATION_XML.format(rsi_attr=""))
    spec = parse_env_xml(xml_path)
    fall_conditions = [c for c in spec.termination_conditions if c.kind == "fall_detection"]
    assert len(fall_conditions) == 1


def test_duplicate_observation_key_raises_clear_error(tmp_path):
    xml_path = tmp_path / "dup.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<rl_env name="dup" step_dt="0.05" scene="x.ttt">
  <observations>
    <joint_position ref="j1" name="pos"/>
    <joint_velocity ref="j1" name="pos"/>
  </observations>
  <actions type="continuous"><joint_velocity ref="j1" range="[-1,1]"/></actions>
  <reward><term type="distance" from="a" to="b" weight="-1.0"/></reward>
  <termination><condition type="max_steps" value="10"/></termination>
</rl_env>
"""
    )
    with pytest.raises(EnvXmlValidationError):
        parse_env_xml(xml_path)
