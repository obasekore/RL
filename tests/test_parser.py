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
