from pathlib import Path

from coppelia_rl.env_schema.parser import parse_env_xml
from coppelia_rl.env_schema.serializer import save_spec_xml, spec_to_xml
from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    CameraRandomizationSpec,
    DomainRandomizationSpec,
    EnvSpec,
    MotionImitationSpec,
    ObservationSpec,
    RangeRandomizationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
    TextureRandomizationSpec,
)

_TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _roundtrip(spec: EnvSpec, tmp_path: Path) -> EnvSpec:
    xml_path = tmp_path / "roundtrip.xml"
    save_spec_xml(spec, xml_path)
    return parse_env_xml(xml_path)


def test_roundtrip_reach_xml(tmp_path):
    spec = parse_env_xml(_TASKS_DIR / "reach.xml")
    assert _roundtrip(spec, tmp_path) == spec


def test_roundtrip_pick_and_place_xml(tmp_path):
    spec = parse_env_xml(_TASKS_DIR / "pick_and_place.xml")
    assert _roundtrip(spec, tmp_path) == spec


def test_roundtrip_mobile_nav_xml(tmp_path):
    spec = parse_env_xml(_TASKS_DIR / "mobile_nav.xml")
    assert _roundtrip(spec, tmp_path) == spec


def test_roundtrip_kitchen_sink_spec(tmp_path):
    """Exercises every observation/action/reward/termination/domain-randomization kind,
    including custom callables and discrete actions, which none of the 3 shipped
    tasks use together."""
    spec = EnvSpec(
        name="kitchen_sink",
        step_dt=0.05,
        scene_path=(tmp_path / "scene.ttt").resolve(),
        observations=[
            ObservationSpec(kind="joint_position", key="joint_position_j1", ref="j1"),
            ObservationSpec(kind="joint_velocity", key="my_vel", ref="j1"),
            ObservationSpec(kind="object_pose", key="object_pose_target", ref="target", frame="world"),
            ObservationSpec(kind="object_position", key="object_position_target", ref="target", frame="base"),
            ObservationSpec(kind="camera", key="cam", ref="cam1", resolution=(64, 48), channels="depth"),
            ObservationSpec(kind="custom", key="custom_obs", callable="pkg.mod:my_obs"),
        ],
        actions=ActionGroupSpec(
            action_type="discrete",
            entries=[
                ActionEntrySpec(kind="event", key="open", signal="grip_open"),
                ActionEntrySpec(kind="custom", key="custom_action", callable="pkg.mod:my_action"),
            ],
        ),
        reward_terms=[
            RewardTermSpec(kind="distance", weight=-1.0, from_ref="a", to_ref="b"),
            RewardTermSpec(kind="collision", weight=5.0, object1="s1", object2="s2"),
            RewardTermSpec(kind="signal_event", weight=10.0, signal="grasp"),
            RewardTermSpec(kind="custom", weight=2.0, callable="pkg.mod:my_reward"),
        ],
        termination_conditions=[
            TerminationConditionSpec(kind="signal", name="task_done"),
            TerminationConditionSpec(kind="max_steps", value=500),
            TerminationConditionSpec(kind="custom", callable="pkg.mod:my_termination"),
        ],
        domain_randomization=DomainRandomizationSpec(
            textures=[TextureRandomizationSpec(ref="table", pool="textures/wood_*")],
            cameras=[CameraRandomizationSpec(ref="cam1", jitter_pos=0.02, jitter_rot_deg=5.0)],
            masses=[RangeRandomizationSpec(ref="cube", value_range=(0.08, 0.15))],
            frictions=[RangeRandomizationSpec(ref="table", value_range=(0.4, 1.0))],
            action_delay_steps=(0, 3),
            observation_noise_std=0.01,
            resample_on="n_steps",
            resample_every_n_steps=7,
        ),
    )

    assert _roundtrip(spec, tmp_path) == spec


def _motion_imitation_spec(tmp_path: Path, *, rsi: bool) -> EnvSpec:
    return EnvSpec(
        name="quad_walk",
        step_dt=0.05,
        scene_path=(tmp_path / "scene.ttt").resolve(),
        observations=[ObservationSpec(kind="joint_position", key="joint_position_FL_hip_pitch", ref="FL_hip_pitch")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[
                ActionEntrySpec(kind="joint_position", key="FL_hip_pitch", ref="FL_hip_pitch", value_range=(-1.0, 1.0))
            ],
        ),
        reward_terms=[
            RewardTermSpec(kind="pose_tracking", weight=0.65),
            RewardTermSpec(kind="velocity_tracking", weight=0.1),
            RewardTermSpec(kind="end_effector_tracking", weight=0.15),
            RewardTermSpec(kind="contact_matching", weight=0.1),
        ],
        termination_conditions=[
            TerminationConditionSpec(kind="fall_detection"),
            TerminationConditionSpec(kind="max_steps", value=500),
        ],
        motion_imitation=MotionImitationSpec(clip_dir=(tmp_path / "clips" / "walk_cycle_01").resolve(), rsi=rsi),
    )


def test_roundtrip_motion_imitation_spec(tmp_path):
    spec = _motion_imitation_spec(tmp_path, rsi=True)
    assert _roundtrip(spec, tmp_path) == spec


def test_roundtrip_motion_imitation_rsi_false(tmp_path):
    spec = _motion_imitation_spec(tmp_path, rsi=False)
    assert _roundtrip(spec, tmp_path) == spec


def test_spec_to_xml_uses_default_key_convention_without_name_attribute(tmp_path):
    spec = EnvSpec(
        name="t",
        step_dt=0.05,
        scene_path=(tmp_path / "s.ttt").resolve(),
        observations=[ObservationSpec(kind="joint_position", key="joint_position_j1", ref="j1")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="j1", ref="j1", value_range=(-1.0, 1.0))],
        ),
        reward_terms=[RewardTermSpec(kind="distance", weight=-1.0, from_ref="a", to_ref="b")],
        termination_conditions=[TerminationConditionSpec(kind="max_steps", value=10)],
    )
    xml = spec_to_xml(spec)
    assert 'name="joint_position_j1"' not in xml
    assert '<joint_position ref="j1"/>' in xml


def test_spec_to_xml_preserves_explicit_name_override(tmp_path):
    spec = EnvSpec(
        name="t",
        step_dt=0.05,
        scene_path=(tmp_path / "s.ttt").resolve(),
        observations=[ObservationSpec(kind="joint_position", key="my_custom_key", ref="j1")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="j1", ref="j1", value_range=(-1.0, 1.0))],
        ),
        reward_terms=[RewardTermSpec(kind="distance", weight=-1.0, from_ref="a", to_ref="b")],
        termination_conditions=[TerminationConditionSpec(kind="max_steps", value=10)],
    )
    xml = spec_to_xml(spec)
    assert 'name="my_custom_key"' in xml

    reparsed = _roundtrip(spec, tmp_path)
    assert reparsed.observations[0].key == "my_custom_key"
