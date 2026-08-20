import pytest

from coppelia_rl.env_schema.serializer import save_spec_xml
from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    DomainRandomizationSpec,
    EnvSpec,
    ObservationSpec,
    RangeRandomizationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
    TextureRandomizationSpec,
)
from coppelia_rl.randomization.randomizer_state import DomainRandomizerState


def test_add_remove_texture_and_mass():
    state = DomainRandomizerState()

    state.add_texture_randomization(ref="table", pool="textures/wood_*")
    state.add_mass_randomization(ref="cube", value_range=(0.08, 0.15))

    assert len(state.domain_randomization.textures) == 1
    assert len(state.domain_randomization.masses) == 1

    state.remove_texture_randomization(0)
    state.remove_mass_randomization(0)
    assert state.domain_randomization.textures == []
    assert state.domain_randomization.masses == []


def test_add_remove_camera_and_friction():
    state = DomainRandomizerState()

    state.add_camera_randomization(ref="cam1", jitter_pos=0.02, jitter_rot_deg=5.0)
    state.add_friction_randomization(ref="table", value_range=(0.4, 1.0))

    assert len(state.domain_randomization.cameras) == 1
    assert len(state.domain_randomization.frictions) == 1

    state.remove_camera_randomization(0)
    state.remove_friction_randomization(0)
    assert state.domain_randomization.cameras == []
    assert state.domain_randomization.frictions == []


def test_set_latency_and_resample_on():
    state = DomainRandomizerState()

    state.set_latency(action_delay_steps=(0, 3), observation_noise_std=0.01)
    state.set_resample_on("n_steps", every_n_steps=10)

    assert state.domain_randomization.action_delay_steps == (0, 3)
    assert state.domain_randomization.observation_noise_std == 0.01
    assert state.domain_randomization.resample_on == "n_steps"
    assert state.domain_randomization.resample_every_n_steps == 10


def test_set_resample_on_rejects_unknown_value():
    state = DomainRandomizerState()
    with pytest.raises(ValueError):
        state.set_resample_on("whenever")


def test_has_domain_randomization_reflects_content():
    state = DomainRandomizerState()
    assert not state.has_domain_randomization()

    state.add_mass_randomization(ref="cube", value_range=(0.08, 0.15))
    assert state.has_domain_randomization()


def test_load_from_env_xml_pulls_only_the_dr_block(tmp_path):
    spec = EnvSpec(
        name="t",
        step_dt=0.05,
        scene_path=(tmp_path / "scene.ttt").resolve(),
        observations=[ObservationSpec(kind="joint_position", key="joint_position_j1", ref="j1")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="j1", ref="j1", value_range=(-1.0, 1.0))],
        ),
        reward_terms=[RewardTermSpec(kind="distance", weight=-1.0, from_ref="a", to_ref="b")],
        termination_conditions=[TerminationConditionSpec(kind="max_steps", value=10)],
        domain_randomization=DomainRandomizationSpec(
            textures=[TextureRandomizationSpec(ref="table", pool="textures/wood_*")],
            masses=[RangeRandomizationSpec(ref="cube", value_range=(0.08, 0.15))],
        ),
    )
    xml_path = tmp_path / "env.xml"
    save_spec_xml(spec, xml_path)

    state = DomainRandomizerState()
    state.load_from_env_xml(xml_path)

    assert len(state.domain_randomization.textures) == 1
    assert len(state.domain_randomization.masses) == 1


def test_load_none_resets_to_empty():
    state = DomainRandomizerState()
    state.add_mass_randomization(ref="cube", value_range=(0.08, 0.15))

    state.load(None)

    assert not state.has_domain_randomization()
