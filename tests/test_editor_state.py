from pathlib import Path

import pytest

from coppelia_rl.env_schema.editor_state import EnvBuilderState
from coppelia_rl.env_schema.parser import parse_env_xml
from coppelia_rl.sim_interface.client import SimClient

_TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _client(fake_sim):
    return SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)


# -- observations -------------------------------------------------------------


def test_add_observation_derives_default_key():
    state = EnvBuilderState()
    obs = state.add_observation("joint_position", ref="j1")
    assert obs.key == "joint_position_j1"
    assert state.observations == [obs]


def test_add_observation_with_explicit_name():
    state = EnvBuilderState()
    obs = state.add_observation("joint_velocity", ref="j1", name="my_vel")
    assert obs.key == "my_vel"


def test_add_observation_rejects_duplicate_key():
    state = EnvBuilderState()
    state.add_observation("joint_position", ref="j1")
    with pytest.raises(ValueError):
        state.add_observation("joint_position", ref="j1")


def test_add_custom_observation_requires_name():
    state = EnvBuilderState()
    with pytest.raises(ValueError):
        state.add_observation("custom", callable="pkg:fn")


def test_remove_observation():
    state = EnvBuilderState()
    state.add_observation("joint_position", ref="j1")
    state.add_observation("joint_position", ref="j2")
    state.remove_observation(0)
    assert [o.ref for o in state.observations] == ["j2"]


# -- actions --------------------------------------------------------------------


def test_add_continuous_action_entry():
    state = EnvBuilderState()
    entry = state.add_action_entry("joint_velocity", ref="j1", value_range=(-1.0, 1.0))
    assert entry.key == "j1"
    assert state.actions.entries == [entry]


def test_add_action_entry_wrong_kind_for_type_rejected():
    state = EnvBuilderState()
    with pytest.raises(ValueError):
        state.add_action_entry("event", name="open", signal="s")


def test_set_action_type_clears_existing_entries():
    state = EnvBuilderState()
    state.add_action_entry("joint_velocity", ref="j1", value_range=(-1.0, 1.0))
    state.set_action_type("discrete")
    assert state.actions.entries == []
    entry = state.add_action_entry("event", name="open", signal="s")
    assert entry in state.actions.entries


def test_remove_action_entry():
    state = EnvBuilderState()
    state.add_action_entry("joint_velocity", ref="j1", value_range=(-1.0, 1.0))
    state.add_action_entry("joint_velocity", ref="j2", value_range=(-1.0, 1.0))
    state.remove_action_entry(0)
    assert [e.ref for e in state.actions.entries] == ["j2"]


# -- reward / termination ---------------------------------------------------------


def test_add_reward_term_validates_required_fields():
    state = EnvBuilderState()
    with pytest.raises(ValueError):
        state.add_reward_term("distance", weight=-1.0)
    term = state.add_reward_term("distance", weight=-1.0, from_ref="a", to_ref="b")
    assert state.reward_terms == [term]


def test_add_termination_condition_validates_required_fields():
    state = EnvBuilderState()
    with pytest.raises(ValueError):
        state.add_termination_condition("max_steps")
    cond = state.add_termination_condition("max_steps", value=200)
    assert state.termination_conditions == [cond]


# -- domain randomization -----------------------------------------------------------


def test_domain_randomization_add_remove():
    state = EnvBuilderState()
    state.add_texture_randomization(ref="table", pool="textures/wood_*")
    state.add_mass_randomization(ref="cube", value_range=(0.08, 0.15))
    state.set_latency(action_delay_steps=(0, 3), observation_noise_std=0.01)
    state.set_resample_on("n_steps")

    assert len(state.domain_randomization.textures) == 1
    assert len(state.domain_randomization.masses) == 1
    assert state.domain_randomization.action_delay_steps == (0, 3)
    assert state.domain_randomization.resample_on == "n_steps"

    state.remove_texture_randomization(0)
    assert state.domain_randomization.textures == []


# -- build_spec / load ---------------------------------------------------------------


def _minimal_state(tmp_path) -> EnvBuilderState:
    state = EnvBuilderState()
    state.scene_path = tmp_path / "scene.ttt"
    state.add_observation("joint_position", ref="j1")
    state.add_action_entry("joint_velocity", ref="j1", value_range=(-1.0, 1.0))
    state.add_reward_term("distance", weight=-1.0, from_ref="a", to_ref="b")
    state.add_termination_condition("max_steps", value=100)
    return state


def test_build_spec_requires_scene_path(tmp_path):
    state = _minimal_state(tmp_path)
    state.scene_path = None
    with pytest.raises(ValueError):
        state.build_spec()


def test_build_spec_requires_at_least_one_observation(tmp_path):
    state = _minimal_state(tmp_path)
    state.remove_observation(0)
    with pytest.raises(ValueError):
        state.build_spec()


def test_build_spec_omits_domain_randomization_when_empty(tmp_path):
    state = _minimal_state(tmp_path)
    spec = state.build_spec()
    assert spec.domain_randomization is None


def test_build_spec_includes_domain_randomization_when_set(tmp_path):
    state = _minimal_state(tmp_path)
    state.add_texture_randomization(ref="table", pool="textures/wood_*")
    spec = state.build_spec()
    assert spec.domain_randomization is not None
    assert len(spec.domain_randomization.textures) == 1


def test_load_then_build_spec_matches_parsed_reach_xml():
    spec = parse_env_xml(_TASKS_DIR / "reach.xml")
    state = EnvBuilderState()
    state.load(spec)
    assert state.build_spec() == spec


def test_load_then_mutate_does_not_affect_original_spec():
    spec = parse_env_xml(_TASKS_DIR / "reach.xml")
    original_count = len(spec.observations)
    state = EnvBuilderState()
    state.load(spec)
    state.remove_observation(0)
    assert len(spec.observations) == original_count


# -- ref validation -----------------------------------------------------------------


def test_validate_refs_reports_missing_refs(fake_sim):
    j1 = fake_sim._new_handle("joint")
    fake_sim.setObjectAlias(j1, "j1")

    state = EnvBuilderState()
    state.add_observation("joint_position", ref="j1")
    state.add_observation("joint_position", ref="does_not_exist")
    state.add_action_entry("joint_velocity", ref="j1", value_range=(-1.0, 1.0))
    state.add_reward_term("distance", weight=-1.0, from_ref="j1", to_ref="also_missing")
    state.add_termination_condition("max_steps", value=100)

    missing = state.validate_refs(_client(fake_sim))

    assert set(missing) == {"does_not_exist", "also_missing"}


def test_validate_refs_empty_when_everything_resolves(fake_sim):
    a = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(a, "a")
    b = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(b, "b")

    state = EnvBuilderState()
    state.add_observation("object_position", ref="a")
    state.add_action_entry("joint_velocity", ref="a", value_range=(-1.0, 1.0))
    state.add_reward_term("distance", weight=-1.0, from_ref="a", to_ref="b")
    state.add_termination_condition("max_steps", value=100)

    assert state.validate_refs(_client(fake_sim)) == []


def test_validate_refs_ignores_custom_and_signal_refs(fake_sim):
    state = EnvBuilderState()
    state.add_observation("custom", name="custom_obs", callable="pkg.mod:fn")
    state.set_action_type("discrete")
    state.add_action_entry("event", name="open", signal="a_signal_name")
    state.add_reward_term("signal_event", weight=10.0, signal="another_signal")
    state.add_termination_condition("signal", name="done_signal")

    assert state.validate_refs(_client(fake_sim)) == []
