from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from coppelia_rl.env_schema.generic_env import XmlDefinedEnv  # noqa: E402
from coppelia_rl.env_schema.spec import (  # noqa: E402
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    ObservationSpec,
    TerminationConditionSpec,
)
from coppelia_rl.sim_interface.client import SimClient  # noqa: E402
from coppelia_rl.training.vec_env import CoppeliaSimVecEnv  # noqa: E402


def _make_env(fake_sim_factory, max_steps=100):
    fake_sim = fake_sim_factory()
    joint = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
    fake_sim.setObjectAlias(joint, "J1")
    client = SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)
    spec = EnvSpec(
        name="t",
        step_dt=0.05,
        scene_path=Path("unused.ttt"),
        observations=[ObservationSpec(kind="joint_position", key="j1_pos", ref="J1")],
        actions=ActionGroupSpec(
            action_type="continuous",
            entries=[ActionEntrySpec(kind="joint_velocity", key="J1", ref="J1", value_range=(-1.0, 1.0))],
        ),
        reward_terms=[],
        termination_conditions=[TerminationConditionSpec(kind="max_steps", value=max_steps)],
    )
    return XmlDefinedEnv(spec, client), fake_sim, joint


def _make_vec_env(fake_sim_factory, num_envs, max_steps=100):
    envs, fake_sims, joints = [], [], []
    for _ in range(num_envs):
        env, fake_sim, joint = _make_env(fake_sim_factory, max_steps=max_steps)
        envs.append(env)
        fake_sims.append(fake_sim)
        joints.append(joint)
    return CoppeliaSimVecEnv(envs), fake_sims, joints


def test_reset_returns_stacked_obs(fake_sim_factory):
    vec, _, _ = _make_vec_env(fake_sim_factory, num_envs=3)
    obs = vec.reset()
    assert obs["j1_pos"].shape == (3, 1)
    vec.close()


def test_step_stacks_rewards_and_applies_actions(fake_sim_factory):
    vec, fake_sims, joints = _make_vec_env(fake_sim_factory, num_envs=3)
    vec.reset()

    actions = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    obs, rewards, dones, infos = vec.step(actions)

    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert len(infos) == 3
    for fake_sim, joint, expected in zip(fake_sims, joints, [0.1, 0.2, 0.3]):
        assert fake_sim.objects[joint]["joint_target_velocity"] == pytest.approx(expected)
    vec.close()


def test_auto_resets_on_done_and_sets_terminal_observation(fake_sim_factory):
    vec, _, _ = _make_vec_env(fake_sim_factory, num_envs=2, max_steps=2)
    vec.reset()

    actions = np.zeros((2, 1), dtype=np.float32)
    vec.step(actions)
    obs, rewards, dones, infos = vec.step(actions)

    assert np.all(dones)
    for info in infos:
        assert "terminal_observation" in info

    # A step right after auto-reset should not still be done.
    _, _, dones2, _ = vec.step(actions)
    assert not np.any(dones2)
    vec.close()


def test_get_attr_and_set_attr(fake_sim_factory):
    vec, _, _ = _make_vec_env(fake_sim_factory, num_envs=2)
    vec.reset()

    assert vec.get_attr("step_count") == [0, 0]

    vec.set_attr("step_count", 5, indices=[0])
    assert vec.get_attr("step_count") == [5, 0]
    vec.close()


def test_env_method_calls_underlying_method(fake_sim_factory):
    vec, _, _ = _make_vec_env(fake_sim_factory, num_envs=2)
    vec.reset()

    results = vec.env_method("_get_obs")
    assert len(results) == 2
    assert all("j1_pos" in r for r in results)
    vec.close()
