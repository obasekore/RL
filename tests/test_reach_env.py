import numpy as np

from coppelia_rl.envs.reach_env import ReachEnv
from coppelia_rl.sim_interface.client import SimClient


def make_env(tmp_path, fake_sim, **kwargs):
    client = SimClient(remote_client=None, sim=fake_sim, scene_load_settle_time=0)
    scene_path = tmp_path / "reach.ttt"
    return ReachEnv(client=client, scene_path=scene_path, ur5_model_path="fake/UR5.ttm", **kwargs)


def test_spaces_have_expected_shape(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim)

    assert env.observation_space.shape == (15,)
    assert env.action_space.shape == (6,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_returns_obs_matching_space(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim)

    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert info == {}


def test_step_calls_underlying_sim_step_action_repeat_times(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim, action_repeat=3, max_steps=10)
    env.reset(seed=0)

    before = fake_sim.step_count
    env.step(env.action_space.sample())

    assert fake_sim.step_count - before == 3


def test_episode_truncates_at_max_steps(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim, max_steps=5, success_threshold=-1.0)
    env.reset(seed=0)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        steps += 1

    assert truncated is True
    assert terminated is False
    assert steps == 5


def test_reward_reflects_tip_to_target_distance(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim, max_steps=100, success_threshold=0.01)
    env.reset(seed=0)

    env._scene.tip.set_position([0.0, 0.0, 0.0])
    env._scene.target.set_position([1.0, 0.0, 0.0])
    _, reward_far, terminated_far, _, info_far = env.step(np.zeros(6, dtype=np.float32))

    env._scene.target.set_position([0.0, 0.0, 0.0])
    _, reward_near, terminated_near, _, info_near = env.step(np.zeros(6, dtype=np.float32))

    assert reward_near > reward_far
    assert terminated_far is False
    assert terminated_near is True
    assert info_near["success"] is True


def test_random_policy_episode_completes(tmp_path, fake_sim):
    env = make_env(tmp_path, fake_sim, max_steps=20)
    env.reset(seed=1)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)

    assert steps <= 20
    env.close()
