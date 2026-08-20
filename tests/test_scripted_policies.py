import numpy as np
from gymnasium import spaces

from coppelia_rl.demos.scripted_policies import ReachTowardTargetPolicy

_ACTION_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


def _obs(tip, target):
    return {
        "object_position_gripper_tip": np.array(tip, dtype=np.float32),
        "object_position_ReachTarget": np.array(target, dtype=np.float32),
    }


def test_actions_stay_within_bounds():
    policy = ReachTowardTargetPolicy(_ACTION_SPACE, rng=np.random.default_rng(0))
    obs = _obs([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    for _ in range(20):
        action = policy(obs)
        assert action.shape == (6,)
        assert np.all(action >= _ACTION_SPACE.low)
        assert np.all(action <= _ACTION_SPACE.high)


def test_success_reflects_distance_threshold():
    policy = ReachTowardTargetPolicy(_ACTION_SPACE)

    close_obs = _obs([0.0, 0.0, 0.0], [0.01, 0.0, 0.0])
    far_obs = _obs([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    assert policy.success(close_obs, threshold=0.05) is True
    assert policy.success(far_obs, threshold=0.05) is False


def test_reset_clears_hill_climbing_state():
    policy = ReachTowardTargetPolicy(_ACTION_SPACE, rng=np.random.default_rng(0))
    obs = _obs([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    policy(obs)
    assert policy._prev_action is not None

    policy.reset()

    assert policy._prev_action is None
    assert policy._prev_distance is None


def test_repeats_direction_when_distance_improves():
    """If the last action reduced distance, the next action should be a small
    perturbation of it (not an unrelated fresh random draw)."""
    policy = ReachTowardTargetPolicy(_ACTION_SPACE, perturb_std=0.01, rng=np.random.default_rng(0))
    policy._prev_action = np.full(6, 0.5, dtype=np.float32)
    policy._prev_distance = 1.0

    action = policy(_obs([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]))  # smaller distance than prev

    assert np.allclose(action, 0.5, atol=0.05)
