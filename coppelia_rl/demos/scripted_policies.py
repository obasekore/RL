"""Example scripted policies for automated demo generation.

Illustrates the "any Python callable policy(obs) -> action" escape hatch -
the same pattern already used for custom observations/actions/rewards/
termination throughout coppelia_rl/env_schema/. `ReachTowardTargetPolicy` is
deliberately a crude heuristic, not a real motion planner: it needs no
Jacobian/inverse-kinematics, trading control quality for being genuinely
"cheap, large-scale" to run. `simIK` is confirmed reachable via the ZMQ
client in this install for whoever wants a real IK-based upgrade later.
"""

from __future__ import annotations

import numpy as np


class ReachTowardTargetPolicy:
    """A stochastic hill-climber over `reach.xml`'s own distance-based
    reward: if the last action reduced tip-to-target distance, keep
    perturbing it in roughly the same direction; otherwise resample a fresh
    random direction."""

    def __init__(
        self,
        action_space,
        *,
        tip_key: str = "object_position_gripper_tip",
        target_key: str = "object_position_ReachTarget",
        perturb_std: float = 0.2,
        rng: np.random.Generator | None = None,
    ):
        self._action_space = action_space
        self._tip_key = tip_key
        self._target_key = target_key
        self._perturb_std = perturb_std
        self._rng = rng if rng is not None else np.random.default_rng()
        self._prev_distance: float | None = None
        self._prev_action: np.ndarray | None = None

    def reset(self) -> None:
        """Call between episodes - clears the hill-climbing state so a new
        episode doesn't start biased by the previous one's last action."""
        self._prev_distance = None
        self._prev_action = None

    def __call__(self, obs: dict) -> np.ndarray:
        distance = self._distance(obs)

        improved = self._prev_action is not None and distance < self._prev_distance
        if improved:
            action = self._prev_action + self._rng.normal(0.0, self._perturb_std, size=self._action_space.shape)
        else:
            action = self._rng.uniform(self._action_space.low, self._action_space.high)

        action = np.clip(action, self._action_space.low, self._action_space.high).astype(self._action_space.dtype)
        self._prev_distance = distance
        self._prev_action = action
        return action

    def success(self, obs: dict, threshold: float = 0.05) -> bool:
        return self._distance(obs) <= threshold

    def _distance(self, obs: dict) -> float:
        return float(np.linalg.norm(obs[self._target_key] - obs[self._tip_key]))
