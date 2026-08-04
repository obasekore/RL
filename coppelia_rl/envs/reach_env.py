"""Hand-authored reach task: drive a UR5's tip to a randomized target position.

No XML env schema yet - this env exists to prove the
Communication Layer's object model is sufficient to build a real RL env on
top of, end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from coppelia_rl.envs.scene_builder import ReachScene, ensure_reach_scene, sample_target_position
from coppelia_rl.sim_interface.client import SimClient

_DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[2] / "scenes" / "reach.ttt"
_TIME_PENALTY = 0.01
_SUCCESS_BONUS = 10.0


class ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 23000,
        scene_path: str | Path = _DEFAULT_SCENE_PATH,
        max_steps: int = 200,
        action_repeat: int = 1,
        success_threshold: float = 0.05,
        max_joint_velocity: float = 1.0,
        client: SimClient | None = None,
        ur5_model_path: str | Path | None = None,
    ):
        super().__init__()
        self._client = client if client is not None else SimClient.connect(host=host, port=port)
        self._max_steps = max_steps
        self._action_repeat = action_repeat
        self._success_threshold = success_threshold
        self._max_joint_velocity = max_joint_velocity
        self._rng = np.random.default_rng()

        self._scene: ReachScene = ensure_reach_scene(self._client, Path(scene_path), ur5_model_path)
        self._num_joints = len(self._scene.joints)
        self._steps = 0

        obs_dim = 2 * self._num_joints + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self._num_joints,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._client.stop_simulation()
        self._scene.target.set_position(sample_target_position(self._rng))
        self._client.set_stepping(True)
        self._client.start_simulation()
        self._steps = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        for joint, a in zip(self._scene.joints, action):
            joint.set_target_velocity(float(a) * self._max_joint_velocity)

        for _ in range(self._action_repeat):
            self._client.step()
        self._steps += 1

        distance = float(np.linalg.norm(self._tip_to_target()))
        success = distance < self._success_threshold
        reward = -distance - _TIME_PENALTY + (_SUCCESS_BONUS if success else 0.0)

        terminated = success
        truncated = self._steps >= self._max_steps
        info = {"distance": distance, "success": success}
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self._client.close()

    def _tip_to_target(self) -> np.ndarray:
        return self._scene.target.get_position() - self._scene.tip.get_position()

    def _get_obs(self) -> np.ndarray:
        positions = np.array([j.get_joint_position() for j in self._scene.joints], dtype=np.float32)
        velocities = np.array([j.get_joint_velocity() for j in self._scene.joints], dtype=np.float32)
        tip_to_target = self._tip_to_target().astype(np.float32)
        return np.concatenate([positions, velocities, tip_to_target])
