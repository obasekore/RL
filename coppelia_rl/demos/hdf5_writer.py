"""Episode-indexed HDF5 storage for recorded demonstrations.

Kept separate from recorder.py so the on-disk layout is independently
testable/swappable without touching DemoRecorder's control flow.

Layout:
    demos.h5
    - attrs: env_name, format_version, created_at, num_episodes
    - episode_0/
        - attrs: success (bool), length (int)
        - observations/<key>   # one dataset per obs key, stacked (T+1, *shape)
        - actions               # (T, *action_shape)
        - rewards                # (T,)
        - terminated             # (T,) bool
        - truncated               # (T,) bool
    - episode_1/
        - ...

`observations` has one more entry than `actions`/`rewards` - it holds every
observation seen (the reset() observation plus the observation after each
step), so `(obs[t], action[t], reward[t], obs[t+1])` transitions can be
reconstructed directly, the layout most imitation-learning codebases expect.
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

_FORMAT_VERSION = 1


class EpisodeBuffer:
    """Accumulates one episode's transitions in memory before being flushed."""

    def __init__(self, initial_obs: dict):
        self._obs_keys = list(initial_obs.keys())
        self.observations: dict[str, list] = {k: [initial_obs[k]] for k in self._obs_keys}
        self.actions: list = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def append(self, action, obs, reward: float, terminated: bool, truncated: bool) -> None:
        self.actions.append(action)
        for key in self._obs_keys:
            self.observations[key].append(obs[key])
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def __len__(self) -> int:
        return len(self.actions)


def open_episode_file(path: str | Path, env_name: str) -> h5py.File:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    h5file = h5py.File(path, "a")
    if is_new:
        h5file.attrs["env_name"] = env_name
        h5file.attrs["format_version"] = _FORMAT_VERSION
        h5file.attrs["created_at"] = time.time()
        h5file.attrs["num_episodes"] = 0
    return h5file


def next_episode_index(h5file: h5py.File) -> int:
    return int(h5file.attrs.get("num_episodes", 0))


def write_episode(h5file: h5py.File, episode_index: int, buffer: EpisodeBuffer, success: bool) -> None:
    if len(buffer) == 0:
        raise ValueError("Cannot write an episode with zero steps")

    group = h5file.create_group(f"episode_{episode_index}")
    group.attrs["success"] = bool(success)
    group.attrs["length"] = len(buffer)

    obs_group = group.create_group("observations")
    for key, values in buffer.observations.items():
        obs_group.create_dataset(key, data=np.stack(values))

    group.create_dataset("actions", data=np.stack(buffer.actions))
    group.create_dataset("rewards", data=np.asarray(buffer.rewards, dtype=np.float32))
    group.create_dataset("terminated", data=np.asarray(buffer.terminated, dtype=bool))
    group.create_dataset("truncated", data=np.asarray(buffer.truncated, dtype=bool))

    h5file.attrs["num_episodes"] = episode_index + 1
    h5file.flush()
