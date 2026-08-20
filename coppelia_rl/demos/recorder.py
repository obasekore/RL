"""DemoRecorder: records every episode of the wrapped env to HDF5.

A gymnasium.Wrapper, not a bespoke proxy class - that's what makes this
literally "the same env object" as training rather than a parallel tool:
observation_space/action_space and any escape-hatch attributes (e.g.
XmlDefinedEnv.client) delegate straight through via gym.Wrapper's own
__getattr__ fallback, so a training script and a demo-recording script are
both just driving "a gym.Env", differing only in whether it's wrapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from coppelia_rl.demos.hdf5_writer import EpisodeBuffer, next_episode_index, open_episode_file, write_episode

SuccessFn = Callable[[Any, bool, bool, dict], bool]


def _default_success_fn(obs: Any, terminated: bool, truncated: bool, info: dict) -> bool:
    """Terminated-without-truncation reads as "finished for a real reason"
    (e.g. a signal-based task_done condition fired) vs. truncated meaning
    "ran out of time" - a reasonable default, but not universal: a task with
    no signal-based termination (e.g. reach.xml, max_steps only) needs its
    own success_fn since terminated is never True there - `obs` is passed
    through so distance-threshold-style criteria don't need a workaround."""
    return terminated and not truncated


class DemoRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        output_path: str | Path,
        *,
        env_name: str | None = None,
        success_fn: SuccessFn | None = None,
    ):
        super().__init__(env)
        self._h5file = open_episode_file(output_path, env_name or type(env.unwrapped).__name__)
        self._success_fn = success_fn or _default_success_fn
        self._success_override: bool | None = None
        self._buffer: EpisodeBuffer | None = None

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._buffer = EpisodeBuffer(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(action, obs, reward, terminated, truncated)

        if terminated or truncated:
            if self._success_override is not None:
                success = self._success_override
                self._success_override = None
            else:
                success = self._success_fn(obs, terminated, truncated, info)
            write_episode(self._h5file, next_episode_index(self._h5file), self._buffer, success)
            self._buffer = None

        return obs, reward, terminated, truncated, info

    def set_success_override(self, success: bool | None) -> None:
        """Lets an external driver (e.g. a human via the teleop script)
        label the *next* episode boundary explicitly, overriding success_fn
        once. Consumed and reset to None after the episode it applies to."""
        self._success_override = success

    def close(self) -> None:
        self._h5file.close()
        self.env.close()
