"""SB3 VecEnv adapter over N independent headless CoppeliaSim instances.

Importing this module requires `stable-baselines3` (the `sb3` optional
extra) - kept in its own module rather than coppelia_rl/training/__init__.py
so training code that doesn't need SB3 never pays that import cost.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info

from coppelia_rl.env_schema import load_env
from coppelia_rl.env_schema.generic_env import XmlDefinedEnv
from coppelia_rl.training import instance_launcher


class CoppeliaSimVecEnv(VecEnv):
    """N `XmlDefinedEnv`s, each backed by its own headless CoppeliaSim
    process, stepped through SB3's VecEnv interface.

    Modeled on SB3's own `DummyVecEnv` (same obs-buffer / auto-reset-on-done
    / `terminal_observation` conventions - SB3's algorithms depend on this
    exact contract), with one deliberate difference: `reset()`/`step_wait()`
    dispatch every sub-env concurrently via a thread pool instead of
    sequentially. `DummyVecEnv` is sequential-by-design for cheap
    same-process envs; here each sub-env talks to a *different* CoppeliaSim
    OS process over its own ZMQ socket, so sequential stepping would incur
    the sum of every instance's step time instead of the max - silently
    throwing away the entire point of running multiple instances.
    """

    actions: np.ndarray

    def __init__(
        self,
        envs: list[XmlDefinedEnv],
        instances: list[instance_launcher.CoppeliaSimInstance] | None = None,
    ):
        if not envs:
            raise ValueError("CoppeliaSimVecEnv needs at least one env")
        self.envs = envs
        self._instances = instances or []
        self._executor = ThreadPoolExecutor(max_workers=len(envs))

        env = envs[0]
        super().__init__(len(envs), env.observation_space, env.action_space)
        self.keys, shapes, dtypes = obs_space_info(env.observation_space)

        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    @classmethod
    def launch(
        cls,
        xml_path: str | Path,
        num_envs: int,
        *,
        base_port: int = 23100,
        host: str = "localhost",
        coppeliasim_root: str | Path | None = None,
        startup_timeout: float = 60.0,
    ) -> "CoppeliaSimVecEnv":
        """Launches `num_envs` headless CoppeliaSim processes on consecutive
        ports starting at `base_port`, each running its own copy of the env
        defined by `xml_path`. On any failure partway through, terminates
        whatever instances were already launched instead of leaking them."""
        instances: list[instance_launcher.CoppeliaSimInstance] = []
        envs: list[XmlDefinedEnv] = []
        try:
            for i in range(num_envs):
                instance, client = instance_launcher.launch_and_connect(
                    base_port + i,
                    host=host,
                    coppeliasim_root=coppeliasim_root,
                    startup_timeout=startup_timeout,
                )
                instances.append(instance)
                envs.append(load_env(xml_path, client=client))
        except Exception:
            for instance in instances:
                instance.terminate()
            raise
        return cls(envs, instances)

    # -- stepping ---------------------------------------------------------------

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        results = list(self._executor.map(lambda i: self.envs[i].step(self.actions[i]), range(self.num_envs)))
        for env_idx, (obs, reward, terminated, truncated, info) in enumerate(results):
            self.buf_rews[env_idx] = reward
            self.buf_infos[env_idx] = info
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self) -> VecEnvObs:
        def reset_one(env_idx: int):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            return self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)

        results = list(self._executor.map(reset_one, range(self.num_envs)))
        for env_idx, (obs, info) in enumerate(results):
            self.reset_infos[env_idx] = info
            self._save_obs(env_idx, obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()
        for instance in self._instances:
            instance.terminate()

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    # -- SB3 VecEnv ABC boilerplate: per-index delegation to the sub-envs -----

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [env.get_wrapper_attr(attr_name) for env in self._get_target_envs(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for env in self._get_target_envs(indices):
            setattr(env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return [env.get_wrapper_attr(method_name)(*method_args, **method_kwargs) for env in self._get_target_envs(indices)]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env, wrapper_class) for env in self._get_target_envs(indices)]

    def _get_target_envs(self, indices: VecEnvIndices) -> list[XmlDefinedEnv]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
