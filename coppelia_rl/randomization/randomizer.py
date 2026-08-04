"""Domain randomization engine.

Deliberately independent of gymnasium/EnvSpec - it takes a SimClient and a
DomainRandomizationSpec and nothing else, so the exact same engine backs both
`XmlDefinedEnv` (env_schema/generic_env.py, "enabled" just by a spec's
domain_randomization block being present) and a standalone add-on that
randomizes any CoppeliaSim scene with no RL env involved at all.
"""

from __future__ import annotations

import collections
import glob
from typing import TYPE_CHECKING

import numpy as np

from coppelia_rl.sim_interface.client import SimClient

if TYPE_CHECKING:
    # env_schema imports this module too (generic_env.py wires Randomizer
    # into XmlDefinedEnv), so this must stay type-checking-only to avoid a
    # runtime circular import when coppelia_rl.randomization is imported
    # before coppelia_rl.env_schema.
    from coppelia_rl.env_schema.spec import DomainRandomizationSpec


def _ref_path(ref: str) -> str:
    return ref if ref.startswith("/") else f"/{ref}"


class Randomizer:
    def __init__(self, client: SimClient, spec: DomainRandomizationSpec, rng: np.random.Generator | None = None):
        if spec.resample_on == "n_steps" and not spec.resample_every_n_steps:
            raise ValueError('resample_on="n_steps" requires resample_every_n_steps to be set')

        self._client = client
        self._spec = spec
        self._rng = rng if rng is not None else np.random.default_rng()
        self._step_count = 0
        self._last_resample_step: int | None = None
        self._action_delay_steps = 0
        self._action_queue: collections.deque = collections.deque()

        # Refs are resolved once up front: fails loudly at construction time
        # if a ref doesn't exist, and avoids re-resolving on every resample().
        self._textures = [(client.get_object(_ref_path(t.ref)), t) for t in spec.textures]
        self._masses = [(client.get_object(_ref_path(m.ref)), m) for m in spec.masses]
        self._frictions = [(client.get_object(_ref_path(f.ref)), f) for f in spec.frictions]

        self._cameras = []
        for cam in spec.cameras:
            vision = client.get_vision_sensor(_ref_path(cam.ref))
            # Jitter is always relative to this base pose, captured once - not
            # the vision sensor's current pose - so repeated resamples don't
            # cumulatively drift the camera away from its authored position.
            self._cameras.append((vision, cam, vision.get_position(), vision.get_orientation()))

    # -- visual + dynamics (mutate scene state) ------------------------------

    def resample(self) -> None:
        """Applies one fresh draw of every configured randomization category."""
        for shape, texture in self._textures:
            self._resample_texture(shape, texture)
        for vision, cam, base_pos, base_orient in self._cameras:
            self._resample_camera(vision, cam, base_pos, base_orient)
        for shape, mass in self._masses:
            self._client.set_mass(shape, float(self._rng.uniform(*mass.value_range)))
        for shape, friction in self._frictions:
            self._client.set_friction(shape, float(self._rng.uniform(*friction.value_range)))

        if self._spec.action_delay_steps is not None:
            lo, hi = self._spec.action_delay_steps
            self._action_delay_steps = int(self._rng.integers(lo, hi + 1))
            self._action_queue.clear()

    def _resample_texture(self, shape, texture) -> None:
        candidates = sorted(glob.glob(texture.pool))
        if not candidates:
            raise FileNotFoundError(f"No files match texture pool {texture.pool!r}")
        path = candidates[int(self._rng.integers(len(candidates)))]
        self._client.set_texture(shape, path)

    def _resample_camera(self, vision, cam, base_pos, base_orient) -> None:
        if cam.jitter_pos:
            offset = self._rng.uniform(-cam.jitter_pos, cam.jitter_pos, size=3)
            vision.set_position(base_pos + offset)
        if cam.jitter_rot_deg:
            offset = np.radians(self._rng.uniform(-cam.jitter_rot_deg, cam.jitter_rot_deg, size=3))
            vision.set_orientation(base_orient + offset)

    # -- resample_on policy ---------------------------------------------------

    def maybe_resample(self, *, episode_start: bool) -> None:
        policy = self._spec.resample_on
        if policy == "once":
            should_resample = self._last_resample_step is None
        elif policy == "episode_start":
            should_resample = episode_start
        elif policy == "n_steps":
            should_resample = (
                self._last_resample_step is None
                or self._step_count - self._last_resample_step >= self._spec.resample_every_n_steps
            )
        else:
            raise ValueError(f"Unknown resample_on {policy!r}")

        if should_resample:
            self.resample()
            self._last_resample_step = self._step_count

    def notify_step(self) -> None:
        self._step_count += 1

    # -- latency (applied every step, not tied to resample_on) ----------------

    def delay_action(self, action):
        """Holds the first action until enough history accumulates, then
        returns actions delayed by `action_delay_steps` (resampled per the
        resample_on policy, from spec.action_delay_steps' range)."""
        delay = self._action_delay_steps
        if delay <= 0:
            return action
        self._action_queue.append(action)
        if len(self._action_queue) <= delay:
            return self._action_queue[0]
        return self._action_queue.popleft()

    def add_observation_noise(self, obs: dict) -> dict:
        std = self._spec.observation_noise_std
        if not std:
            return obs
        noisy = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                noisy[key] = value + self._rng.normal(0.0, std, size=value.shape).astype(value.dtype)
            else:
                noisy[key] = value
        return noisy
