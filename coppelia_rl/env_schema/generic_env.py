"""A single gymnasium.Env driven entirely by a parsed EnvSpec.

Registries below map each XML element "kind" to a resolver that binds a
built-in observation/action/reward/termination handler against SimClient
object-model instances. `type="custom"` entries go through
escape_hatch.resolve_callable instead, per the escape-hatch rule.

Every reader/applier/term/check uses a uniform `(env) -> value` (or
`(env, value) -> None` for actions) signature, whether built-in or custom, so
custom.read/apply/compute/check can be substituted transparently.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from coppelia_rl.env_schema import motion_imitation as motion_imitation_module
from coppelia_rl.env_schema.escape_hatch import resolve_callable
from coppelia_rl.env_schema.motion_imitation import _MotionImitationRuntime
from coppelia_rl.env_schema.spec import (
    ActionEntrySpec,
    ActionGroupSpec,
    EnvSpec,
    ObservationSpec,
    RewardTermSpec,
    TerminationConditionSpec,
)
from coppelia_rl.randomization.randomizer import Randomizer
from coppelia_rl.sim_interface.client import SimClient


def _ref_path(ref: str) -> str:
    return ref if ref.startswith("/") else f"/{ref}"


def _resolve_frame(client: SimClient, frame: str | None) -> int:
    if frame is None or frame == "world":
        return -1
    return client.get_object(_ref_path(frame)).handle


class _ObservationReader:
    def __init__(self, space: gym.Space, read: Callable[["XmlDefinedEnv"], Any]):
        self.space = space
        self.read = read


class _ActionApplier:
    def __init__(self, apply: Callable[["XmlDefinedEnv", float], None], reset_hook: Callable[[], None] | None = None):
        self.apply = apply
        # Re-run after every load_scene() (i.e. every reset()), not just once
        # at env construction - confirmed live that some object properties
        # (joint dynamic control mode, like vision sensors' explicit-handling
        # flag in milestone 6) are runtime-only and silently revert on scene
        # reload.
        self.reset_hook = reset_hook


# -- observations -----------------------------------------------------------


def _build_observation_reader(client: SimClient, obs: ObservationSpec) -> _ObservationReader:
    if obs.kind == "joint_position":
        joint = client.get_joint(_ref_path(obs.ref))
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        return _ObservationReader(space, lambda env: np.array([joint.get_joint_position()], dtype=np.float32))

    if obs.kind == "joint_velocity":
        joint = client.get_joint(_ref_path(obs.ref))
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        return _ObservationReader(space, lambda env: np.array([joint.get_joint_velocity()], dtype=np.float32))

    if obs.kind == "object_pose":
        target = client.get_object(_ref_path(obs.ref))
        relative_to = _resolve_frame(client, obs.frame)
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        return _ObservationReader(space, lambda env: target.get_pose(relative_to).astype(np.float32))

    if obs.kind == "object_position":
        target = client.get_object(_ref_path(obs.ref))
        relative_to = _resolve_frame(client, obs.frame)
        space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        return _ObservationReader(space, lambda env: target.get_position(relative_to).astype(np.float32))

    if obs.kind == "camera":
        vision = client.get_vision_sensor(_ref_path(obs.ref))
        width, height = obs.resolution
        if obs.channels == "depth":
            space = spaces.Box(low=0.0, high=1.0, shape=(height, width), dtype=np.float32)
            return _ObservationReader(space, lambda env: vision.get_depth())
        space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        return _ObservationReader(space, lambda env: vision.get_rgb())

    if obs.kind == "custom":
        custom = resolve_callable(obs.callable)
        return _ObservationReader(custom.space, custom.read)

    raise ValueError(f"Unknown observation kind {obs.kind!r}")


# -- actions ------------------------------------------------------------------


def _build_continuous_action(client: SimClient, entry: ActionEntrySpec) -> tuple[_ActionApplier, tuple[float, float]]:
    if entry.kind == "joint_velocity":
        joint = client.get_joint(_ref_path(entry.ref))
        joint.set_control_mode("velocity")
        applier = _ActionApplier(
            lambda env, value: joint.set_target_velocity(float(value)),
            reset_hook=lambda: joint.set_control_mode("velocity"),
        )
        return applier, entry.value_range

    if entry.kind == "joint_position":
        joint = client.get_joint(_ref_path(entry.ref))
        joint.set_control_mode("position")
        applier = _ActionApplier(
            lambda env, value: joint.set_target_position(float(value)),
            reset_hook=lambda: joint.set_control_mode("position"),
        )
        return applier, entry.value_range

    if entry.kind == "custom":
        # Continuous custom entries contribute exactly one scalar dimension,
        # like the built-in joint_* entries: the object exposes .value_range
        # instead of a full gymnasium.Space (no cross-entry offset bookkeeping needed).
        custom = resolve_callable(entry.callable)
        return _ActionApplier(custom.apply), custom.value_range

    raise ValueError(f"Unknown continuous action kind {entry.kind!r}")


def _build_discrete_action(client: SimClient, entry: ActionEntrySpec) -> _ActionApplier:
    if entry.kind == "event":
        signal = client.get_signal(entry.signal)
        return _ActionApplier(lambda env, value: signal.set_int32(int(value)))

    if entry.kind == "custom":
        custom = resolve_callable(entry.callable)
        return _ActionApplier(custom.apply)

    raise ValueError(f"Unknown discrete action kind {entry.kind!r}")


def _build_action_space(client: SimClient, actions: ActionGroupSpec) -> tuple[list[_ActionApplier], gym.Space]:
    if actions.action_type == "continuous":
        appliers = []
        lows = []
        highs = []
        for entry in actions.entries:
            applier, value_range = _build_continuous_action(client, entry)
            appliers.append(applier)
            lows.append(value_range[0])
            highs.append(value_range[1])
        space = spaces.Box(low=np.array(lows, dtype=np.float32), high=np.array(highs, dtype=np.float32), dtype=np.float32)
        return appliers, space

    appliers = [_build_discrete_action(client, entry) for entry in actions.entries]
    return appliers, spaces.Discrete(len(appliers))


# -- reward ---------------------------------------------------------------------


def _build_reward_term(
    client: SimClient, term: RewardTermSpec, motion_imitation: "_MotionImitationRuntime | None", step_dt: float
) -> Callable[["XmlDefinedEnv"], float]:
    weight = term.weight

    if term.kind == "distance":
        obj_from = client.get_object(_ref_path(term.from_ref))
        obj_to = client.get_object(_ref_path(term.to_ref))
        return lambda env: weight * float(np.linalg.norm(obj_from.get_position() - obj_to.get_position()))

    if term.kind == "collision":
        obj1 = client.get_object(_ref_path(term.object1))
        obj2 = client.get_object(_ref_path(term.object2))
        return lambda env: weight if client.check_collision(obj1, obj2) else 0.0

    if term.kind == "signal_event":
        signal = client.get_signal(term.signal)

        def compute(env: "XmlDefinedEnv") -> float:
            triggered = bool(signal.get_int32())
            if triggered:
                signal.set_int32(0)
            return weight if triggered else 0.0

        return compute

    if term.kind in ("pose_tracking", "velocity_tracking", "end_effector_tracking", "contact_matching"):
        if motion_imitation is None:
            raise ValueError(f'reward type="{term.kind}" requires a <motion_imitation> block')
        if term.kind == "pose_tracking":
            return lambda env: weight * motion_imitation_module.pose_tracking_reward(motion_imitation)
        if term.kind == "velocity_tracking":
            return lambda env: weight * motion_imitation_module.velocity_tracking_reward(motion_imitation)
        if term.kind == "end_effector_tracking":
            return lambda env: weight * motion_imitation_module.end_effector_tracking_reward(motion_imitation)
        return lambda env: weight * motion_imitation_module.contact_matching_reward(motion_imitation, step_dt)

    if term.kind == "custom":
        fn = resolve_callable(term.callable)
        return lambda env: weight * float(fn(env))

    raise ValueError(f"Unknown reward term kind {term.kind!r}")


# -- termination ------------------------------------------------------------------


def _build_termination_check(
    client: SimClient, cond: TerminationConditionSpec, motion_imitation: "_MotionImitationRuntime | None"
) -> tuple[bool, Callable[["XmlDefinedEnv"], bool]]:
    """Returns (is_truncation, check). max_steps is a truncation; everything else terminates."""
    if cond.kind == "max_steps":
        max_steps = int(cond.value)
        return True, lambda env: env.step_count >= max_steps

    if cond.kind == "signal":
        signal = client.get_signal(cond.name)

        def check(env: "XmlDefinedEnv") -> bool:
            triggered = bool(signal.get_int32())
            if triggered:
                signal.set_int32(0)
            return triggered

        return False, check

    if cond.kind == "fall_detection":
        if motion_imitation is None:
            raise ValueError('termination type="fall_detection" requires a <motion_imitation> block')
        return False, motion_imitation_module.make_fall_detection_check(motion_imitation)

    if cond.kind == "custom":
        return False, resolve_callable(cond.callable)

    raise ValueError(f"Unknown termination condition kind {cond.kind!r}")


def _collect_reset_signals(client: SimClient, spec: EnvSpec) -> list:
    signals = []
    for term in spec.reward_terms:
        if term.kind == "signal_event":
            signals.append(client.get_signal(term.signal))
    for cond in spec.termination_conditions:
        if cond.kind == "signal":
            signals.append(client.get_signal(cond.name))
    if spec.actions.action_type == "discrete":
        for entry in spec.actions.entries:
            if entry.kind == "event":
                signals.append(client.get_signal(entry.signal))
    return signals


class XmlDefinedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: EnvSpec, client: SimClient):
        super().__init__()
        # NOTE: `spec` collides in name (not type) with gymnasium.Env's own
        # reserved `.spec` attribute (a registry EnvSpec, unrelated to ours).
        # Direct access (`env.spec.step_dt`) works fine unwrapped, but
        # gym.Wrapper.__init__ tries to *copy* env.spec assuming it's the
        # gymnasium type, fails, and silently falls back to `self.spec =
        # None` on the wrapper (only a UserWarning, easy to miss) - so any
        # `gym.Wrapper` around this env (e.g. DemoRecorder) must reach this
        # via `wrapped_env.unwrapped.spec`, never `wrapped_env.spec`.
        self.spec = spec
        self._client = client
        self.step_count = 0

        # Refs are resolved against whatever scene is currently open, so it
        # must be spec.scene_path - not whatever a previous env/script left
        # loaded on this CoppeliaSim instance.
        client.stop_simulation()
        client.load_scene(spec.scene_path)

        # Built before observations/actions/reward/termination since the
        # pose_tracking/velocity_tracking/end_effector_tracking/contact_matching
        # reward kinds and the fall_detection termination kind need it.
        self._motion_imitation: "_MotionImitationRuntime | None" = None
        self._reset_state_hooks: list[Callable[[], None]] = []
        if spec.motion_imitation is not None:
            self._motion_imitation = motion_imitation_module.build_motion_imitation_runtime(
                client, spec.motion_imitation
            )
            expected_dt = 1.0 / self._motion_imitation.clip.header.clip.frame_rate
            if not math.isclose(spec.step_dt, expected_dt, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    f"step_dt={spec.step_dt} must equal 1/clip.frame_rate={expected_dt} for a "
                    "motion_imitation env - v1 does no interpolation/resampling between the two rates"
                )
            self._reset_state_hooks.append(
                lambda: self._motion_imitation.inject_reset_state(self.np_random, spec.motion_imitation.rsi)
            )

        self._obs_readers = {obs.key: _build_observation_reader(client, obs) for obs in spec.observations}
        self.observation_space = spaces.Dict({key: reader.space for key, reader in self._obs_readers.items()})

        self._action_appliers, self.action_space = _build_action_space(client, spec.actions)

        self._reward_terms = [
            _build_reward_term(client, term, self._motion_imitation, spec.step_dt) for term in spec.reward_terms
        ]
        self._termination_checks = [
            _build_termination_check(client, cond, self._motion_imitation) for cond in spec.termination_conditions
        ]
        if self._motion_imitation is not None and self._motion_imitation.loop == "one_shot":
            # Running out of reference frames is "the script ended," not a failure
            # state like fall_detection - truncates, matching max_steps's precedent,
            # not terminates.
            self._termination_checks.append((True, lambda env: env._motion_imitation.exhausted))
        self._reset_signals = _collect_reset_signals(client, spec)

        # Domain randomization is "enabled" simply by the spec having a
        # domain_randomization block - no separate flag.
        self._randomizer = Randomizer(client, spec.domain_randomization) if spec.domain_randomization else None

    @property
    def client(self) -> SimClient:
        """Public access to the SimClient, for custom escape-hatch callables.

        Still the object-model layer (SceneObject/Joint/...), not raw sim.*,
        so this doesn't violate the Communication Layer boundary - it's the
        sanctioned "power user" doorway the spec's escape-hatch rule requires.
        """
        return self._client

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._client.stop_simulation()
        self._client.load_scene(self.spec.scene_path)
        # Runs unconditionally whenever a motion_imitation block exists, even with
        # RSI disabled - it also resets the runtime's own per-episode state (frame
        # index, ping_pong direction, one_shot exhaustion, the contact-matching slip
        # cache), which must happen every episode regardless of the RSI flag.
        for hook in self._reset_state_hooks:
            hook()
        for applier in self._action_appliers:
            if applier.reset_hook is not None:
                applier.reset_hook()
        for signal in self._reset_signals:
            signal.set_int32(0)
        if self._randomizer is not None:
            self._randomizer.maybe_resample(episode_start=True)
        self._client.set_stepping(True)
        self._client.start_simulation()
        self.step_count = 0

        obs = self._get_obs()
        if self._randomizer is not None:
            obs = self._randomizer.add_observation_noise(obs)
        return obs, {}

    def step(self, action):
        if self._randomizer is not None:
            action = self._randomizer.delay_action(action)
        self._apply_action(action)
        self._client.step()
        self.step_count += 1
        if self._motion_imitation is not None:
            self._motion_imitation.advance_frame()

        if self._randomizer is not None:
            self._randomizer.notify_step()
            self._randomizer.maybe_resample(episode_start=False)

        reward = sum(term(self) for term in self._reward_terms)

        terminated = False
        truncated = False
        for is_truncation, check in self._termination_checks:
            if check(self):
                if is_truncation:
                    truncated = True
                else:
                    terminated = True

        obs = self._get_obs()
        if self._randomizer is not None:
            obs = self._randomizer.add_observation_noise(obs)
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._client.close()

    def _apply_action(self, action) -> None:
        if self.spec.actions.action_type == "continuous":
            for applier, value in zip(self._action_appliers, np.asarray(action, dtype=np.float32)):
                applier.apply(self, float(value))
        else:
            chosen = int(action)
            for index, applier in enumerate(self._action_appliers):
                applier.apply(self, 1.0 if index == chosen else 0.0)

    def _get_obs(self) -> dict:
        return {key: reader.read(self) for key, reader in self._obs_readers.items()}
