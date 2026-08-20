"""Custom escape-hatch callables used only by test_generic_env.py.

Referenced by dotted path (e.g. "tests.custom_fixtures:constant_observation"),
exactly how a real task's custom callables would be referenced from XML.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


class _ConstantObservation:
    space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def read(self, env):
        return np.array([0.5, -0.5], dtype=np.float32)


constant_observation = _ConstantObservation()


class _RecordingAction:
    value_range = (-2.0, 2.0)

    def __init__(self):
        self.last_value = None

    def apply(self, env, value):
        self.last_value = value


recording_action = _RecordingAction()


def custom_reward(env) -> float:
    return 3.0


def custom_termination(env) -> bool:
    return env.step_count >= 2
