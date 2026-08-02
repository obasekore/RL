"""Custom escape-hatch termination condition for the mobile-nav task.

Referenced from tasks/mobile_nav.xml as
type="custom" callable="tasks.mobile_nav_task:reached_goal" - demonstrates
the schema's custom-callback escape hatch working end to end for termination,
using the same SimClient/object-model the built-in condition types use.
"""

from __future__ import annotations

import numpy as np

_SUCCESS_DISTANCE = 0.2


def reached_goal(env) -> bool:
    base = env.client.get_object("/PioneerP3DX")
    goal = env.client.get_object("/NavGoal")
    distance = float(np.linalg.norm(base.get_position() - goal.get_position()))
    return distance < _SUCCESS_DISTANCE
