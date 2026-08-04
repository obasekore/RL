"""Procedurally builds (or reloads) the mobile-navigation task scene.

Loads a Pioneer P3DX (a standard differential-drive robot) and a goal dummy.
No obstacles - this is a schema-generality proof (a non-arm morphology, wheel
velocities instead of joint angles), not a navigation-difficulty benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coppelia_rl.sim_interface.client import SimClient
from coppelia_rl.sim_interface.objects import Joint, SceneObject

_PIONEER_MODEL_RELATIVE_PATH = Path("models") / "robots" / "mobile" / "pioneer p3dx.ttm"
# Standard CoppeliaSim naming for the Pioneer p3dx's two motorized wheel joints.
_LEFT_MOTOR_ALIAS = "/Pioneer_p3dx_leftMotor"
_RIGHT_MOTOR_ALIAS = "/Pioneer_p3dx_rightMotor"


class PioneerModelNotFoundError(RuntimeError):
    pass


def _default_pioneer_model_path() -> Path:
    override = os.environ.get("COPPELIASIM_ROOT")
    if override:
        candidate = Path(override) / _PIONEER_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate
        raise PioneerModelNotFoundError(f"COPPELIASIM_ROOT is set to {override!r} but {candidate} does not exist")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _PIONEER_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise PioneerModelNotFoundError(
        f"Could not locate {_PIONEER_MODEL_RELATIVE_PATH} by walking up from this file. "
        "Set the COPPELIASIM_ROOT environment variable to the install directory."
    )


@dataclass
class MobileNavScene:
    base: SceneObject
    left_wheel: Joint
    right_wheel: Joint
    goal: SceneObject


def ensure_mobile_nav_scene(
    client: SimClient,
    scene_path: str | Path,
    pioneer_model_path: str | Path | None = None,
) -> MobileNavScene:
    scene_path = Path(scene_path)

    if scene_path.exists():
        client.load_scene(scene_path)
        base = client.get_object("/PioneerP3DX")
        left_wheel = client.get_joint("/left_wheel")
        right_wheel = client.get_joint("/right_wheel")
        goal = client.get_object("/NavGoal")
        return MobileNavScene(base=base, left_wheel=left_wheel, right_wheel=right_wheel, goal=goal)

    client.close_scene()
    model_path = Path(pioneer_model_path) if pioneer_model_path else _default_pioneer_model_path()
    base = client.load_model(model_path)
    base.set_name("PioneerP3DX")

    joints = client.get_joints_in_tree(base)
    left_wheel, right_wheel = _resolve_motor_joints(client, joints)
    left_wheel.set_name("left_wheel")
    right_wheel.set_name("right_wheel")

    goal = client.create_dummy(size=0.05)
    goal.set_name("NavGoal")
    goal.set_position(sample_goal_position(np.random.default_rng()))

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    client.save_scene(scene_path)

    return MobileNavScene(base=base, left_wheel=left_wheel, right_wheel=right_wheel, goal=goal)


def _resolve_motor_joints(client: SimClient, joints: list[Joint]) -> tuple[Joint, Joint]:
    """Prefers the model's own well-known joint names; falls back to tree order."""
    try:
        return client.get_joint(_LEFT_MOTOR_ALIAS), client.get_joint(_RIGHT_MOTOR_ALIAS)
    except Exception:
        if len(joints) < 2:
            raise RuntimeError("Pioneer p3dx model does not expose at least 2 joints") from None
        return joints[0], joints[1]


def sample_goal_position(rng: np.random.Generator) -> np.ndarray:
    """Samples a goal position on the open floor around the robot's start pose."""
    x = rng.uniform(-2.0, 2.0)
    y = rng.uniform(-2.0, 2.0)
    return np.array([x, y, 0.05], dtype=np.float64)
