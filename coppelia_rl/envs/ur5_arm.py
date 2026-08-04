"""Shared UR5-loading helper used by both the reach and pick-place scene builders.

Kept separate from scene_builder.py (reach-specific) since pick_place_scene.py
needs the exact same "load UR5, resolve its 6 joints, resolve+alias the tip as
gripper_tip" logic but with a different target object attached afterwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from coppelia_rl.sim_interface.client import SimClient
from coppelia_rl.sim_interface.objects import Joint, SceneObject

_UR5_MODEL_RELATIVE_PATH = Path("models") / "robots" / "non-mobile" / "UR5.ttm"


class Ur5ModelNotFoundError(RuntimeError):
    pass


def default_ur5_model_path() -> Path:
    override = os.environ.get("COPPELIASIM_ROOT")
    if override:
        candidate = Path(override) / _UR5_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate
        raise Ur5ModelNotFoundError(f"COPPELIASIM_ROOT is set to {override!r} but {candidate} does not exist")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _UR5_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise Ur5ModelNotFoundError(
        f"Could not locate {_UR5_MODEL_RELATIVE_PATH} by walking up from this file. "
        "Set the COPPELIASIM_ROOT environment variable to the install directory."
    )


@dataclass
class Ur5Arm:
    base: SceneObject
    joints: list[Joint]
    tip: SceneObject


def load_ur5_arm(client: SimClient, ur5_model_path: str | Path | None = None) -> Ur5Arm:
    """Loads a fresh UR5 model into the scene and resolves its joints/tip."""
    model_path = Path(ur5_model_path) if ur5_model_path else default_ur5_model_path()
    base = client.load_model(model_path)
    base.set_name("UR5")
    return _resolve_arm(client, base)


def resolve_ur5_arm(client: SimClient) -> Ur5Arm:
    """Resolves a UR5 already present in a just-loaded scene, by its "/UR5" alias."""
    base = client.get_object("/UR5")
    return _resolve_arm(client, base)


def _resolve_arm(client: SimClient, base: SceneObject) -> Ur5Arm:
    joints = client.get_joints_in_tree(base)
    if not joints:
        raise RuntimeError("UR5 model has no joints in its tree")

    # Idempotent: backfills stable aliases on older saved scenes too, and gives
    # the XML schema layer addressable per-joint refs ("UR5_joint1".."UR5_joint6",
    # matching the spec's own example) instead of only positional access.
    for index, joint in enumerate(joints, start=1):
        joint.set_name(f"UR5_joint{index}")

    tip = _resolve_tip(client, joints)
    tip.set_name("gripper_tip")

    return Ur5Arm(base=base, joints=joints, tip=tip)


def _resolve_tip(client: SimClient, joints: list[Joint]) -> SceneObject:
    """Uses the last joint's dummy child as the end-effector proxy, if any.

    The stock UR5.ttm has no such dummy, so a dedicated one is created and
    parented there instead - it must be a distinct object from the last
    joint, since both need their own stable alias ("UR5_jointN" vs.
    "gripper_tip") and CoppeliaSim aliases are one object to one name.
    """
    last_joint = joints[-1]
    children = client.get_child_dummies(last_joint)
    if children:
        return children[0]

    tip = client.create_dummy(size=0.02)
    tip.set_parent(last_joint, keep_in_place=False)
    return tip
