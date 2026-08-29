"""Procedurally builds (or reloads) the biped_nao physics-verification scene.

Loads the stock NAO.ttm model and renames its joints/base to match
skeletons/biped_nao.yaml's DOF/bone names, per the DOF/bone -> scene-object
convention documented in coppelia_rl/env_schema/motion_imitation.py: every
ordered_dof_names(skeleton) entry aliased to a joint object, root_bone and every
end_effector aliased to a plain scene object.

NAO's real joint chain (confirmed live via sim.getObjectsInTree/getObjectParent
against this exact CoppeliaSim install, not assumed from general NAO knowledge) is
already a clean single-axis-per-joint serial chain per limb - see
skeletons/biped_nao.yaml's own field-by-field derivation. This module only renames
existing joints, it doesn't restructure the kinematic tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from coppelia_rl.sim_interface.client import SimClient
from coppelia_rl.sim_interface.objects import Joint, SceneObject

_NAO_MODEL_RELATIVE_PATH = Path("models") / "robots" / "mobile" / "NAO.ttm"

# Our skeleton DOF name -> NAO's stock joint alias (see skeletons/biped_nao.yaml).
_DOF_TO_NAO_ALIAS = {
    "L_hip_yaw": "LHipYawPitch",
    "L_hip_roll": "LHipRoll",
    "L_hip_pitch": "LHipPitch",
    "L_knee": "LKneePitch",
    "L_ankle_pitch": "LAnklePitch",
    "L_ankle_roll": "LAnkleRoll",
    "R_hip_yaw": "RHipYawPitch",
    "R_hip_roll": "RHipRoll",
    "R_hip_pitch": "RHipPitch",
    "R_knee": "RKneePitch",
    "R_ankle_pitch": "RAnklePitch",
    "R_ankle_roll": "RAnkleRoll",
    "L_shoulder_pitch": "LShoulderPitch",
    "L_shoulder_roll": "LShoulderRoll",
    "L_elbow_yaw": "LElbowYaw",
    "L_elbow": "LElbowRoll",
    "R_shoulder_pitch": "RShoulderPitch",
    "R_shoulder_roll": "RShoulderRoll",
    "R_elbow_yaw": "RElbowYaw",
    "R_elbow": "RElbowRoll",
}

# End-effector marker dummies: our bone name -> (owning DOF joint alias, local offset
# from that joint to the end-effector point, matching skeletons/biped_nao.yaml's own
# bone offsets for the same bone).
_END_EFFECTOR_DUMMIES = {
    "L_foot": ("L_ankle_roll", (0.0, 0.0, -0.03)),
    "R_foot": ("R_ankle_roll", (0.0, 0.0, -0.03)),
    "L_hand": ("L_elbow", (0.0, 0.0, 0.0)),
    "R_hand": ("R_elbow", (0.0, 0.0, 0.0)),
}

# Real joints that exist on the stock model but aren't part of biped_nao.yaml's DOF
# set - frozen at 0 (position-mode, held) rather than left to hang loose under gravity.
_FROZEN_JOINT_ALIASES = ["LWristYaw", "RWristYaw"]


class NaoModelNotFoundError(RuntimeError):
    pass


def _default_nao_model_path() -> Path:
    override = os.environ.get("COPPELIASIM_ROOT")
    if override:
        candidate = Path(override) / _NAO_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate
        raise NaoModelNotFoundError(f"COPPELIASIM_ROOT is set to {override!r} but {candidate} does not exist")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _NAO_MODEL_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise NaoModelNotFoundError(
        f"Could not locate {_NAO_MODEL_RELATIVE_PATH} by walking up from this file. "
        "Set the COPPELIASIM_ROOT environment variable to the install directory."
    )


@dataclass
class BipedNaoScene:
    torso: SceneObject
    dof_joints: dict[str, Joint]
    end_effectors: dict[str, SceneObject]


def ensure_biped_nao_scene(
    client: SimClient,
    scene_path: str | Path,
    nao_model_path: str | Path | None = None,
) -> BipedNaoScene:
    scene_path = Path(scene_path)

    if scene_path.exists():
        client.load_scene(scene_path)
        torso = client.get_object("/torso")
        dof_joints = {dof_name: client.get_joint(f"/{dof_name}") for dof_name in _DOF_TO_NAO_ALIAS}
        end_effectors = {name: client.get_object(f"/{name}") for name in _END_EFFECTOR_DUMMIES}
        return BipedNaoScene(torso=torso, dof_joints=dof_joints, end_effectors=end_effectors)

    client.close_scene()
    model_path = Path(nao_model_path) if nao_model_path else _default_nao_model_path()
    torso = client.load_model(model_path)
    torso.set_name("torso")

    dof_joints: dict[str, Joint] = {}
    for dof_name, nao_alias in _DOF_TO_NAO_ALIAS.items():
        joint = client.get_joint(f"/{nao_alias}")
        joint.set_name(dof_name)
        joint.set_control_mode("position")
        dof_joints[dof_name] = joint

    for nao_alias in _FROZEN_JOINT_ALIASES:
        joint = client.get_joint(f"/{nao_alias}")
        joint.set_control_mode("position")
        joint.set_target_position(0.0)

    end_effectors: dict[str, SceneObject] = {}
    for name, (owning_dof, local_offset) in _END_EFFECTOR_DUMMIES.items():
        dummy = client.create_dummy(size=0.01)
        dummy.set_name(name)
        owner = dof_joints[owning_dof]
        dummy.set_parent(owner, keep_in_place=False)
        dummy.set_position(list(local_offset), relative_to=owner.handle)
        end_effectors[name] = dummy

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    client.save_scene(scene_path)

    return BipedNaoScene(torso=torso, dof_joints=dof_joints, end_effectors=end_effectors)
