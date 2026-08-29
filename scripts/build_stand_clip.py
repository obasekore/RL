"""Builds the hand-crafted "stand" clip used to verify biped_nao physics survival -
the simplest possible clip (a held standing pose, not a walk cycle), deliberately
scoped to isolate whether the scene itself (mass/contact/limits/friction) is
physically viable before anything as complex as locomotion.

Root height is derived from forward_kinematics at the skeleton's rest pose (all
DOF angles zero, matching NAO's own confirmed rest configuration) plus each foot's
contact_geometry offset, so the feet land exactly at ground level (world z=0) -
not guessed.

Usage:
    .venv/Scripts/python.exe scripts/build_stand_clip.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coppelia_rl.motion.clip import Clip, save_clip
from coppelia_rl.motion.schema import ChannelsBlock, ClipHeader, ClipMeta, Provenance, SkeletonBinding
from coppelia_rl.motion.tracks_hdf5 import ContactTrack, TracksData
from coppelia_rl.retargeting.fk import end_effector_position, forward_kinematics
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml

_ROOT_DIR = Path(__file__).resolve().parents[1]
_SKELETON_PATH = _ROOT_DIR / "skeletons" / "biped_nao.yaml"
_CLIP_DIR = _ROOT_DIR / "clips" / "stand"

_FRAME_RATE = 20.0
_N_FRAMES = 5


def _standing_root_height(skeleton) -> float:
    """Root z such that both feet' contact points land at world z=0, computed from
    forward kinematics at the rest pose (all DOF angles zero) - not guessed."""
    dof_names = ordered_dof_names(skeleton)
    transforms = forward_kinematics(skeleton, {})  # missing keys default to 0.0
    heights = []
    for foot_bone in ("L_foot", "R_foot"):
        bone = next(b for b in skeleton.bones if b.name == foot_bone)
        foot_pos = end_effector_position(skeleton, transforms, foot_bone)
        contact_z_offset = bone.contact_geometry.offset[2]
        heights.append(-(foot_pos[2] + contact_z_offset))
    return float(np.mean(heights))


def main() -> None:
    skeleton = parse_skeleton_yaml(_SKELETON_PATH)
    dof_names = ordered_dof_names(skeleton)
    n_dof = len(dof_names)

    root_z = _standing_root_height(skeleton)
    print(f"Computed standing root height: {root_z:.4f} m")

    header = ClipHeader(
        clip=ClipMeta(
            name="stand",
            format_version=1,
            frame_rate=_FRAME_RATE,
            duration_s=_N_FRAMES / _FRAME_RATE,
            loop="cyclic",
        ),
        provenance=Provenance(source_type="create", notes="hand-crafted standing pose, physics-scene follow-up"),
        skeleton=SkeletonBinding(target=_SKELETON_PATH.resolve(), morphology_class="biped"),
        # Hand-authored directly (no retargeting solver involved) - "authored", not
        # "derived", per Doc 2's authored/derived contract.
        channels=ChannelsBlock(root_pose="authored", joint_angles="authored", contact_state="authored"),
    )

    joint_angles = np.zeros((_N_FRAMES, n_dof), dtype=np.float32)
    root_pose = np.tile(np.array([0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (_N_FRAMES, 1))
    contact = [
        ContactTrack(name="L_foot", contact=np.ones(_N_FRAMES, dtype=bool)),
        ContactTrack(name="R_foot", contact=np.ones(_N_FRAMES, dtype=bool)),
    ]
    tracks = TracksData(
        root_pose=root_pose,
        joint_angles=joint_angles,
        joint_angles_names=dof_names,
        contact=contact,
        phase_variable=np.linspace(0.0, 1.0, _N_FRAMES, dtype=np.float32),
    )

    save_clip(_CLIP_DIR, Clip(header=header, tracks=tracks), skeleton=skeleton)
    print(f"Stand clip written to {_CLIP_DIR}")


if __name__ == "__main__":
    main()
