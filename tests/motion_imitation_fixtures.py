"""Shared synthetic-clip + FakeSim-scene builders for motion_imitation env tests.

Referenced by tests.test_motion_imitation_env - kept separate so the fixture-building
logic (hand-crafted, non-random clip data + FakeSim scene aliasing) doesn't crowd out
the actual test assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from coppelia_rl.motion.clip import Clip, save_clip  # noqa: E402
from coppelia_rl.motion.schema import (  # noqa: E402
    ChannelsBlock,
    ClipHeader,
    ClipMeta,
    Provenance,
    SkeletonBinding,
    ValidationBlock,
)
from coppelia_rl.motion.tracks_hdf5 import ContactTrack, TracksData  # noqa: E402
from coppelia_rl.retargeting.fk import dof_rest_angles_rad  # noqa: E402
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml  # noqa: E402

QUADRUPED_PATH = Path(__file__).resolve().parents[1] / "skeletons" / "quadruped_generic.yaml"
QUADRUPED_SKELETON = parse_skeleton_yaml(QUADRUPED_PATH)
QUADRUPED_DOF_NAMES = ordered_dof_names(QUADRUPED_SKELETON)
QUADRUPED_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
QUADRUPED_REST_RAD = dof_rest_angles_rad(QUADRUPED_SKELETON, QUADRUPED_DOF_NAMES)
FL_HIP_PITCH_DELTA = 0.3  # rad, applied to frames 1+ so a known pose_tracking error is computable


def build_synthetic_quadruped_clip(
    tmp_path: Path,
    *,
    frame_rate: float = 20.0,
    loop: str = "cyclic",
    n_frames: int = 4,
    validation_status: str = "ok",
    subdir: str = "clip",
) -> Path:
    """Hand-crafted (non-random) clip, saved to `tmp_path/subdir`, returning that
    directory. Frame 0 = the skeleton's own rest pose; frames 1+ = rest pose +
    FL_HIP_PITCH_DELTA on FL_hip_pitch only (so pose/velocity-tracking error against
    a known live state is directly computable in tests). FL_foot is in contact
    (stance) on frame 0 only, swing thereafter; RL_foot is in contact every frame
    (steady stance, tests the "no slip" contact_matching case).
    """
    joint_angles = np.tile(QUADRUPED_REST_RAD, (n_frames, 1))
    if n_frames > 1:
        perturbed = QUADRUPED_REST_RAD.copy()
        perturbed[QUADRUPED_DOF_NAMES.index("FL_hip_pitch")] += FL_HIP_PITCH_DELTA
        joint_angles[1:] = perturbed

    root_pose = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (n_frames, 1)).astype(np.float32)

    fl_contact = np.zeros(n_frames, dtype=bool)
    fl_contact[0] = True
    rl_contact = np.ones(n_frames, dtype=bool)

    header = ClipHeader(
        clip=ClipMeta(
            name="synthetic_quad", format_version=1, frame_rate=frame_rate, duration_s=n_frames / frame_rate, loop=loop
        ),
        provenance=Provenance(source_type="create"),
        skeleton=SkeletonBinding(target=QUADRUPED_PATH.resolve(), morphology_class="quadruped"),
        channels=ChannelsBlock(root_pose="authored", joint_angles="derived"),
        validation=ValidationBlock(status=validation_status),
    )
    tracks = TracksData(
        root_pose=root_pose,
        joint_angles=joint_angles.astype(np.float32),
        joint_angles_names=list(QUADRUPED_DOF_NAMES),
        contact=[
            ContactTrack(name="FL_foot", contact=fl_contact),
            ContactTrack(name="RL_foot", contact=rl_contact),
        ],
        phase_variable=np.linspace(0.0, 1.0, n_frames, dtype=np.float32),
    )

    clip_dir = tmp_path / subdir
    save_clip(clip_dir, Clip(header=header, tracks=tracks), skeleton=QUADRUPED_SKELETON)
    return clip_dir


def build_fake_sim_scene_for_quadruped(fake_sim, skeleton=QUADRUPED_SKELETON) -> dict[str, int]:
    """Aliases FakeSim objects per the DOF/bone -> scene-object convention documented
    in coppelia_rl/env_schema/motion_imitation.py: one joint object per DOF name, one
    plain object for root_bone, one plain object per end effector."""
    handles: dict[str, int] = {}
    for name in ordered_dof_names(skeleton):
        handle = fake_sim._new_handle("joint", joint_position=0.0, joint_velocity=0.0)
        fake_sim.setObjectAlias(handle, name)
        handles[name] = handle

    root_handle = fake_sim._new_handle("dummy")
    fake_sim.setObjectAlias(root_handle, skeleton.root_bone)
    handles[skeleton.root_bone] = root_handle

    for name in skeleton.end_effectors:
        handle = fake_sim._new_handle("dummy")
        fake_sim.setObjectAlias(handle, name)
        handles[name] = handle

    return handles
