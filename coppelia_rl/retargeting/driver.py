"""Top-level retargeting driver: contact-lock -> per-frame warm-started DLS-IK ->
joint-velocity post-check, at both the array level (`retarget_tracks`) and the
`Clip` level (`retarget_clip`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coppelia_rl.motion.clip import Clip
from coppelia_rl.motion.schema import ChannelsBlock, ClipHeader, ValidationBlock, ValidationIssue
from coppelia_rl.motion.tracks_hdf5 import EndEffectorTrack, TracksData
from coppelia_rl.retargeting.contact_lock import contiguous_true_runs, lock_contacts
from coppelia_rl.retargeting.fk import (
    dof_bounds_rad,
    dof_max_velocity_rad_per_s,
    dof_rest_angles_rad,
    root_relative_to_world,
    world_to_root_relative,
)
from coppelia_rl.retargeting.ik_solver import solve_frame_ik
from coppelia_rl.skeleton.parser import dof_joint_names as skeleton_dof_joint_names
from coppelia_rl.skeleton.parser import ordered_dof_names
from coppelia_rl.skeleton.schema import SkeletonSpec


def check_joint_velocities(
    skeleton: SkeletonSpec,
    dof_names: list[str],
    joint_names: list[str],  # parallel array to dof_names, e.g. skeleton_dof_joint_names(skeleton)
    joint_angles_rad: np.ndarray,  # (N, J)
    frame_rate: float,
) -> list[ValidationIssue]:
    """Forward-differences joint_angles_rad and flags contiguous per-JOINT exceedance
    runs (max_velocity is declared once per joint, so a multi-axis joint's DOFs share
    one limit) as type='joint_velocity_exceeded', severity='warning'."""
    if joint_angles_rad.shape[0] < 2:
        return []

    velocities = np.diff(joint_angles_rad, axis=0) * frame_rate  # (N-1, J)
    max_velocity = dof_max_velocity_rad_per_s(skeleton, dof_names)  # (J,)
    exceeds = np.abs(velocities) > max_velocity[np.newaxis, :]  # (N-1, J)

    joints_seen: list[str] = []
    for joint in joint_names:
        if joint not in joints_seen:
            joints_seen.append(joint)

    issues: list[ValidationIssue] = []
    for joint in joints_seen:
        cols = [i for i, name in enumerate(joint_names) if name == joint]
        joint_exceeds = np.any(exceeds[:, cols], axis=1)
        for start, end in contiguous_true_runs(joint_exceeds):
            # `start`/`end` index the (N-1,)-length diff array; diff index i is the
            # transition between frame i and frame i+1.
            issues.append(
                ValidationIssue(
                    type="joint_velocity_exceeded",
                    joint=joint,
                    frame_range=(start, end + 1),
                    severity="warning",
                )
            )
    return issues


@dataclass
class RetargetResult:
    joint_angles_rad: np.ndarray  # (N, J)
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    iterations_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def retarget_tracks(
    skeleton: SkeletonSpec,
    dof_names: list[str],  # must equal ordered_dof_names(skeleton)
    n_frames: int,
    end_effector_targets: dict[str, np.ndarray],  # bone -> (N,3), root-relative
    contact: dict[str, np.ndarray],  # bone -> (N,) bool
    frame_rate: float,
    **solver_kwargs,
) -> RetargetResult:
    contact_capable_bones = {bone.name for bone in skeleton.bones if bone.contact_capable}
    locked_targets = lock_contacts(end_effector_targets, contact, contact_capable_bones)

    bounds_rad = dof_bounds_rad(skeleton, dof_names)
    rest_rad = dof_rest_angles_rad(skeleton, dof_names)
    n_dof = len(dof_names)

    joint_angles_rad = np.zeros((n_frames, n_dof))
    iterations_per_frame = np.zeros(n_frames, dtype=int)
    unreachable_frames_by_bone: dict[str, list[int]] = {}

    theta = rest_rad.copy()
    for frame in range(n_frames):
        targets = {bone: curve[frame] for bone, curve in locked_targets.items()}
        result = solve_frame_ik(skeleton, dof_names, bounds_rad, rest_rad, targets, theta, **solver_kwargs)
        joint_angles_rad[frame] = result.theta_rad
        iterations_per_frame[frame] = result.iterations
        for bone in result.unreachable_bones:
            unreachable_frames_by_bone.setdefault(bone, []).append(frame)
        theta = result.theta_rad

    issues: list[ValidationIssue] = []
    for bone, frames in unreachable_frames_by_bone.items():
        flags = np.zeros(n_frames, dtype=bool)
        flags[frames] = True
        for start, end in contiguous_true_runs(flags):
            issues.append(
                ValidationIssue(type="ik_target_unreachable", joint=bone, frame_range=(start, end), severity="error")
            )

    issues.extend(
        check_joint_velocities(
            skeleton, dof_names, skeleton_dof_joint_names(skeleton), joint_angles_rad, frame_rate
        )
    )

    return RetargetResult(
        joint_angles_rad=joint_angles_rad, validation_issues=issues, iterations_per_frame=iterations_per_frame
    )


def retarget_clip(clip: Clip, skeleton: SkeletonSpec, **solver_kwargs) -> Clip:
    """Non-mutating - returns a new Clip, matching parse_clip_yaml/open_clip/save_clip's
    functional style. Converts frame='world' end-effector curves to root-relative
    before solving. `channels.end_effector_targets`-marked 'derived' curves (the
    default when a bone has no entry) get the contact-locked curve persisted back;
    'authored' curves are solved-against internally but left untouched in the output -
    see this step's plan for why (Doc 4's contact-lock vs. Doc 2's authored/derived
    no-clobber contract)."""
    dof_names = ordered_dof_names(skeleton)
    n_frames = clip.tracks.root_pose.shape[0]

    root_relative_targets: dict[str, np.ndarray] = {}
    for ee in clip.tracks.end_effectors:
        if ee.frame == "world":
            root_relative_targets[ee.name] = world_to_root_relative(ee.target, clip.tracks.root_pose)
        else:
            root_relative_targets[ee.name] = ee.target

    contact = {c.name: c.contact for c in clip.tracks.contact}

    result = retarget_tracks(
        skeleton, dof_names, n_frames, root_relative_targets, contact, clip.header.clip.frame_rate, **solver_kwargs
    )

    contact_capable_bones = {bone.name for bone in skeleton.bones if bone.contact_capable}
    locked_root_relative = lock_contacts(root_relative_targets, contact, contact_capable_bones)

    new_end_effectors = []
    for ee in clip.tracks.end_effectors:
        state = clip.header.channels.end_effector_targets.get(ee.name, "derived")
        if state != "derived":
            new_end_effectors.append(ee)
            continue
        locked_local = locked_root_relative[ee.name]
        new_target = root_relative_to_world(locked_local, clip.tracks.root_pose) if ee.frame == "world" else locked_local
        new_end_effectors.append(EndEffectorTrack(name=ee.name, target=new_target, frame=ee.frame))

    new_tracks = TracksData(
        root_pose=clip.tracks.root_pose,
        joint_angles=result.joint_angles_rad,
        joint_angles_names=dof_names,
        contact=clip.tracks.contact,
        phase_variable=clip.tracks.phase_variable,
        root_velocity=clip.tracks.root_velocity,
        joint_velocities=clip.tracks.joint_velocities,
        end_effectors=new_end_effectors,
    )

    new_channels = ChannelsBlock(
        root_pose=clip.header.channels.root_pose,
        joint_angles="derived",
        end_effector_targets=dict(clip.header.channels.end_effector_targets),
        contact_state=clip.header.channels.contact_state,
    )

    if any(issue.severity == "error" for issue in result.validation_issues):
        status = "error"
    elif result.validation_issues:
        status = "warning"
    else:
        status = "ok"
    new_validation = ValidationBlock(status=status, issues=result.validation_issues)

    new_header = ClipHeader(
        clip=clip.header.clip,
        provenance=clip.header.provenance,
        skeleton=clip.header.skeleton,
        channels=new_channels,
        validation=new_validation,
    )

    return Clip(header=new_header, tracks=new_tracks)
