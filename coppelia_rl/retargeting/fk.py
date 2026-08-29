"""Forward kinematics for `skeletons/*.yaml` skeletons.

Rotation-axis convention (not specified by the skeleton-schema/retargeting specs -
resolved here since Stage B's IK solve needs it): standard aircraft body-axis
convention. `roll` = rotation about `skeleton.forward_axis`, `yaw` = rotation about
`skeleton.up_axis`, `pitch` = rotation about `lateral_axis = normalize(up_axis x
forward_axis)` (right-hand rule). A joint's `axes: [...]` compose in listed order as
intrinsic (local-frame) rotations. This only needs to be internally consistent (FK
and IK must agree, limits must clamp in the matching direction) - it is not required
to match any real anatomical convention.

Per-bone transform: `T_bone = T_parent @ Translate(bone.offset) @ R_axis1(a1) @
R_axis2(a2) @ ...`. The root bone's transform is always the identity - Stage B never
solves root pose, and `BoneSpec.offset` on the root bone is parsed but unused here.

All angles taken/returned by this module are radians. `skeletons/*.yaml`'s
`limits`/`rest_angle` are authored in degrees (per the schema spec's own examples) and
convert to radians at the dof_bounds_rad/dof_rest_angles_rad boundary; `max_velocity`
is already rad/s in the schema and is passed through unconverted.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from coppelia_rl.skeleton.parser import dof_name
from coppelia_rl.skeleton.schema import JointSpec, SkeletonSpec

_WORLD_AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _resolve_up_forward_lateral(up_axis: str, forward_axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = _WORLD_AXIS_VECTORS[up_axis]
    forward = _WORLD_AXIS_VECTORS[forward_axis]
    lateral = np.cross(up, forward)
    norm = np.linalg.norm(lateral)
    if norm < 1e-9:
        raise ValueError(f"up_axis {up_axis!r} and forward_axis {forward_axis!r} must be distinct")
    return up, forward, lateral / norm


def _named_axis_vector(axis_name: str, up_axis: str, forward_axis: str) -> np.ndarray:
    if axis_name in _WORLD_AXIS_VECTORS:  # prismatic joints name a literal x/y/z axis
        return _WORLD_AXIS_VECTORS[axis_name]
    up, forward, lateral = _resolve_up_forward_lateral(up_axis, forward_axis)
    if axis_name == "roll":
        return forward
    if axis_name == "yaw":
        return up
    if axis_name == "pitch":
        return lateral
    raise ValueError(f"Unknown joint axis name {axis_name!r}")


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula - 3x3 rotation about a unit `axis` by `angle_rad`."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    comp = 1.0 - c
    return np.array(
        [
            [x * x * comp + c, x * y * comp - z * s, x * z * comp + y * s],
            [y * x * comp + z * s, y * y * comp + c, y * z * comp - x * s],
            [z * x * comp - y * s, z * y * comp + x * s, z * z * comp + c],
        ]
    )


def _translate(offset) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = offset
    return t


def _to_4x4(rot3x3: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot3x3
    return t


def forward_kinematics(skeleton: SkeletonSpec, dof_angles_rad: Mapping[str, float]) -> dict[str, np.ndarray]:
    """bone name -> 4x4 root-relative transform. Missing `dof_angles_rad` keys
    default to 0.0. Resolves via memoized recursion on bone.parent (does not assume
    skeletons/*.yaml lists bones in parent-before-child order), with a visiting-set
    cycle guard.
    """
    bones_by_name = {bone.name: bone for bone in skeleton.bones}
    transforms: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def resolve(bone_name: str) -> np.ndarray:
        if bone_name in transforms:
            return transforms[bone_name]
        bone = bones_by_name[bone_name]
        if bone.is_root:
            transforms[bone_name] = np.eye(4)
            return transforms[bone_name]
        if bone_name in visiting:
            raise ValueError(f"Cycle detected in bone graph at {bone_name!r}")
        visiting.add(bone_name)

        parent_transform = resolve(bone.parent) if bone.parent is not None else np.eye(4)
        local = _translate(bone.offset)
        joint = bone.joint
        if joint is not None and joint.type != "fixed":
            for axis in joint.axes:
                angle = dof_angles_rad.get(dof_name(joint, axis), 0.0)
                axis_vec = _named_axis_vector(axis, skeleton.up_axis, skeleton.forward_axis)
                local = local @ _to_4x4(_rotation_matrix(axis_vec, angle))

        transform = parent_transform @ local
        visiting.discard(bone_name)
        transforms[bone_name] = transform
        return transform

    for bone in skeleton.bones:
        resolve(bone.name)
    return transforms


def end_effector_position(skeleton: SkeletonSpec, bone_transforms: dict[str, np.ndarray], bone_name: str) -> np.ndarray:
    """(3,) - bone origin + contact_geometry.offset if present, else bone origin."""
    bone = next(b for b in skeleton.bones if b.name == bone_name)
    local_point = np.array(bone.contact_geometry.offset) if bone.contact_geometry is not None else np.zeros(3)
    world_point = bone_transforms[bone_name] @ np.append(local_point, 1.0)
    return world_point[:3]


def effector_positions(
    skeleton: SkeletonSpec, dof_names: list[str], theta_rad: np.ndarray, bone_names: list[str]
) -> np.ndarray:
    """(len(bone_names), 3) - one FK pass, then position extraction for each bone."""
    transforms = forward_kinematics(skeleton, dict(zip(dof_names, theta_rad)))
    return np.array([end_effector_position(skeleton, transforms, name) for name in bone_names])


def _dof_axis_lookup(skeleton: SkeletonSpec) -> dict[str, tuple[JointSpec, str]]:
    lookup: dict[str, tuple[JointSpec, str]] = {}
    for bone in skeleton.bones:
        joint = bone.joint
        if joint is None or joint.type == "fixed":
            continue
        for axis in joint.axes:
            lookup[dof_name(joint, axis)] = (joint, axis)
    return lookup


def dof_bounds_rad(skeleton: SkeletonSpec, dof_names: list[str]) -> np.ndarray:
    lookup = _dof_axis_lookup(skeleton)
    bounds = np.zeros((len(dof_names), 2))
    for i, name in enumerate(dof_names):
        joint, axis = lookup[name]
        lo, hi = joint.limits[axis]
        bounds[i] = (np.radians(lo), np.radians(hi))
    return bounds


def dof_rest_angles_rad(skeleton: SkeletonSpec, dof_names: list[str]) -> np.ndarray:
    lookup = _dof_axis_lookup(skeleton)
    values = np.zeros(len(dof_names))
    for i, name in enumerate(dof_names):
        joint, axis = lookup[name]
        values[i] = np.radians(joint.rest_angle.get(axis, 0.0))
    return values


def dof_max_velocity_rad_per_s(skeleton: SkeletonSpec, dof_names: list[str]) -> np.ndarray:
    lookup = _dof_axis_lookup(skeleton)
    values = np.full(len(dof_names), np.inf)
    for i, name in enumerate(dof_names):
        joint, _axis = lookup[name]
        if joint.max_velocity is not None:
            values[i] = joint.max_velocity
    return values


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a [qw,qx,qy,qz] (scalar-first) quaternion - the
    convention tracks.h5's /root_pose uses (motion-representation-format-SPEC.md)."""
    qw, qx, qy, qz = q
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def world_to_root_relative(world_curve: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    """(N,3) world-frame positions -> (N,3) root-relative, given (N,7) root_pose
    [x,y,z,qw,qx,qy,qz]."""
    out = np.zeros_like(world_curve)
    for i in range(world_curve.shape[0]):
        rot = quat_to_matrix(root_pose[i, 3:])
        out[i] = rot.T @ (world_curve[i] - root_pose[i, :3])
    return out


def root_relative_to_world(local_curve: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    """Inverse of world_to_root_relative."""
    out = np.zeros_like(local_curve)
    for i in range(local_curve.shape[0]):
        rot = quat_to_matrix(root_pose[i, 3:])
        out[i] = rot @ local_curve[i] + root_pose[i, :3]
    return out
