"""Parses `skeletons/*.yaml` documents into SkeletonSpec.

See specs/skeleton-schema-and-frame-convention-SPEC.md for the format this parses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from coppelia_rl.skeleton.schema import (
    BoneSpec,
    ContactGeometrySpec,
    FallDetectionSpec,
    GaitDefaultsSpec,
    GaitPresetSpec,
    JointSpec,
    SkeletonSpec,
)

_AXES_COUNT_BY_TYPE = {
    "revolute_1dof": 1,
    "revolute_2dof": 2,
    "revolute_3dof": 3,
    "prismatic": 1,
    "fixed": 0,
}


class SkeletonValidationError(ValueError):
    pass


def _as_tuple2(value) -> tuple[float, float]:
    lo, hi = value
    return float(lo), float(hi)


def _as_tuple3(value) -> tuple[float, float, float]:
    x, y, z = value
    return float(x), float(y), float(z)


def _parse_contact_geometry(raw: dict) -> ContactGeometrySpec:
    return ContactGeometrySpec(
        type=raw["type"],
        radius=float(raw["radius"]) if "radius" in raw else None,
        size=_as_tuple3(raw["size"]) if "size" in raw else None,
        offset=_as_tuple3(raw["offset"]) if "offset" in raw else (0.0, 0.0, 0.0),
    )


def _parse_joint(raw: dict) -> JointSpec:
    return JointSpec(
        name=raw["name"],
        type=raw["type"],
        axes=list(raw.get("axes", [])),
        limits={k: _as_tuple2(v) for k, v in raw.get("limits", {}).items()},
        rest_angle={k: float(v) for k, v in raw.get("rest_angle", {}).items()},
        max_velocity=float(raw["max_velocity"]) if "max_velocity" in raw else None,
        max_torque=float(raw["max_torque"]) if "max_torque" in raw else None,
    )


def _parse_bone(raw: dict) -> BoneSpec:
    return BoneSpec(
        name=raw["name"],
        parent=raw.get("parent"),
        is_root=bool(raw.get("is_root", False)),
        offset=_as_tuple3(raw["offset"]) if "offset" in raw else (0.0, 0.0, 0.0),
        joint=_parse_joint(raw["joint"]) if "joint" in raw else None,
        is_end_effector=bool(raw.get("is_end_effector", False)),
        contact_capable=bool(raw.get("contact_capable", False)),
        contact_geometry=_parse_contact_geometry(raw["contact_geometry"]) if "contact_geometry" in raw else None,
    )


def _parse_gait_defaults(raw: dict) -> GaitDefaultsSpec:
    presets = {
        preset_name: GaitPresetSpec(
            phase_offset={k: float(v) for k, v in preset_raw["phase_offset"].items()},
            duty_factor=float(preset_raw["duty_factor"]),
        )
        for preset_name, preset_raw in raw.get("presets", {}).items()
    }
    limb_groups = {group_name: list(bones) for group_name, bones in raw.get("limb_groups", {}).items()}
    return GaitDefaultsSpec(limb_groups=limb_groups, presets=presets)


def _parse_fall_detection(raw: dict) -> FallDetectionSpec:
    return FallDetectionSpec(
        type=raw["type"],
        min_root_height=float(raw["min_root_height"]),
        max_tilt_deg=float(raw["max_tilt_deg"]),
    )


def parse_skeleton_yaml(path: str | Path) -> SkeletonSpec:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    skeleton_raw = doc["skeleton"]
    bones = [_parse_bone(b) for b in doc["bones"]]

    spec = SkeletonSpec(
        name=skeleton_raw["name"],
        format_version=int(skeleton_raw.get("format_version", 1)),
        morphology_class=skeleton_raw["morphology_class"],
        up_axis=skeleton_raw["up_axis"],
        forward_axis=skeleton_raw["forward_axis"],
        units=skeleton_raw.get("units", "meters"),
        root_bone=doc["root"]["bone"],
        bones=bones,
        end_effectors=list(doc["end_effectors"]),
        gait_defaults=_parse_gait_defaults(doc["gait_defaults"]) if "gait_defaults" in doc else None,
        fall_detection=_parse_fall_detection(doc["fall_detection"]) if "fall_detection" in doc else None,
    )
    _validate(spec)
    return spec


def _validate(spec: SkeletonSpec) -> None:
    bones_by_name = {bone.name: bone for bone in spec.bones}
    if len(bones_by_name) != len(spec.bones):
        raise SkeletonValidationError("Duplicate bone name in `bones`")

    for bone in spec.bones:
        if bone.parent is not None and bone.parent not in bones_by_name:
            raise SkeletonValidationError(f"Bone {bone.name!r} references unknown parent {bone.parent!r}")

    root_bones = [bone for bone in spec.bones if bone.is_root]
    if len(root_bones) != 1:
        raise SkeletonValidationError(f"Expected exactly one bone with is_root: true, found {len(root_bones)}")
    if root_bones[0].name != spec.root_bone:
        raise SkeletonValidationError(
            f"root.bone {spec.root_bone!r} does not match the bone marked is_root: true ({root_bones[0].name!r})"
        )

    for ee_name in spec.end_effectors:
        bone = bones_by_name.get(ee_name)
        if bone is None:
            raise SkeletonValidationError(f"end_effectors references unknown bone {ee_name!r}")
        if not bone.is_end_effector:
            raise SkeletonValidationError(f"end_effectors references bone {ee_name!r} not marked is_end_effector: true")

    for bone in spec.bones:
        if bone.joint is None:
            continue
        joint = bone.joint
        expected_axes = _AXES_COUNT_BY_TYPE.get(joint.type)
        if expected_axes is None:
            raise SkeletonValidationError(f"Bone {bone.name!r} has unknown joint type {joint.type!r}")
        if len(joint.axes) != expected_axes:
            raise SkeletonValidationError(
                f"Bone {bone.name!r} joint type {joint.type!r} expects {expected_axes} axes, got {joint.axes!r}"
            )
        if joint.type == "fixed" and joint.limits:
            raise SkeletonValidationError(f"Bone {bone.name!r} has type: fixed but declares limits")
        if set(joint.limits.keys()) != set(joint.axes):
            raise SkeletonValidationError(
                f"Bone {bone.name!r} joint limits keys {set(joint.limits.keys())} do not match axes {set(joint.axes)}"
            )
        if joint.type == "fixed" and joint.rest_angle:
            raise SkeletonValidationError(f"Bone {bone.name!r} has type: fixed but declares rest_angle")
        if not set(joint.rest_angle.keys()) <= set(joint.axes):
            raise SkeletonValidationError(
                f"Bone {bone.name!r} joint rest_angle keys {set(joint.rest_angle.keys())} "
                f"are not a subset of axes {set(joint.axes)}"
            )
        for axis, rest_value in joint.rest_angle.items():
            lo, hi = joint.limits[axis]
            if not (lo <= rest_value <= hi):
                raise SkeletonValidationError(
                    f"Bone {bone.name!r} joint rest_angle[{axis!r}]={rest_value} is outside limits {(lo, hi)}"
                )

        if bone.contact_capable and bone.contact_geometry is None:
            raise SkeletonValidationError(f"Bone {bone.name!r} is contact_capable but declares no contact_geometry")
        if bone.contact_geometry is not None:
            geom = bone.contact_geometry
            if geom.type == "sphere" and geom.radius is None:
                raise SkeletonValidationError(f"Bone {bone.name!r} contact_geometry type sphere requires radius")
            if geom.type == "box" and geom.size is None:
                raise SkeletonValidationError(f"Bone {bone.name!r} contact_geometry type box requires size")

    if spec.gait_defaults is not None:
        end_effector_set = set(spec.end_effectors)
        for group_name, group_bones in spec.gait_defaults.limb_groups.items():
            for bone_name in group_bones:
                if bone_name not in end_effector_set:
                    raise SkeletonValidationError(
                        f"gait_defaults.limb_groups[{group_name!r}] references {bone_name!r}, not in end_effectors"
                    )
        for preset_name, preset in spec.gait_defaults.presets.items():
            for bone_name in preset.phase_offset:
                if bone_name not in end_effector_set:
                    raise SkeletonValidationError(
                        f"gait_defaults.presets[{preset_name!r}].phase_offset references {bone_name!r}, "
                        "not in end_effectors"
                    )


def dof_name(joint: JointSpec, axis: str) -> str:
    """Per-DOF-axis name for one axis of `joint`: the bare joint.name if the
    joint has a single axis, else f"{joint.name}_{axis}" (e.g. FL_hip_pitch,
    FL_hip_roll) - see the "Spec clarification applied" note in this extension's
    build plan for why per-axis names are used instead of one name per joint.
    """
    return joint.name if len(joint.axes) == 1 else f"{joint.name}_{axis}"


def ordered_dof_names(skeleton: SkeletonSpec) -> list[str]:
    """Ground truth for tracks.h5's joint_angles_names / J.

    Walks skeleton.bones in file order, skips the root bone (no joint) and any
    type: fixed joint (0 DOF), and emits dof_name(joint, axis) for each axis.
    """
    names: list[str] = []
    for bone in skeleton.bones:
        joint = bone.joint
        if joint is None or joint.type == "fixed":
            continue
        names.extend(dof_name(joint, axis) for axis in joint.axes)
    return names


def dof_joint_names(skeleton: SkeletonSpec) -> list[str]:
    """Parallel array to ordered_dof_names(): dof_joint_names(spec)[k] is the
    bare JointSpec.name owning ordered_dof_names(spec)[k] - e.g. for
    FL_hip_pitch/FL_hip_roll/FL_knee this is [FL_hip, FL_hip, FL_knee]. Used to
    report per-joint (not per-DOF-axis) validation issues, e.g. joint-velocity
    limit checks, since max_velocity is declared once per joint.
    """
    names: list[str] = []
    for bone in skeleton.bones:
        joint = bone.joint
        if joint is None or joint.type == "fixed":
            continue
        names.extend(joint.name for _ in joint.axes)
    return names
