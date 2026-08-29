"""Dataclasses mirroring `skeletons/*.yaml` (see specs/skeleton-schema-and-frame-convention-SPEC.md).

The parser (parser.py) produces these from YAML. Kept as plain dataclasses, same
convention as env_schema/spec.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContactGeometrySpec:
    type: str  # sphere | box
    radius: float | None = None  # sphere
    size: tuple[float, float, float] | None = None  # box
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class JointSpec:
    name: str
    type: str  # revolute_1dof | revolute_2dof | revolute_3dof | prismatic | fixed
    axes: list[str] = field(default_factory=list)  # pitch/roll/yaw, or 1 translation axis for prismatic
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)  # degrees, keyed by axis
    rest_angle: dict[str, float] = field(default_factory=dict)  # degrees, keyed by axis; missing -> 0.0
    max_velocity: float | None = None  # rad/s
    max_torque: float | None = None  # Nm


@dataclass
class BoneSpec:
    name: str
    parent: str | None
    is_root: bool = False
    # Rest-pose translation (meters) from the parent bone's origin to this bone's
    # origin, in the parent's local frame at zero rotation - static bind-pose
    # geometry used by forward kinematics. Distinct from ContactGeometrySpec.offset
    # below, which is a point local to *this* bone, not a parent-to-bone translation.
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint: JointSpec | None = None  # None only for the root bone
    is_end_effector: bool = False
    contact_capable: bool = False
    contact_geometry: ContactGeometrySpec | None = None


@dataclass
class GaitPresetSpec:
    phase_offset: dict[str, float]  # end-effector bone name -> 0..1
    duty_factor: float


@dataclass
class GaitDefaultsSpec:
    limb_groups: dict[str, list[str]] = field(default_factory=dict)  # group name -> end-effector bone names
    presets: dict[str, GaitPresetSpec] = field(default_factory=dict)  # preset name -> preset


@dataclass
class FallDetectionSpec:
    type: str  # root_height_and_orientation
    min_root_height: float
    max_tilt_deg: float


@dataclass
class SkeletonSpec:
    name: str
    morphology_class: str  # biped | quadruped | arm_only | custom
    up_axis: str
    forward_axis: str
    root_bone: str  # flattened from `root.bone`
    bones: list[BoneSpec]
    end_effectors: list[str]  # bone names
    format_version: int = 1
    units: str = "meters"
    gait_defaults: GaitDefaultsSpec | None = None
    fall_detection: FallDetectionSpec | None = None
