"""Dataclasses mirroring `clip.yaml` (see specs/motion-representation-format-SPEC.md).

clip_yaml_parser.py produces these from YAML; clip_yaml_serializer.py is the inverse.
Kept as plain dataclasses, same convention as env_schema/spec.py and skeleton/schema.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClipMeta:
    name: str
    format_version: int
    frame_rate: float  # Hz
    duration_s: float
    loop: str  # cyclic | one_shot | ping_pong


@dataclass
class Provenance:
    source_type: str  # mocap | choreographe_native | create | edited
    source_ref: str | None = None
    authoring_tool: str | None = None
    created_at: str | None = None
    notes: str | None = None


@dataclass
class SkeletonBinding:
    target: Path  # abstract skeleton descriptor, resolved absolute
    morphology_class: str
    retargeted_from: Path | None = None  # null for Create-mode clips with no mocap source


@dataclass
class ChannelsBlock:
    root_pose: str  # authored | derived
    joint_angles: str  # authored | derived
    end_effector_targets: dict[str, str] = field(default_factory=dict)  # bone name -> authored|derived
    contact_state: str = "authored"  # authored | derived


@dataclass
class ValidationIssue:
    type: str  # e.g. joint_velocity_exceeded, ik_target_unreachable
    joint: str | None = None
    frame_range: tuple[int, int] | None = None
    severity: str = "warning"  # warning | error
    message: str | None = None


@dataclass
class ValidationBlock:
    status: str = "ok"  # ok | warning | error
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class ClipHeader:
    clip: ClipMeta
    provenance: Provenance
    skeleton: SkeletonBinding
    channels: ChannelsBlock
    validation: ValidationBlock = field(default_factory=ValidationBlock)
