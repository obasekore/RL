"""Parses `clip.yaml` documents into ClipHeader (see specs/motion-representation-format-SPEC.md)."""

from __future__ import annotations

import datetime
from pathlib import Path

import yaml

from coppelia_rl.motion.schema import (
    ChannelsBlock,
    ClipHeader,
    ClipMeta,
    Provenance,
    SkeletonBinding,
    ValidationBlock,
    ValidationIssue,
)

_LOOP_VALUES = {"cyclic", "one_shot", "ping_pong"}
_SOURCE_TYPES = {"mocap", "choreographe_native", "create", "edited"}
_MORPHOLOGY_CLASSES = {"biped", "quadruped", "arm_only", "custom"}
_CHANNEL_STATES = {"authored", "derived"}
_VALIDATION_STATUSES = {"ok", "warning", "error"}
_ISSUE_SEVERITIES = {"warning", "error"}


class ClipYamlValidationError(ValueError):
    pass


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ClipYamlValidationError(f"Missing required field {key!r} in {context}")
    return mapping[key]


def _normalize_created_at(value) -> str | None:
    """PyYAML's safe_load implicitly parses an unquoted ISO8601 timestamp (e.g.
    `2026-08-20T00:00:00Z`, per Doc 2's own clip.yaml example) into a
    datetime.datetime/date rather than leaving it as a str. Convert back to a
    string so ClipHeader.created_at is always `str | None` regardless of
    whether the source file quoted the value.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None and value.utcoffset() == datetime.timedelta(0):
            return value.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def _resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    return (base_dir / raw).resolve()


def _parse_validation(raw: dict) -> ValidationBlock:
    issues = [
        ValidationIssue(
            type=_require(issue, "type", "validation.issues[]"),
            joint=issue.get("joint"),
            frame_range=tuple(issue["frame_range"]) if issue.get("frame_range") is not None else None,
            severity=issue.get("severity", "warning"),
            message=issue.get("message"),
        )
        for issue in raw.get("issues", [])
    ]
    return ValidationBlock(status=raw.get("status", "ok"), issues=issues)


def parse_clip_yaml(path: str | Path) -> ClipHeader:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    clip_raw = _require(doc, "clip", "document")
    provenance_raw = _require(doc, "provenance", "document")
    skeleton_raw = _require(doc, "skeleton", "document")
    channels_raw = _require(doc, "channels", "document")
    validation_raw = doc.get("validation")

    header = ClipHeader(
        clip=ClipMeta(
            name=_require(clip_raw, "name", "clip"),
            format_version=int(_require(clip_raw, "format_version", "clip")),
            frame_rate=float(_require(clip_raw, "frame_rate", "clip")),
            duration_s=float(_require(clip_raw, "duration_s", "clip")),
            loop=_require(clip_raw, "loop", "clip"),
        ),
        provenance=Provenance(
            source_type=_require(provenance_raw, "source_type", "provenance"),
            source_ref=provenance_raw.get("source_ref"),
            authoring_tool=provenance_raw.get("authoring_tool"),
            created_at=_normalize_created_at(provenance_raw.get("created_at")),
            notes=provenance_raw.get("notes"),
        ),
        skeleton=SkeletonBinding(
            target=(path.parent / _require(skeleton_raw, "target", "skeleton")).resolve(),
            morphology_class=_require(skeleton_raw, "morphology_class", "skeleton"),
            retargeted_from=_resolve_optional_path(path.parent, skeleton_raw.get("retargeted_from")),
        ),
        channels=ChannelsBlock(
            root_pose=_require(channels_raw, "root_pose", "channels"),
            joint_angles=_require(channels_raw, "joint_angles", "channels"),
            end_effector_targets=dict(channels_raw.get("end_effector_targets", {})),
            contact_state=channels_raw.get("contact_state", "authored"),
        ),
        validation=_parse_validation(validation_raw) if validation_raw is not None else ValidationBlock(),
    )
    _validate(header)
    return header


def _validate(header: ClipHeader) -> None:
    if header.clip.loop not in _LOOP_VALUES:
        raise ClipYamlValidationError(f"clip.loop must be one of {sorted(_LOOP_VALUES)}, got {header.clip.loop!r}")
    if header.provenance.source_type not in _SOURCE_TYPES:
        raise ClipYamlValidationError(
            f"provenance.source_type must be one of {sorted(_SOURCE_TYPES)}, got {header.provenance.source_type!r}"
        )
    if header.skeleton.morphology_class not in _MORPHOLOGY_CLASSES:
        raise ClipYamlValidationError(
            f"skeleton.morphology_class must be one of {sorted(_MORPHOLOGY_CLASSES)}, "
            f"got {header.skeleton.morphology_class!r}"
        )
    for field_name, value in (
        ("root_pose", header.channels.root_pose),
        ("joint_angles", header.channels.joint_angles),
        ("contact_state", header.channels.contact_state),
    ):
        if value not in _CHANNEL_STATES:
            raise ClipYamlValidationError(
                f"channels.{field_name} must be one of {sorted(_CHANNEL_STATES)}, got {value!r}"
            )
    for bone_name, value in header.channels.end_effector_targets.items():
        if value not in _CHANNEL_STATES:
            raise ClipYamlValidationError(
                f"channels.end_effector_targets[{bone_name!r}] must be one of {sorted(_CHANNEL_STATES)}, "
                f"got {value!r}"
            )
    if header.validation.status not in _VALIDATION_STATUSES:
        raise ClipYamlValidationError(
            f"validation.status must be one of {sorted(_VALIDATION_STATUSES)}, got {header.validation.status!r}"
        )
    for issue in header.validation.issues:
        if issue.severity not in _ISSUE_SEVERITIES:
            raise ClipYamlValidationError(
                f"validation.issues[].severity must be one of {sorted(_ISSUE_SEVERITIES)}, got {issue.severity!r}"
            )
