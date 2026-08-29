"""Serializes a ClipHeader back to `clip.yaml` - the inverse of clip_yaml_parser.py.

Independently round-trip tested (parse(save(header)) == header), mirroring
env_schema/serializer.py's relationship to env_schema/parser.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from coppelia_rl.motion.schema import ClipHeader


def _validation_issue_to_dict(issue) -> dict:
    return {
        "type": issue.type,
        "joint": issue.joint,
        "frame_range": list(issue.frame_range) if issue.frame_range is not None else None,
        "severity": issue.severity,
        "message": issue.message,
    }


def _clip_header_to_dict(header: ClipHeader, skeleton_target: str | Path | None = None) -> dict:
    return {
        "clip": {
            "name": header.clip.name,
            "format_version": header.clip.format_version,
            "frame_rate": header.clip.frame_rate,
            "duration_s": header.clip.duration_s,
            "loop": header.clip.loop,
        },
        "provenance": {
            "source_type": header.provenance.source_type,
            "source_ref": header.provenance.source_ref,
            "authoring_tool": header.provenance.authoring_tool,
            "created_at": header.provenance.created_at,
            "notes": header.provenance.notes,
        },
        "skeleton": {
            "target": str(skeleton_target if skeleton_target is not None else header.skeleton.target),
            "morphology_class": header.skeleton.morphology_class,
            "retargeted_from": str(header.skeleton.retargeted_from)
            if header.skeleton.retargeted_from is not None
            else None,
        },
        "channels": {
            "root_pose": header.channels.root_pose,
            "joint_angles": header.channels.joint_angles,
            "end_effector_targets": dict(header.channels.end_effector_targets),
            "contact_state": header.channels.contact_state,
        },
        "validation": {
            "status": header.validation.status,
            "issues": [_validation_issue_to_dict(issue) for issue in header.validation.issues],
        },
    }


def clip_header_to_yaml(header: ClipHeader, skeleton_target: str | Path | None = None) -> str:
    """Renders `header` as a YAML document string.

    `skeleton_target` overrides `header.skeleton.target` in the output (e.g. to write a
    path relative to where the caller is about to save the file); defaults to
    `header.skeleton.target` (always absolute, since that's what parse_clip_yaml produces).
    """
    doc = _clip_header_to_dict(header, skeleton_target=skeleton_target)
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def save_clip_yaml(header: ClipHeader, path: str | Path, skeleton_target: str | Path | None = None) -> None:
    Path(path).write_text(clip_header_to_yaml(header, skeleton_target=skeleton_target), encoding="utf-8")
