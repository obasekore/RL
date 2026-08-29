"""Bundles clip.yaml + tracks.h5 into one `Clip` (Doc 2's "one clip = one directory"
file layout), plus cross-file validation that needs both files and the target
skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coppelia_rl.motion.clip_yaml_parser import parse_clip_yaml
from coppelia_rl.motion.clip_yaml_serializer import save_clip_yaml
from coppelia_rl.motion.schema import ClipHeader
from coppelia_rl.motion.tracks_hdf5 import TracksData, read_tracks, write_tracks
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml
from coppelia_rl.skeleton.schema import SkeletonSpec

_CLIP_YAML_FILENAME = "clip.yaml"
_TRACKS_H5_FILENAME = "tracks.h5"


@dataclass
class Clip:
    header: ClipHeader
    tracks: TracksData


class ClipValidationError(ValueError):
    pass


def validate_clip_consistency(header: ClipHeader, tracks: TracksData, skeleton: SkeletonSpec) -> None:
    """Cross-file checks that need both clip.yaml and tracks.h5 (plus the target skeleton).

    Per-file structural validation (enum values, mandatory channels, per-array
    shapes) already happened in clip_yaml_parser.parse_clip_yaml and
    tracks_hdf5.read_tracks/write_tracks - this only covers checks that span files.
    """
    expected_names = ordered_dof_names(skeleton)
    if tracks.joint_angles_names != expected_names:
        raise ClipValidationError(
            f"tracks.h5 joint_angles_names {tracks.joint_angles_names} does not match target "
            f"skeleton {skeleton.name!r}'s ordered DOF names {expected_names}"
        )

    expected_frame_count = round(header.clip.duration_s * header.clip.frame_rate)
    actual_frame_count = tracks.root_pose.shape[0]
    if actual_frame_count != expected_frame_count:
        raise ClipValidationError(
            f"tracks.h5 frame count {actual_frame_count} does not match "
            f"round(duration_s * frame_rate) = round({header.clip.duration_s} * {header.clip.frame_rate}) "
            f"= {expected_frame_count}"
        )


def open_clip(clip_dir: str | Path, *, skeleton: SkeletonSpec | None = None, validate: bool = True) -> Clip:
    clip_dir = Path(clip_dir)
    header = parse_clip_yaml(clip_dir / _CLIP_YAML_FILENAME)
    tracks = read_tracks(clip_dir / _TRACKS_H5_FILENAME)
    if validate:
        resolved_skeleton = skeleton if skeleton is not None else parse_skeleton_yaml(header.skeleton.target)
        validate_clip_consistency(header, tracks, resolved_skeleton)
    return Clip(header=header, tracks=tracks)


def save_clip(
    clip_dir: str | Path, clip: Clip, *, skeleton: SkeletonSpec | None = None, validate: bool = True
) -> None:
    clip_dir = Path(clip_dir)
    if validate:
        resolved_skeleton = skeleton if skeleton is not None else parse_skeleton_yaml(clip.header.skeleton.target)
        validate_clip_consistency(clip.header, clip.tracks, resolved_skeleton)
    clip_dir.mkdir(parents=True, exist_ok=True)
    save_clip_yaml(clip.header, clip_dir / _CLIP_YAML_FILENAME)
    write_tracks(clip_dir / _TRACKS_H5_FILENAME, clip.tracks)
