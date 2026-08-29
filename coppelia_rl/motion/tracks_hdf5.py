"""Reads/writes `tracks.h5`, the per-frame payload half of a motion clip
(see specs/motion-representation-format-SPEC.md).

Unlike coppelia_rl/demos/hdf5_writer.py's episode-indexed, incrementally-appended
layout, a clip's tracks.h5 is one complete payload written once - hence a single
file (no reader/writer split) opened with mode "w", not "a".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

_FORMAT_VERSION = 1
_FORCE_HINT_SUFFIX = "_force_hint"


@dataclass
class EndEffectorTrack:
    name: str
    target: np.ndarray  # (N, 3) float32, world or root-relative per `frame`
    frame: str  # "world" | "root_relative"


@dataclass
class ContactTrack:
    name: str
    contact: np.ndarray  # (N,) bool
    force_hint: np.ndarray | None = None  # (N,) float32, optional


@dataclass
class TracksData:
    root_pose: np.ndarray  # (N, 7) float32  [x,y,z, qw,qx,qy,qz]
    joint_angles: np.ndarray  # (N, J) float32
    joint_angles_names: list[str]  # length J, ordered per skeleton.parser.ordered_dof_names()
    contact: list[ContactTrack]  # mandatory, at least one channel
    phase_variable: np.ndarray  # (N,) float32
    root_velocity: np.ndarray | None = None  # (N, 6) float32, optional/derived
    joint_velocities: np.ndarray | None = None  # (N, J) float32, optional/derived
    end_effectors: list[EndEffectorTrack] = field(default_factory=list)  # optional at storage level


class TracksValidationError(ValueError):
    pass


def _require_dataset(f: h5py.File, name: str) -> None:
    if name not in f:
        raise TracksValidationError(f"tracks.h5 missing mandatory dataset /{name}")


def _require_group(f: h5py.File, name: str) -> None:
    if name not in f:
        raise TracksValidationError(f"tracks.h5 missing mandatory group /{name}")


def _validate_tracks(tracks: TracksData) -> None:
    n = tracks.root_pose.shape[0]
    j = len(tracks.joint_angles_names)

    if tracks.root_pose.ndim != 2 or tracks.root_pose.shape[1] != 7:
        raise TracksValidationError(f"root_pose must have shape (N, 7), got {tracks.root_pose.shape}")
    if tracks.joint_angles.shape != (n, j):
        raise TracksValidationError(
            f"joint_angles must have shape (N, J)=({n}, {j}) matching joint_angles_names, "
            f"got {tracks.joint_angles.shape}"
        )
    if not tracks.contact:
        raise TracksValidationError("tracks.contact must be non-empty - at least one contact channel is mandatory")
    for c in tracks.contact:
        if c.contact.shape != (n,):
            raise TracksValidationError(f"contact[{c.name!r}] must have shape (N,)=({n},), got {c.contact.shape}")
        if c.force_hint is not None and c.force_hint.shape != (n,):
            raise TracksValidationError(
                f"contact[{c.name!r}].force_hint must have shape (N,)=({n},), got {c.force_hint.shape}"
            )
    if tracks.phase_variable.shape != (n,):
        raise TracksValidationError(f"phase_variable must have shape (N,)=({n},), got {tracks.phase_variable.shape}")
    if tracks.root_velocity is not None and tracks.root_velocity.shape != (n, 6):
        raise TracksValidationError(
            f"root_velocity must have shape (N, 6)=({n}, 6), got {tracks.root_velocity.shape}"
        )
    if tracks.joint_velocities is not None and tracks.joint_velocities.shape != (n, j):
        raise TracksValidationError(
            f"joint_velocities must have shape (N, J)=({n}, {j}), got {tracks.joint_velocities.shape}"
        )
    for ee in tracks.end_effectors:
        if ee.target.shape != (n, 3):
            raise TracksValidationError(
                f"end_effectors[{ee.name!r}].target must have shape (N, 3)=({n}, 3), got {ee.target.shape}"
            )


def write_tracks(path: str | Path, tracks: TracksData) -> None:
    _validate_tracks(tracks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = _FORMAT_VERSION

        f.create_dataset("root_pose", data=np.asarray(tracks.root_pose, dtype=np.float32))
        if tracks.root_velocity is not None:
            f.create_dataset("root_velocity", data=np.asarray(tracks.root_velocity, dtype=np.float32))

        f.create_dataset("joint_angles", data=np.asarray(tracks.joint_angles, dtype=np.float32))
        f.create_dataset(
            "joint_angles_names", data=tracks.joint_angles_names, dtype=h5py.string_dtype(encoding="utf-8")
        )
        if tracks.joint_velocities is not None:
            f.create_dataset("joint_velocities", data=np.asarray(tracks.joint_velocities, dtype=np.float32))

        if tracks.end_effectors:
            ee_group = f.create_group("end_effectors")
            for ee in tracks.end_effectors:
                sub = ee_group.create_group(ee.name)
                sub.create_dataset("target", data=np.asarray(ee.target, dtype=np.float32))
                sub.create_dataset("frame", data=ee.frame, dtype=h5py.string_dtype(encoding="utf-8"))

        contact_group = f.create_group("contact")
        for c in tracks.contact:
            contact_group.create_dataset(c.name, data=np.asarray(c.contact, dtype=bool))
            if c.force_hint is not None:
                contact_group.create_dataset(
                    f"{c.name}{_FORCE_HINT_SUFFIX}", data=np.asarray(c.force_hint, dtype=np.float32)
                )

        f.create_dataset("phase_variable", data=np.asarray(tracks.phase_variable, dtype=np.float32))


def read_tracks(path: str | Path) -> TracksData:
    with h5py.File(path, "r") as f:
        _require_dataset(f, "root_pose")
        _require_dataset(f, "joint_angles")
        _require_dataset(f, "joint_angles_names")
        _require_dataset(f, "phase_variable")
        _require_group(f, "contact")

        joint_angles_names = list(f["joint_angles_names"].asstr()[:])

        contact_group = f["contact"]
        contact_names = [name for name in contact_group if not name.endswith(_FORCE_HINT_SUFFIX)]
        if not contact_names:
            raise TracksValidationError("tracks.h5 /contact group is empty - at least one contact channel is mandatory")
        contact = []
        for name in contact_names:
            force_hint_key = f"{name}{_FORCE_HINT_SUFFIX}"
            contact.append(
                ContactTrack(
                    name=name,
                    contact=np.asarray(contact_group[name][()], dtype=bool),
                    force_hint=(
                        np.asarray(contact_group[force_hint_key][()], dtype=np.float32)
                        if force_hint_key in contact_group
                        else None
                    ),
                )
            )

        end_effectors = []
        if "end_effectors" in f:
            ee_group = f["end_effectors"]
            for name in ee_group:
                sub = ee_group[name]
                end_effectors.append(
                    EndEffectorTrack(
                        name=name,
                        target=np.asarray(sub["target"][()], dtype=np.float32),
                        frame=sub["frame"].asstr()[()],
                    )
                )

        tracks = TracksData(
            root_pose=np.asarray(f["root_pose"][()], dtype=np.float32),
            joint_angles=np.asarray(f["joint_angles"][()], dtype=np.float32),
            joint_angles_names=joint_angles_names,
            contact=contact,
            phase_variable=np.asarray(f["phase_variable"][()], dtype=np.float32),
            root_velocity=(
                np.asarray(f["root_velocity"][()], dtype=np.float32) if "root_velocity" in f else None
            ),
            joint_velocities=(
                np.asarray(f["joint_velocities"][()], dtype=np.float32) if "joint_velocities" in f else None
            ),
            end_effectors=end_effectors,
        )

    _validate_tracks(tracks)
    return tracks
