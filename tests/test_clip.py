from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from coppelia_rl.motion.clip import Clip, ClipValidationError, open_clip, save_clip  # noqa: E402
from coppelia_rl.motion.schema import ChannelsBlock, ClipHeader, ClipMeta, Provenance, SkeletonBinding  # noqa: E402
from coppelia_rl.motion.tracks_hdf5 import ContactTrack, TracksData, TracksValidationError  # noqa: E402
from coppelia_rl.skeleton.parser import ordered_dof_names, parse_skeleton_yaml  # noqa: E402

_QUADRUPED_PATH = Path(__file__).resolve().parents[1] / "skeletons" / "quadruped_generic.yaml"
_SKELETON = parse_skeleton_yaml(_QUADRUPED_PATH)
_DOF_NAMES = ordered_dof_names(_SKELETON)
_N = 5
_FRAME_RATE = 10.0
_DURATION_S = _N / _FRAME_RATE  # 0.5, so round(duration_s * frame_rate) == N


def _make_header(**overrides) -> ClipHeader:
    defaults = dict(
        clip=ClipMeta(
            name="test_clip", format_version=1, frame_rate=_FRAME_RATE, duration_s=_DURATION_S, loop="cyclic"
        ),
        provenance=Provenance(source_type="create"),
        skeleton=SkeletonBinding(target=_QUADRUPED_PATH.resolve(), morphology_class="quadruped"),
        channels=ChannelsBlock(root_pose="authored", joint_angles="derived"),
    )
    defaults.update(overrides)
    return ClipHeader(**defaults)


def _make_tracks(*, n: int = _N, joint_angles_names: list[str] | None = None) -> TracksData:
    names = joint_angles_names if joint_angles_names is not None else list(_DOF_NAMES)
    rng = np.random.default_rng(0)
    return TracksData(
        root_pose=rng.random((n, 7), dtype=np.float32),
        joint_angles=rng.random((n, len(names)), dtype=np.float32),
        joint_angles_names=names,
        contact=[ContactTrack(name="FL_foot", contact=np.zeros(n, dtype=bool))],
        phase_variable=np.linspace(0.0, 1.0, n, dtype=np.float32),
    )


def test_open_clip_roundtrip(tmp_path):
    header = _make_header()
    tracks = _make_tracks()
    clip_dir = tmp_path / "walk_01"
    save_clip(clip_dir, Clip(header=header, tracks=tracks))

    reopened = open_clip(clip_dir)
    assert reopened.header == header
    assert reopened.tracks.joint_angles_names == _DOF_NAMES
    assert reopened.tracks.root_pose.shape[0] == round(header.clip.duration_s * header.clip.frame_rate)


def test_open_clip_validates_against_explicit_skeleton_param(tmp_path):
    header = _make_header()
    tracks = _make_tracks()
    clip_dir = tmp_path / "walk_01"
    save_clip(clip_dir, Clip(header=header, tracks=tracks), skeleton=_SKELETON)

    reopened = open_clip(clip_dir, skeleton=_SKELETON)
    assert reopened.tracks.joint_angles_names == _DOF_NAMES


def test_joint_angles_names_mismatch_raises(tmp_path):
    header = _make_header()
    tracks = _make_tracks(joint_angles_names=[f"wrong_{i}" for i in range(len(_DOF_NAMES))])
    clip_dir = tmp_path / "walk_01"
    with pytest.raises(ClipValidationError):
        save_clip(clip_dir, Clip(header=header, tracks=tracks))


def test_frame_count_mismatch_raises(tmp_path):
    header = _make_header()
    tracks = _make_tracks(n=_N + 1)
    clip_dir = tmp_path / "walk_01"
    with pytest.raises(ClipValidationError):
        save_clip(clip_dir, Clip(header=header, tracks=tracks))


def test_mandatory_channel_missing_propagates_as_tracks_error(tmp_path):
    header = _make_header()
    tracks = _make_tracks()
    tracks.contact = []
    clip_dir = tmp_path / "walk_01"
    with pytest.raises(TracksValidationError):
        save_clip(clip_dir, Clip(header=header, tracks=tracks))


def test_round_trip_preserves_authored_vs_derived_channel_marking(tmp_path):
    header = _make_header(
        channels=ChannelsBlock(
            root_pose="authored",
            joint_angles="derived",
            end_effector_targets={"FL_foot": "derived", "RL_foot": "authored"},
            contact_state="authored",
        )
    )
    tracks = _make_tracks()
    clip_dir = tmp_path / "walk_01"
    save_clip(clip_dir, Clip(header=header, tracks=tracks))

    reopened = open_clip(clip_dir)
    assert reopened.header.channels.end_effector_targets == {"FL_foot": "derived", "RL_foot": "authored"}
