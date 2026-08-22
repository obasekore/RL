import numpy as np
import pytest

pytest.importorskip("h5py")

import h5py  # noqa: E402

from coppelia_rl.motion.tracks_hdf5 import (  # noqa: E402
    ContactTrack,
    EndEffectorTrack,
    TracksData,
    TracksValidationError,
    read_tracks,
    write_tracks,
)

_N = 5
_J = 12
_JOINT_NAMES = [f"j{i}" for i in range(_J)]


def _make_tracks(*, with_optional: bool) -> TracksData:
    rng = np.random.default_rng(0)
    contact = [
        ContactTrack(
            name="FL_foot",
            contact=np.array([True, True, False, False, True]),
            force_hint=rng.random(_N, dtype=np.float32) if with_optional else None,
        ),
        ContactTrack(name="FR_foot", contact=np.array([False, True, True, False, True])),
    ]
    end_effectors = (
        [
            EndEffectorTrack(name="FL_foot", target=rng.random((_N, 3), dtype=np.float32), frame="root_relative"),
            EndEffectorTrack(name="FR_foot", target=rng.random((_N, 3), dtype=np.float32), frame="world"),
        ]
        if with_optional
        else []
    )
    return TracksData(
        root_pose=rng.random((_N, 7), dtype=np.float32),
        joint_angles=rng.random((_N, _J), dtype=np.float32),
        joint_angles_names=list(_JOINT_NAMES),
        contact=contact,
        phase_variable=np.linspace(0.0, 1.0, _N, dtype=np.float32),
        root_velocity=rng.random((_N, 6), dtype=np.float32) if with_optional else None,
        joint_velocities=rng.random((_N, _J), dtype=np.float32) if with_optional else None,
        end_effectors=end_effectors,
    )


def _assert_tracks_equal(a: TracksData, b: TracksData) -> None:
    np.testing.assert_array_equal(a.root_pose, b.root_pose)
    np.testing.assert_array_equal(a.joint_angles, b.joint_angles)
    assert a.joint_angles_names == b.joint_angles_names
    np.testing.assert_array_equal(a.phase_variable, b.phase_variable)

    if a.root_velocity is None:
        assert b.root_velocity is None
    else:
        np.testing.assert_array_equal(a.root_velocity, b.root_velocity)
    if a.joint_velocities is None:
        assert b.joint_velocities is None
    else:
        np.testing.assert_array_equal(a.joint_velocities, b.joint_velocities)

    a_contact = {c.name: c for c in a.contact}
    b_contact = {c.name: c for c in b.contact}
    assert set(a_contact) == set(b_contact)
    for name in a_contact:
        np.testing.assert_array_equal(a_contact[name].contact, b_contact[name].contact)
        if a_contact[name].force_hint is None:
            assert b_contact[name].force_hint is None
        else:
            np.testing.assert_array_equal(a_contact[name].force_hint, b_contact[name].force_hint)

    a_ee = {e.name: e for e in a.end_effectors}
    b_ee = {e.name: e for e in b.end_effectors}
    assert set(a_ee) == set(b_ee)
    for name in a_ee:
        np.testing.assert_array_equal(a_ee[name].target, b_ee[name].target)
        assert a_ee[name].frame == b_ee[name].frame


# -- round trip -----------------------------------------------------------------


def test_round_trip_with_optional_fields(tmp_path):
    tracks = _make_tracks(with_optional=True)
    path = tmp_path / "tracks.h5"
    write_tracks(path, tracks)
    _assert_tracks_equal(tracks, read_tracks(path))


def test_round_trip_without_optional_fields(tmp_path):
    tracks = _make_tracks(with_optional=False)
    path = tmp_path / "tracks.h5"
    write_tracks(path, tracks)
    _assert_tracks_equal(tracks, read_tracks(path))


def test_write_creates_parent_directories(tmp_path):
    tracks = _make_tracks(with_optional=False)
    path = tmp_path / "nested" / "dir" / "tracks.h5"
    write_tracks(path, tracks)
    assert path.exists()


def test_bool_contact_dtype_preserved(tmp_path):
    tracks = _make_tracks(with_optional=False)
    path = tmp_path / "tracks.h5"
    write_tracks(path, tracks)
    for c in read_tracks(path).contact:
        assert c.contact.dtype == np.bool_


def test_contact_force_hint_dataset_naming(tmp_path):
    tracks = _make_tracks(with_optional=True)
    path = tmp_path / "tracks.h5"
    write_tracks(path, tracks)

    with h5py.File(path, "r") as f:
        assert "FL_foot" in f["contact"]
        assert "FL_foot_force_hint" in f["contact"]
        assert "FR_foot_force_hint" not in f["contact"]

    read_back = read_tracks(path)
    fl = next(c for c in read_back.contact if c.name == "FL_foot")
    fr = next(c for c in read_back.contact if c.name == "FR_foot")
    assert fl.force_hint is not None
    assert fr.force_hint is None


# -- validation -------------------------------------------------------------------


def test_missing_mandatory_channel_raises_on_read(tmp_path):
    path = tmp_path / "tracks.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("root_pose", data=np.zeros((_N, 7), dtype=np.float32))
        f.create_dataset("joint_angles", data=np.zeros((_N, _J), dtype=np.float32))
        f.create_dataset("joint_angles_names", data=_JOINT_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("phase_variable", data=np.zeros(_N, dtype=np.float32))
        # no /contact group at all
    with pytest.raises(TracksValidationError):
        read_tracks(path)


def test_empty_contact_group_raises_on_read(tmp_path):
    path = tmp_path / "tracks.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("root_pose", data=np.zeros((_N, 7), dtype=np.float32))
        f.create_dataset("joint_angles", data=np.zeros((_N, _J), dtype=np.float32))
        f.create_dataset("joint_angles_names", data=_JOINT_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
        f.create_dataset("phase_variable", data=np.zeros(_N, dtype=np.float32))
        f.create_group("contact")
    with pytest.raises(TracksValidationError):
        read_tracks(path)


def test_write_rejects_empty_contact(tmp_path):
    tracks = _make_tracks(with_optional=False)
    tracks.contact = []
    with pytest.raises(TracksValidationError):
        write_tracks(tmp_path / "tracks.h5", tracks)


def test_write_rejects_shape_mismatch(tmp_path):
    tracks = _make_tracks(with_optional=False)
    tracks.phase_variable = tracks.phase_variable[:-1]
    with pytest.raises(TracksValidationError):
        write_tracks(tmp_path / "tracks.h5", tracks)
