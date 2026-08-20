import pytest

from coppelia_rl.training.instance_launcher import (
    CoppeliaSimExecutableNotFoundError,
    build_launch_args,
    find_coppeliasim_executable,
)


def test_build_launch_args_sets_headless_and_port():
    args = build_launch_args("coppeliaSim.exe", 23100)
    assert args == ["coppeliaSim.exe", "-h", "-GzmqRemoteApi.rpcPort=23100"]


def test_build_launch_args_appends_extra_args():
    args = build_launch_args("coppeliaSim.exe", 23100, extra_args=["-q"])
    assert args[-1] == "-q"


def test_find_coppeliasim_executable_walks_up_from_install(monkeypatch):
    monkeypatch.delenv("COPPELIASIM_ROOT", raising=False)
    # Exercises the real install this project lives inside of - the same
    # walk-up-parents pattern already used by envs/ur5_arm.py.
    exe = find_coppeliasim_executable()
    assert exe.exists()
    assert exe.name == "coppeliaSim.exe"


def test_find_coppeliasim_executable_honors_root_override(tmp_path):
    fake_exe = tmp_path / "coppeliaSim.exe"
    fake_exe.touch()
    exe = find_coppeliasim_executable(coppeliasim_root=tmp_path)
    assert exe == fake_exe


def test_find_coppeliasim_executable_raises_when_root_override_is_wrong(tmp_path):
    with pytest.raises(CoppeliaSimExecutableNotFoundError):
        find_coppeliasim_executable(coppeliasim_root=tmp_path / "does_not_exist")
