"""Spawns headless CoppeliaSim instances on distinct ZMQ ports, for vectorized training.

Platform-specific bits (executable name/location) are isolated here so a
CoppeliaSim version/platform change never leaks into the rest of the add-on.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from pathlib import Path


class CoppeliaSimNotFoundError(RuntimeError):
    pass


def _executable_relative_path() -> str:
    system = platform.system()
    if system == "Windows":
        return "coppeliaSim.exe"
    if system == "Darwin":
        return "coppeliaSim.app/Contents/MacOS/coppeliaSim"
    return "coppeliaSim.sh"


def find_coppeliasim_executable() -> Path:
    """Locates the CoppeliaSim executable.

    Honors the COPPELIASIM_ROOT env var if set, otherwise walks up from this
    file's location (this add-on ships inside a CoppeliaSim install).
    """
    exe_relative = _executable_relative_path()

    override = os.environ.get("COPPELIASIM_ROOT")
    if override:
        candidate = Path(override) / exe_relative
        if candidate.exists():
            return candidate
        raise CoppeliaSimNotFoundError(
            f"COPPELIASIM_ROOT is set to {override!r} but {candidate} does not exist"
        )

    for parent in Path(__file__).resolve().parents:
        candidate = parent / exe_relative
        if candidate.exists():
            return candidate

    raise CoppeliaSimNotFoundError(
        "Could not locate a CoppeliaSim executable by walking up from this file. "
        "Set the COPPELIASIM_ROOT environment variable to the install directory."
    )


def spawn_headless_instance(
    port: int,
    scene_path: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launches a headless CoppeliaSim instance bound to its own ZMQ RPC port."""
    exe = find_coppeliasim_executable()
    args = [str(exe), "-h", f"-GzmqRemoteApi.rpcPort={port}"]
    if extra_args:
        args.extend(extra_args)
    if scene_path is not None:
        args.append(str(scene_path))
    return subprocess.Popen(args)


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> None:
    """Blocks until `port` accepts TCP connections, or raises TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"CoppeliaSim did not open port {port} within {timeout}s")
