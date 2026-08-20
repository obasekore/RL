"""Launches/connects to headless CoppeliaSim instances on distinct ZMQ ports.

The vectorization strategy is multiple headless CoppeliaSim
*processes*, not intra-process parallelism - this module is the one place
that knows how to spawn and reach one such process. Everything above it
(vec_env.py, rllib_adapter.py) only ever holds a connected SimClient.

Port assignment: the "ZMQ remote API server" add-on (addOns/ZMQ remote API
server.lua) picks its port from the named int32 param `zmqRemoteApi.rpcPort`
if already set, else falls back to `23000 + sim.intparam_processid` - which
isn't confirmed to be the OS PID and can't be predicted from Python. So every
instance here is launched with an explicit `-GzmqRemoteApi.rpcPort=<port>`
override (confirmed working live: launched two concurrent headless instances
this way, both independently reachable at their assigned ports).
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from coppelia_rl.sim_interface.client import SimClient

_IS_WINDOWS = platform.system() == "Windows"

_EXECUTABLE_NAMES_BY_SYSTEM = {
    "Windows": "coppeliaSim.exe",
    "Linux": "coppeliaSim.sh",
    "Darwin": "coppeliaSim.app/Contents/MacOS/coppeliaSim",
}


class CoppeliaSimExecutableNotFoundError(RuntimeError):
    pass


def find_coppeliasim_executable(coppeliasim_root: str | Path | None = None) -> Path:
    executable_name = _EXECUTABLE_NAMES_BY_SYSTEM.get(platform.system())
    if executable_name is None:
        raise CoppeliaSimExecutableNotFoundError(f"Unsupported platform {platform.system()!r}")

    root = coppeliasim_root or os.environ.get("COPPELIASIM_ROOT")
    if root:
        candidate = Path(root) / executable_name
        if candidate.exists():
            return candidate
        raise CoppeliaSimExecutableNotFoundError(f"COPPELIASIM_ROOT is set to {root!r} but {candidate} does not exist")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / executable_name
        if candidate.exists():
            return candidate

    raise CoppeliaSimExecutableNotFoundError(
        f"Could not locate {executable_name} by walking up from this file. "
        "Set the COPPELIASIM_ROOT environment variable to the install directory."
    )


def build_launch_args(executable: str | Path, port: int, extra_args: list[str] | None = None) -> list[str]:
    return [str(executable), "-h", f"-GzmqRemoteApi.rpcPort={port}"] + list(extra_args or [])


@dataclass
class CoppeliaSimInstance:
    process: subprocess.Popen
    host: str
    port: int

    def terminate(self, timeout: float = 10.0) -> None:
        """Kills the whole process tree, not just `process` itself.

        CoppeliaSim spawns its own child processes for every Python add-on
        it loads (`python/pythonLauncher.py`, one per add-on - this project
        ships two: RL Environment Builder, RL Domain Randomizer). Plain
        `process.terminate()` only signals the top-level coppeliaSim.exe and
        leaves those launcher children running as orphans - confirmed live
        (killed a coppeliaSim.exe this way and found 5 orphaned
        pythonLauncher.py processes still running afterward).
        """
        if self.process.poll() is not None:
            return
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)


def launch_headless_instance(
    port: int,
    *,
    host: str = "localhost",
    coppeliasim_root: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> CoppeliaSimInstance:
    executable = find_coppeliasim_executable(coppeliasim_root)
    args = build_launch_args(executable, port, extra_args)
    # On POSIX, put the process in its own session/group so terminate() can
    # kill its whole tree (including pythonLauncher.py children) without
    # touching unrelated processes. taskkill /T on Windows doesn't need this.
    popen_kwargs = {} if _IS_WINDOWS else {"start_new_session": True}
    process = subprocess.Popen(args, **popen_kwargs)
    return CoppeliaSimInstance(process=process, host=host, port=port)


def connect_when_ready(host: str, port: int, *, timeout: float = 30.0, poll_interval: float = 0.5) -> SimClient:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return SimClient.connect(host=host, port=port)
        except Exception as e:  # noqa: BLE001 - genuinely any connect failure should retry
            last_error = e
            time.sleep(poll_interval)
    raise TimeoutError(f"CoppeliaSim instance at {host}:{port} did not become reachable within {timeout}s") from last_error


def launch_and_connect(
    port: int,
    *,
    host: str = "localhost",
    coppeliasim_root: str | Path | None = None,
    extra_args: list[str] | None = None,
    startup_timeout: float = 30.0,
) -> tuple[CoppeliaSimInstance, SimClient]:
    instance = launch_headless_instance(port, host=host, coppeliasim_root=coppeliasim_root, extra_args=extra_args)
    client = connect_when_ready(host, port, timeout=startup_timeout)
    return instance, client
