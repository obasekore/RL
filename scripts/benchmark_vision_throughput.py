"""Benchmarks step throughput to establish whether rendering or physics
stepping is the likely bottleneck for vectorized image-based training.
Not a portable number - results are this-machine/this-run, recorded
with methodology in docs/benchmarks.md.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_vision_throughput.py
"""

from __future__ import annotations

import time

from coppelia_rl.env_schema import load_env
from coppelia_rl.training.instance_launcher import launch_and_connect
from coppelia_rl.training.vec_env import CoppeliaSimVecEnv

_WARMUP_STEPS = 10
_TIMED_STEPS = 50


def _benchmark_single(xml_path: str, port: int) -> float:
    instance, client = launch_and_connect(port, startup_timeout=60)
    try:
        env = load_env(xml_path, client=client)
        try:
            env.reset()
            for _ in range(_WARMUP_STEPS):
                env.step(env.action_space.sample())

            start = time.perf_counter()
            for _ in range(_TIMED_STEPS):
                env.step(env.action_space.sample())
            elapsed = time.perf_counter() - start
        finally:
            env.close()
    finally:
        instance.terminate()
    return _TIMED_STEPS / elapsed


def _benchmark_vectorized(xml_path: str, num_envs: int, base_port: int) -> float:
    vec_env = CoppeliaSimVecEnv.launch(xml_path, num_envs, base_port=base_port, startup_timeout=60)
    try:
        vec_env.reset()
        for _ in range(_WARMUP_STEPS):
            vec_env.step([vec_env.action_space.sample() for _ in range(num_envs)])

        start = time.perf_counter()
        for _ in range(_TIMED_STEPS):
            vec_env.step([vec_env.action_space.sample() for _ in range(num_envs)])
        elapsed = time.perf_counter() - start
    finally:
        vec_env.close()
    env_steps_per_sec = (_TIMED_STEPS * num_envs) / elapsed
    return env_steps_per_sec


def main() -> None:
    results = {}

    results["reach (no camera), single instance"] = _benchmark_single("tasks/reach.xml", 23300)
    results["pick_and_place (128x128 wrist_cam), single instance"] = _benchmark_single(
        "tasks/pick_and_place.xml", 23301
    )
    results["pick_and_place (128x128 wrist_cam), 2 instances"] = _benchmark_vectorized(
        "tasks/pick_and_place.xml", 2, 23310
    )
    results["pick_and_place (128x128 wrist_cam), 4 instances"] = _benchmark_vectorized(
        "tasks/pick_and_place.xml", 4, 23320
    )

    print()
    print("Benchmark results (steps/sec, or env-steps/sec for vectorized rows):")
    for label, value in results.items():
        print(f"  {label}: {value:.1f}")


if __name__ == "__main__":
    main()
