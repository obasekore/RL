"""Generates auto-labeled demonstration episodes for reach.xml using
ReachTowardTargetPolicy, recorded via DemoRecorder to episode-indexed HDF5.

Usage:
    .venv/Scripts/python.exe scripts/generate_scripted_demos.py --episodes 10 --out demos/reach_scripted.h5
"""

from __future__ import annotations

import argparse

from coppelia_rl.demos.recorder import DemoRecorder
from coppelia_rl.demos.scripted_policies import ReachTowardTargetPolicy
from coppelia_rl.env_schema import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", default="tasks/reach.xml")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", default="demos/reach_scripted.h5")
    parser.add_argument("--success-threshold", type=float, default=0.05)
    args = parser.parse_args()

    env = load_env(args.xml_path, host=args.host, port=args.port)
    policy = ReachTowardTargetPolicy(env.action_space)

    def success_fn(obs, terminated, truncated, info):
        # reach.xml has no signal-based termination to derive success from -
        # judge by final tip-to-target distance instead.
        return policy.success(obs, threshold=args.success_threshold)

    recorder = DemoRecorder(env, args.out, env_name="reach", success_fn=success_fn)

    try:
        successes = 0
        for episode in range(args.episodes):
            policy.reset()
            obs, info = recorder.reset()
            terminated = truncated = False
            steps = 0
            episode_reward = 0.0
            while not (terminated or truncated):
                action = policy(obs)
                obs, reward, terminated, truncated, info = recorder.step(action)
                episode_reward += reward
                steps += 1
            success = policy.success(obs, threshold=args.success_threshold)
            successes += int(success)
            print(f"episode {episode}: steps={steps} return={episode_reward:.3f} success={success}")
        print(f"\n{successes}/{args.episodes} episodes succeeded (threshold={args.success_threshold})")
        print(f"Saved to {args.out}")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
