"""Manual smoke test: connect to a running CoppeliaSim instance and run
random-policy episodes of any XML-defined task. Generalizes
scripts/random_rollout.py to whatever tasks/*.xml is passed in.

Usage:
    .venv/Scripts/python.exe scripts/random_rollout_xml.py tasks/reach.xml --episodes 3
"""

from __future__ import annotations

import argparse

from coppelia_rl.env_schema import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = load_env(args.xml_path, host=args.host, port=args.port)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            print(
                f"episode {episode}: steps={steps} return={episode_reward:.3f} "
                f"terminated={terminated} truncated={truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
