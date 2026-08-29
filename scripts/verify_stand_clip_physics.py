"""Live physics-survival check for the biped_nao "stand" clip - not a trained-policy
rollout. Each step, the action holds every DOF at its reference (rest) angle, so this
isolates whether the *scene* (mass/contact/limits/friction) can physically hold a
standing pose at all, independent of any policy/reward-tuning question.

Deliberately not scripts/random_rollout_xml.py: a uniformly random joint_position
action would yank every DOF toward a random target every step regardless of scene
quality, which tests policy-vs-reward wiring, not scene physical viability.

Usage (with a CoppeliaSim instance + ZMQ remote API add-on running, and
scripts/build_biped_nao_scene.py + scripts/build_stand_clip.py already run):
    .venv/Scripts/python.exe scripts/verify_stand_clip_physics.py tasks/biped_nao_stand.xml --episodes 3
"""

from __future__ import annotations

import argparse

import numpy as np

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
        mi = env.unwrapped._motion_imitation
        for episode in range(args.episodes):
            env.reset()
            action = mi.reference_dof_angles().astype(np.float32)
            terminated = truncated = False
            steps = 0
            heights = []
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(action)
                heights.append(float(obs["torso_pos"][2]))
                steps += 1
            fell = terminated  # fall_detection terminates; max_steps truncates
            print(
                f"episode {episode}: steps={steps} fell={fell} "
                f"height[0]={heights[0]:.4f} height[-1]={heights[-1]:.4f} "
                f"height_min={min(heights):.4f} height_max={max(heights):.4f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
