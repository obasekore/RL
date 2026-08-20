"""Trains PPO (SB3, MultiInputPolicy) against a task via vectorized headless
CoppeliaSim instances (coppelia_rl/training/vec_env.py), with a before/after
evaluation against a dedicated instance so learning progress is visible, not
just trusted from SB3's own console logging.

Usage:
    .venv/Scripts/python.exe scripts/train_sb3_ppo.py --xml-path tasks/reach.xml --num-envs 4 --total-timesteps 100000 --out models/reach_ppo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from coppelia_rl.env_schema import load_env
from coppelia_rl.training.instance_launcher import launch_and_connect
from coppelia_rl.training.vec_env import CoppeliaSimVecEnv


def evaluate(xml_path: str, client, model, episodes: int) -> list[float]:
    env = load_env(xml_path, client=client)
    try:
        returns = []
        for _ in range(episodes):
            obs, info = env.reset()
            terminated = truncated = False
            total = 0.0
            while not (terminated or truncated):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
            returns.append(total)
        return returns
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml-path", default="tasks/reach.xml")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=23700)
    parser.add_argument("--eval-port", type=int, default=23799)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--out", default="models/reach_ppo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Launching a dedicated evaluation instance on port {args.eval_port}...")
    eval_instance, eval_client = launch_and_connect(args.eval_port, host=args.host, startup_timeout=90)

    print(f"Launching {args.num_envs} training instances starting at port {args.base_port}...")
    vec_env = CoppeliaSimVecEnv.launch(args.xml_path, args.num_envs, base_port=args.base_port, startup_timeout=90)
    vec_env = VecMonitor(vec_env)

    try:
        model = PPO("MultiInputPolicy", vec_env, verbose=1, seed=args.seed)

        print(f"\nBaseline (random policy), {args.eval_episodes} episodes...")
        baseline_returns = evaluate(args.xml_path, eval_client, None, args.eval_episodes)
        print(f"  returns: {[round(r, 2) for r in baseline_returns]}")
        print(f"  mean: {np.mean(baseline_returns):.3f}")

        print(f"\nTraining for {args.total_timesteps} timesteps across {args.num_envs} instances...")
        start = time.time()
        model.learn(total_timesteps=args.total_timesteps)
        elapsed = time.time() - start
        print(f"Training took {elapsed / 60:.1f} min ({args.total_timesteps / elapsed:.1f} timesteps/sec)")
    finally:
        vec_env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"\nSaved model to {out_path}")

    print(f"\nTrained policy, {args.eval_episodes} episodes...")
    trained_returns = evaluate(args.xml_path, eval_client, model, args.eval_episodes)
    print(f"  returns: {[round(r, 2) for r in trained_returns]}")
    print(f"  mean: {np.mean(trained_returns):.3f}")

    eval_instance.terminate()

    print("\n=== Summary ===")
    print(f"Baseline (random) mean return: {np.mean(baseline_returns):.3f}")
    print(f"Trained policy   mean return: {np.mean(trained_returns):.3f}")


if __name__ == "__main__":
    main()
