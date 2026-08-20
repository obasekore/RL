"""CleanRL-style PPO example against the reach task, vectorized across
multiple headless CoppeliaSim instances.

Per the spec, CleanRL support is an *example script*, not a formal adapter
like the SB3 VecEnv (coppelia_rl/training/vec_env.py) or the RLlib env
creator - this file demonstrates the pattern (continuous-action PPO, GAE,
clipped surrogate loss) adapted from CleanRL's canonical
`ppo_continuous_action.py` structure, trimmed to the essentials. It reuses
`CoppeliaSimVecEnv.launch()` directly for vectorization rather than
inventing a second mechanism.

reach.xml's observation space is a Dict of flat vectors (no images), so it's
flattened into a single Box here by concatenation - Dict-space policies are
Milestone 6's concern (image observation support), out of scope for this
example.

Usage:
    .venv/Scripts/python.exe scripts/train_cleanrl_reach.py --num-envs 2 --total-timesteps 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from coppelia_rl.training.vec_env import CoppeliaSimVecEnv


def flatten_obs(obs: dict, keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(obs[k].shape[0], -1) for k in keys], axis=-1)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", default="tasks/reach.xml")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=23150)
    parser.add_argument("--num-steps", type=int, default=64, help="Rollout length per env before each PPO update.")
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vec_env = CoppeliaSimVecEnv.launch(args.xml_path, args.num_envs, base_port=args.base_port)
    try:
        obs_keys = sorted(vec_env.observation_space.spaces.keys())
        obs_dim = sum(int(np.prod(vec_env.observation_space[k].shape)) for k in obs_keys)
        action_dim = int(np.prod(vec_env.action_space.shape))
        action_low = torch.as_tensor(vec_env.action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(vec_env.action_space.high, dtype=torch.float32)

        agent = Agent(obs_dim, action_dim)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

        num_updates = max(1, args.total_timesteps // (args.num_steps * args.num_envs))
        next_obs = torch.as_tensor(flatten_obs(vec_env.reset(), obs_keys))

        for update in range(num_updates):
            obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim))
            actions_buf = torch.zeros((args.num_steps, args.num_envs, action_dim))
            logprobs_buf = torch.zeros((args.num_steps, args.num_envs))
            rewards_buf = torch.zeros((args.num_steps, args.num_envs))
            dones_buf = torch.zeros((args.num_steps, args.num_envs))
            values_buf = torch.zeros((args.num_steps, args.num_envs))

            start = time.time()
            for step in range(args.num_steps):
                obs_buf[step] = next_obs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                clipped_action = torch.clamp(action, action_low, action_high)

                obs, reward, done, infos = vec_env.step(clipped_action.numpy())

                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value.flatten()
                rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32)
                dones_buf[step] = torch.as_tensor(done, dtype=torch.float32)
                next_obs = torch.as_tensor(flatten_obs(obs, obs_keys))

            with torch.no_grad():
                # GAE, indexed so dones_buf[t] (did step t's transition end
                # the episode?) alone decides whether values_buf[t+1]/next_value
                # is a valid bootstrap target - self-consistent regardless of
                # step t, unlike schemes that peek at dones_buf[t+1].
                next_value = agent.get_value(next_obs).flatten()
                advantages = torch.zeros_like(rewards_buf)
                last_gae_lambda = 0.0
                for t in reversed(range(args.num_steps)):
                    next_non_terminal = 1.0 - dones_buf[t]
                    next_values = next_value if t == args.num_steps - 1 else values_buf[t + 1]
                    delta = rewards_buf[t] + args.gamma * next_values * next_non_terminal - values_buf[t]
                    advantages[t] = last_gae_lambda = (
                        delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lambda
                    )
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_dim)
            b_actions = actions_buf.reshape(-1, action_dim)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            for _ in range(args.update_epochs):
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs, b_actions)
                log_ratio = new_logprob - b_logprobs
                ratio = log_ratio.exp()

                normalized_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                policy_loss_1 = -normalized_advantages * ratio
                policy_loss_2 = -normalized_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_loss = 0.5 * ((new_value.flatten() - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - 0.01 * entropy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            elapsed = time.time() - start
            print(
                f"update {update + 1}/{num_updates}: "
                f"mean_reward={rewards_buf.mean().item():.4f} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f} "
                f"({elapsed:.1f}s)"
            )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
