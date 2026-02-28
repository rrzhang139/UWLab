# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Off-policy runner for SAC+HER training loop."""

from __future__ import annotations

import logging
import os
import time

import torch

from .her import HERRelabeler
from .replay_buffer import ReplayBuffer
from .runner_cfg import SacHerRunnerCfg
from .sac import SAC
from .vec_env_wrapper import SacHerVecEnvWrapper

logger = logging.getLogger(__name__)


class OffPolicyRunner:
    """Off-policy training loop for SAC+HER.

    Orchestrates: env stepping -> buffer insertion -> sampling -> SAC updates.
    Analogous to rsl_rl's OnPolicyRunner but for off-policy training.
    """

    def __init__(self, env: SacHerVecEnvWrapper, cfg: SacHerRunnerCfg, log_dir: str | None = None):
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.device = cfg.device

        # Determine dimensions
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim
        self.goal_dim = cfg.goal_dim if cfg.use_her else 0

        # For HER: obs_dim includes appended goal, obs_dim_without_goal is raw
        if cfg.use_her:
            self.obs_dim_without_goal = self.obs_dim - cfg.goal_dim
        else:
            self.obs_dim_without_goal = self.obs_dim

        # SAC agent
        self.sac = SAC(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            use_layer_norm=cfg.use_layer_norm,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            alpha_lr=cfg.alpha_lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            auto_alpha=cfg.auto_alpha,
            init_alpha=cfg.init_alpha,
            max_grad_norm=cfg.max_grad_norm,
            device=cfg.device,
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            goal_dim=cfg.goal_dim,
            device=cfg.device,
        )

        # HER relabeler
        self.her = None
        if cfg.use_her:
            self.her = HERRelabeler(
                replay_buffer=self.buffer,
                goal_dim=cfg.goal_dim,
                her_k=cfg.her_k,
                goal_threshold=cfg.goal_threshold,
            )

        # Logger
        self._wandb_run = None
        self._setup_logger()

        # Metrics
        self._metrics = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": 0.0,
            "episode_reward": 0.0,
            "episode_length": 0.0,
            "success_rate": 0.0,
        }
        self._episode_rewards = torch.zeros(env.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(env.num_envs, device=self.device)
        self._completed_episodes = 0
        self._completed_successes = 0
        self._log_episode_rewards = []
        self._log_episode_lengths = []

    def _setup_logger(self):
        """Initialize W&B or tensorboard logger."""
        if self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)

        if self.cfg.logger == "wandb":
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=self.cfg.wandb_project,
                    name=self.cfg.run_name or os.path.basename(self.log_dir),
                    dir=self.log_dir,
                    config={
                        "algorithm": "SAC+HER" if self.cfg.use_her else "SAC",
                        "obs_dim": self.obs_dim,
                        "action_dim": self.action_dim,
                        "goal_dim": self.goal_dim,
                        "buffer_capacity": self.cfg.buffer_capacity,
                        "batch_size": self.cfg.batch_size,
                        "utd_ratio": self.cfg.utd_ratio,
                        "actor_lr": self.cfg.actor_lr,
                        "critic_lr": self.cfg.critic_lr,
                        "gamma": self.cfg.gamma,
                        "tau": self.cfg.tau,
                        "use_her": self.cfg.use_her,
                        "her_k": self.cfg.her_k,
                        "warmup_steps": self.cfg.warmup_steps,
                        "num_envs": self.env.num_envs,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to init wandb: {e}. Falling back to no logging.")
                self._wandb_run = None

    def learn(self, max_steps: int | None = None):
        """Main training loop.

        Args:
            max_steps: Override for cfg.max_steps.
        """
        max_steps = max_steps or self.cfg.max_steps
        total_env_steps = 0
        update_count = 0

        # Reset env
        obs = self.env.reset()

        logger.info(f"Starting SAC{'_HER' if self.cfg.use_her else ''} training")
        logger.info(f"  obs_dim={self.obs_dim}, action_dim={self.action_dim}, goal_dim={self.goal_dim}")
        logger.info(f"  num_envs={self.env.num_envs}, buffer_capacity={self.cfg.buffer_capacity}")
        logger.info(f"  warmup_steps={self.cfg.warmup_steps}, max_steps={max_steps}")

        start_time = time.time()
        log_start_time = start_time

        while total_env_steps < max_steps:
            # 1. Get actions
            if total_env_steps < self.cfg.warmup_steps:
                # Random actions during warmup
                actions = torch.rand(self.env.num_envs, self.action_dim, device=self.device) * 2 - 1
            else:
                actions = self.sac.get_action(obs)

            # 2. Step environment
            next_obs, rewards, terminated, truncated, achieved_goals, next_achieved_goals, infos = self.env.step(
                actions
            )

            # Determine dones for buffer storage
            dones = ((terminated + truncated) > 0).float()

            # 3. Store transitions in replay buffer
            self.buffer.add_batch(
                obs=obs,
                actions=actions,
                rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
                next_obs=next_obs,
                dones=terminated.unsqueeze(-1) if terminated.dim() == 1 else terminated,
                truncated=truncated.unsqueeze(-1) if truncated.dim() == 1 else truncated,
                achieved_goals=achieved_goals,
                next_achieved_goals=next_achieved_goals,
                desired_goals=self.env.desired_goal,
                episode_ids=self.env.episode_ids,
                timestep_in_episode=self.env.timestep_in_episode,
            )

            # 4. Track episode stats
            self._episode_rewards += rewards
            self._episode_lengths += 1

            done_mask = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
            n_done = done_mask.sum().item()
            if n_done > 0:
                self._log_episode_rewards.extend(self._episode_rewards[done_mask].cpu().tolist())
                self._log_episode_lengths.extend(self._episode_lengths[done_mask].cpu().tolist())
                self._completed_episodes += int(n_done)

                # Check for success (reward > threshold at terminal step)
                # Use terminated (not truncated) as success signal
                terminated_mask = terminated.bool().squeeze(-1) if terminated.dim() > 1 else terminated.bool()
                self._completed_successes += int(terminated_mask.sum().item())

                self._episode_rewards[done_mask] = 0
                self._episode_lengths[done_mask] = 0

            # 5. Update policy if enough data
            if total_env_steps >= self.cfg.warmup_steps and len(self.buffer) >= self.cfg.batch_size:
                for _ in range(self.cfg.utd_ratio):
                    batch = self.buffer.sample(self.cfg.batch_size)

                    # HER relabeling
                    if self.her is not None:
                        batch = self.her.relabel_batch(batch, self.obs_dim_without_goal)

                    # SAC updates
                    critic_loss = self.sac.update_critic(batch)
                    actor_loss, alpha_loss = self.sac.update_actor_and_alpha(batch)
                    self.sac.soft_update_target()

                    update_count += 1
                    self._metrics["critic_loss"] = critic_loss
                    self._metrics["actor_loss"] = actor_loss
                    self._metrics["alpha_loss"] = alpha_loss
                    self._metrics["alpha"] = self.sac.alpha.item()

            total_env_steps += self.env.num_envs
            obs = next_obs

            # 6. Logging
            if total_env_steps % self.cfg.log_interval < self.env.num_envs:
                elapsed = time.time() - log_start_time
                fps = self.cfg.log_interval / elapsed if elapsed > 0 else 0
                log_start_time = time.time()

                avg_reward = sum(self._log_episode_rewards[-100:]) / max(len(self._log_episode_rewards[-100:]), 1)
                avg_length = sum(self._log_episode_lengths[-100:]) / max(len(self._log_episode_lengths[-100:]), 1)
                success_rate = self._completed_successes / max(self._completed_episodes, 1)

                log_data = {
                    "env_steps": total_env_steps,
                    "updates": update_count,
                    "fps": fps,
                    "buffer_size": len(self.buffer),
                    "critic_loss": self._metrics["critic_loss"],
                    "actor_loss": self._metrics["actor_loss"],
                    "alpha": self._metrics["alpha"],
                    "avg_episode_reward": avg_reward,
                    "avg_episode_length": avg_length,
                    "success_rate": success_rate,
                    "completed_episodes": self._completed_episodes,
                }

                logger.info(
                    f"Steps: {total_env_steps:>10d} | "
                    f"FPS: {fps:.0f} | "
                    f"Buf: {len(self.buffer)} | "
                    f"C_loss: {self._metrics['critic_loss']:.4f} | "
                    f"A_loss: {self._metrics['actor_loss']:.4f} | "
                    f"Alpha: {self._metrics['alpha']:.4f} | "
                    f"Rew: {avg_reward:.3f} | "
                    f"Succ: {success_rate:.3f}"
                )

                if self._wandb_run is not None:
                    import wandb

                    wandb.log(log_data, step=total_env_steps)

                # Reset counters for success rate tracking per log interval
                self._completed_episodes = 0
                self._completed_successes = 0

            # 7. Save checkpoint
            if total_env_steps % self.cfg.save_interval < self.env.num_envs and self.log_dir is not None:
                ckpt_path = os.path.join(self.log_dir, f"checkpoint_{total_env_steps}.pt")
                self.save(ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        total_time = time.time() - start_time
        logger.info(f"Training complete. Total steps: {total_env_steps}, Time: {total_time:.1f}s")

        # Save final checkpoint
        if self.log_dir is not None:
            final_path = os.path.join(self.log_dir, "checkpoint_final.pt")
            self.save(final_path)
            logger.info(f"Saved final checkpoint: {final_path}")

    def save(self, path: str):
        """Save SAC networks and training state."""
        self.sac.save(path)

    def load(self, path: str):
        """Load SAC networks from checkpoint."""
        self.sac.load(path)
        logger.info(f"Loaded checkpoint from: {path}")
