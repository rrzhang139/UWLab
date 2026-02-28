# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC algorithm: actor/critic updates, target networks, auto-alpha tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import GaussianActor, TwinQCritic


class SAC:
    """Soft Actor-Critic with automatic entropy tuning.

    Manages actor, twin critic, target critic, and entropy temperature.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        use_layer_norm: bool = True,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_alpha: bool = True,
        init_alpha: float = 0.2,
        max_grad_norm: float = 1.0,
        device: str = "cuda:0",
    ):
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Networks
        self.actor = GaussianActor(obs_dim, action_dim, actor_hidden_dims, use_layer_norm).to(self.device)
        self.critic = TwinQCritic(obs_dim, action_dim, critic_hidden_dims, use_layer_norm).to(self.device)
        self.critic_target = TwinQCritic(obs_dim, action_dim, critic_hidden_dims, use_layer_norm).to(self.device)
        # Initialize target with same weights
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Entropy temperature
        if auto_alpha:
            self.target_entropy = -action_dim  # heuristic: -dim(A)
            self.log_alpha = torch.tensor(
                [float(torch.tensor(init_alpha).log())], requires_grad=True, device=self.device
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor([float(torch.tensor(init_alpha).log())], device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update_critic(self, batch: dict[str, torch.Tensor]) -> float:
        """One gradient step on the twin Q-critic."""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        truncated = batch["truncated"]

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_prob = self.actor(next_obs)
            # Target Q-values
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_prob
            # Bootstrap only if not terminated (but DO bootstrap if truncated)
            not_terminated = 1.0 - dones * (1.0 - truncated)
            td_target = rewards + self.gamma * not_terminated * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        """One gradient step on the actor (and alpha if auto-tuning)."""
        obs = batch["obs"]

        actions_new, log_prob = self.actor(obs)
        q1 = self.critic.q1_forward(obs, actions_new)
        actor_loss = (self.alpha.detach() * log_prob - q1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss_t = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss_t.item()

        return actor_loss.item(), alpha_loss

    def soft_update_target(self):
        """Polyak averaging update of target critic."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.lerp_(param.data, self.tau)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from the actor (no gradient)."""
        with torch.no_grad():
            if deterministic:
                return self.actor.get_action_deterministic(obs)
            action, _ = self.actor(obs)
            return action

    def save(self, path: str):
        """Save all networks and optimizers."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            },
            path,
        )

    def load(self, path: str):
        """Load all networks and optimizers."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(self.auto_alpha)
        if self.auto_alpha and ckpt["alpha_optimizer"] is not None:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_optimizer.defaults["lr"])
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
