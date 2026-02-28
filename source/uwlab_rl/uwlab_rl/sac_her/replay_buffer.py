# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-resident circular replay buffer with episode tracking for HER."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Fixed-capacity circular replay buffer that stores transitions on GPU.

    Tracks episode boundaries via episode_id for HER relabeling.
    Supports vectorized batch insertion from parallel environments.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int, device: str = "cuda:0"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.ptr = 0

        # Pre-allocate storage on GPU
        self.obs = torch.zeros(capacity, obs_dim, device=self.device)
        self.actions = torch.zeros(capacity, action_dim, device=self.device)
        self.rewards = torch.zeros(capacity, 1, device=self.device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=self.device)
        self.dones = torch.zeros(capacity, 1, device=self.device)
        self.truncated = torch.zeros(capacity, 1, device=self.device)

        # HER-specific: achieved_goal at current and next state, desired_goal, episode tracking
        self.achieved_goals = torch.zeros(capacity, goal_dim, device=self.device)
        self.next_achieved_goals = torch.zeros(capacity, goal_dim, device=self.device)
        self.desired_goals = torch.zeros(capacity, goal_dim, device=self.device)
        self.episode_ids = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.timestep_in_episode = torch.zeros(capacity, dtype=torch.long, device=self.device)

    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        truncated: torch.Tensor,
        achieved_goals: torch.Tensor,
        next_achieved_goals: torch.Tensor,
        desired_goals: torch.Tensor,
        episode_ids: torch.Tensor,
        timestep_in_episode: torch.Tensor,
    ):
        """Add a batch of N transitions (one per parallel env).

        All tensors should have shape (N, ...) and be on the correct device.
        """
        batch_size = obs.shape[0]

        if self.ptr + batch_size <= self.capacity:
            # Contiguous insertion
            idx = slice(self.ptr, self.ptr + batch_size)
            self.obs[idx] = obs
            self.actions[idx] = actions
            self.rewards[idx] = rewards.view(-1, 1) if rewards.dim() == 1 else rewards
            self.next_obs[idx] = next_obs
            self.dones[idx] = dones.view(-1, 1) if dones.dim() == 1 else dones
            self.truncated[idx] = truncated.view(-1, 1) if truncated.dim() == 1 else truncated
            self.achieved_goals[idx] = achieved_goals
            self.next_achieved_goals[idx] = next_achieved_goals
            self.desired_goals[idx] = desired_goals
            self.episode_ids[idx] = episode_ids
            self.timestep_in_episode[idx] = timestep_in_episode
        else:
            # Wrap around: need to split
            first = self.capacity - self.ptr
            second = batch_size - first

            self.obs[self.ptr:] = obs[:first]
            self.obs[:second] = obs[first:]
            self.actions[self.ptr:] = actions[:first]
            self.actions[:second] = actions[first:]
            self.rewards[self.ptr:] = rewards[:first].view(-1, 1) if rewards.dim() == 1 else rewards[:first]
            self.rewards[:second] = rewards[first:].view(-1, 1) if rewards.dim() == 1 else rewards[first:]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.next_obs[:second] = next_obs[first:]
            self.dones[self.ptr:] = dones[:first].view(-1, 1) if dones.dim() == 1 else dones[:first]
            self.dones[:second] = dones[first:].view(-1, 1) if dones.dim() == 1 else dones[first:]
            self.truncated[self.ptr:] = truncated[:first].view(-1, 1) if truncated.dim() == 1 else truncated[:first]
            self.truncated[:second] = truncated[first:].view(-1, 1) if truncated.dim() == 1 else truncated[first:]
            self.achieved_goals[self.ptr:] = achieved_goals[:first]
            self.achieved_goals[:second] = achieved_goals[first:]
            self.next_achieved_goals[self.ptr:] = next_achieved_goals[:first]
            self.next_achieved_goals[:second] = next_achieved_goals[first:]
            self.desired_goals[self.ptr:] = desired_goals[:first]
            self.desired_goals[:second] = desired_goals[first:]
            self.episode_ids[self.ptr:] = episode_ids[:first]
            self.episode_ids[:second] = episode_ids[first:]
            self.timestep_in_episode[self.ptr:] = timestep_in_episode[:first]
            self.timestep_in_episode[:second] = timestep_in_episode[first:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns dict with keys: obs, actions, rewards, next_obs, dones, truncated,
        achieved_goals, next_achieved_goals, desired_goals, episode_ids, timestep_in_episode, indices.
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "truncated": self.truncated[indices],
            "achieved_goals": self.achieved_goals[indices],
            "next_achieved_goals": self.next_achieved_goals[indices],
            "desired_goals": self.desired_goals[indices],
            "episode_ids": self.episode_ids[indices],
            "timestep_in_episode": self.timestep_in_episode[indices],
            "indices": indices,
        }

    def __len__(self) -> int:
        return self.size
