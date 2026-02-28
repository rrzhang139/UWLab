# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vector environment wrapper for SAC+HER.

Wraps an Isaac Lab gymnasium env to:
1. Extract achieved_goal (insertive_asset_in_receptive_asset_frame) from obs
2. Optionally append desired_goal to policy obs for goal-conditioned policy
3. Separate terminated vs truncated for correct SAC bootstrapping
4. Track episode IDs for HER relabeling
"""

from __future__ import annotations

import gymnasium as gym
import torch


class SacHerVecEnvWrapper:
    """Wraps Isaac Lab ManagerBasedRLEnv for SAC+HER training.

    The wrapper extracts achieved_goal from the last `goal_dim` dimensions
    of the policy observation (insertive_asset_in_receptive_asset_frame = 6D).
    When use_her=True, it also appends the desired_goal to obs.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_dim: int = 6,
        use_her: bool = False,
        clip_actions: bool = True,
    ):
        self.env = env
        self.goal_dim = goal_dim
        self.use_her = use_her
        self.clip_actions = clip_actions

        # Access underlying Isaac Lab env
        self.unwrapped_env = env.unwrapped
        self.device = self.unwrapped_env.device
        self.num_envs = self.unwrapped_env.num_envs

        # Get action and observation spaces
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[-1]
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")

        # Get raw obs dim from a test observation
        obs, _ = self.env.reset()
        if isinstance(obs, dict):
            self._obs_key = "policy"
            raw_obs = obs["policy"]
        else:
            self._obs_key = None
            raw_obs = obs

        self.raw_obs_dim = raw_obs.shape[-1]
        # achieved_goal is the last goal_dim dims of raw obs
        self.obs_dim_without_goal_part = self.raw_obs_dim - goal_dim

        if use_her:
            # obs = raw_obs + desired_goal appended
            self.obs_dim = self.raw_obs_dim + goal_dim
        else:
            self.obs_dim = self.raw_obs_dim

        # Desired goal: the assembly target (peg fully in hole = zeros in relative frame)
        self.desired_goal = torch.zeros(self.num_envs, goal_dim, device=self.device)

        # Episode tracking for HER
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_counter = 0
        self.timestep_in_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Store last raw obs for achieved_goal extraction
        self._last_raw_obs = raw_obs.clone()

    def _extract_obs(self, obs_data) -> torch.Tensor:
        """Extract flat observation tensor from gym output."""
        if isinstance(obs_data, dict):
            return obs_data[self._obs_key]
        return obs_data

    def _extract_achieved_goal(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Extract achieved_goal from the last goal_dim dims of raw obs."""
        return raw_obs[:, -self.goal_dim:]

    def _build_obs(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Build final obs tensor, optionally appending desired_goal."""
        if self.use_her:
            return torch.cat([raw_obs, self.desired_goal], dim=-1)
        return raw_obs

    def reset(self) -> torch.Tensor:
        """Reset all environments and return initial observation."""
        obs_data, _ = self.env.reset()
        raw_obs = self._extract_obs(obs_data)
        self._last_raw_obs = raw_obs.clone()

        # Reset episode tracking
        self.episode_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_counter = self.num_envs
        self.timestep_in_episode.zero_()

        return self._build_obs(raw_obs)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments.

        Returns:
            obs: Policy observation (num_envs, obs_dim)
            rewards: Per-env reward (num_envs,)
            terminated: True if episode ended due to success/failure (num_envs,)
            truncated: True if episode ended due to time limit (num_envs,)
            achieved_goals: Current achieved goal before step (num_envs, goal_dim)
            next_achieved_goals: Achieved goal after step (num_envs, goal_dim)
            infos: Dict of extra info
        """
        if self.clip_actions:
            actions = actions.clamp(-1.0, 1.0)

        # Extract achieved_goal BEFORE step
        achieved_goals = self._extract_achieved_goal(self._last_raw_obs).clone()

        # Step the environment
        obs_data, rewards, terminated_gym, truncated_gym, infos = self.env.step(actions)
        raw_obs = self._extract_obs(obs_data)

        # Extract achieved_goal AFTER step
        next_achieved_goals = self._extract_achieved_goal(raw_obs).clone()

        # Isaac Lab wraps terminated+truncated into a single "done" signal via gymnasium.
        # The manager-based env sets terminated based on termination conditions
        # and truncated based on time_out.
        # Convert to tensors on device
        if isinstance(terminated_gym, torch.Tensor):
            terminated = terminated_gym.float()
        else:
            terminated = torch.tensor(terminated_gym, dtype=torch.float32, device=self.device)

        if isinstance(truncated_gym, torch.Tensor):
            truncated = truncated_gym.float()
        else:
            truncated = torch.tensor(truncated_gym, dtype=torch.float32, device=self.device)

        dones = ((terminated + truncated) > 0).float()

        # Update episode tracking for envs that finished
        done_envs = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
        n_done = done_envs.sum().item()
        if n_done > 0:
            new_ids = torch.arange(
                self.episode_counter, self.episode_counter + int(n_done), dtype=torch.long, device=self.device
            )
            self.episode_ids[done_envs] = new_ids
            self.episode_counter += int(n_done)
            self.timestep_in_episode[done_envs] = 0

        self.timestep_in_episode += 1
        self._last_raw_obs = raw_obs.clone()

        obs = self._build_obs(raw_obs)

        # Ensure rewards is (num_envs,)
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)

        return obs, rewards, terminated, truncated, achieved_goals, next_achieved_goals, infos

    def close(self):
        """Close the underlying environment."""
        self.env.close()

    @property
    def step_dt(self):
        return self.unwrapped_env.step_dt
