# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hindsight Experience Replay (HER) relabeling at sample time."""

from __future__ import annotations

import torch

from .replay_buffer import ReplayBuffer


class HERRelabeler:
    """Implements the 'future' HER strategy.

    At sample time, for a fraction of the batch, replaces the desired_goal
    with the achieved_goal from a future timestep in the same episode.
    Recomputes binary sparse reward based on the new goal.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        goal_dim: int,
        her_k: int = 4,
        goal_threshold: float = 0.02,
    ):
        self.buffer = replay_buffer
        self.goal_dim = goal_dim
        self.her_k = her_k
        self.goal_threshold = goal_threshold
        # Fraction of batch to relabel: k / (k + 1)
        self.relabel_fraction = her_k / (her_k + 1)

    def relabel_batch(self, batch: dict[str, torch.Tensor], obs_dim_without_goal: int) -> dict[str, torch.Tensor]:
        """Relabel a sampled batch with HER 'future' strategy.

        For each transition selected for relabeling:
        1. Find transitions from the same episode with later timesteps
        2. Pick one at random as the new goal
        3. Replace desired_goal in obs/next_obs
        4. Recompute sparse reward

        Args:
            batch: Dict from replay_buffer.sample()
            obs_dim_without_goal: The obs dimension before goal was appended.
                Used to locate the goal portion in obs tensor.

        Returns:
            Modified batch dict with relabeled goals and rewards.
        """
        device = batch["obs"].device
        bs = batch["obs"].shape[0]

        # Determine which transitions to relabel
        n_relabel = int(bs * self.relabel_fraction)
        relabel_mask = torch.zeros(bs, dtype=torch.bool, device=device)
        relabel_indices = torch.randperm(bs, device=device)[:n_relabel]
        relabel_mask[relabel_indices] = True

        if n_relabel == 0:
            return batch

        # For each transition to relabel, find a future transition from the same episode
        indices_to_relabel = batch["indices"][relabel_mask]  # buffer indices
        episode_ids = batch["episode_ids"][relabel_mask]
        timesteps = batch["timestep_in_episode"][relabel_mask]

        # We need to search the buffer for same-episode, future-timestep transitions
        # This is done on GPU for efficiency
        new_goals = torch.zeros(n_relabel, self.goal_dim, device=device)
        valid_relabel = torch.zeros(n_relabel, dtype=torch.bool, device=device)

        buf_episode_ids = self.buffer.episode_ids[: self.buffer.size]
        buf_timesteps = self.buffer.timestep_in_episode[: self.buffer.size]
        buf_achieved = self.buffer.achieved_goals[: self.buffer.size]

        for i in range(n_relabel):
            ep_id = episode_ids[i]
            t = timesteps[i]

            # Find all transitions in same episode with later timestep
            same_ep = buf_episode_ids == ep_id
            future = buf_timesteps > t
            candidates = same_ep & future

            n_candidates = candidates.sum().item()
            if n_candidates > 0:
                candidate_indices = candidates.nonzero(as_tuple=False).squeeze(-1)
                # Pick one at random
                pick = torch.randint(0, n_candidates, (1,), device=device)
                chosen_idx = candidate_indices[pick[0]]
                new_goals[i] = buf_achieved[chosen_idx]
                valid_relabel[i] = True

        # Apply relabeling only where we found valid future transitions
        if valid_relabel.sum() == 0:
            return batch

        # Update desired_goals
        batch_relabel_indices = relabel_indices[valid_relabel]
        valid_new_goals = new_goals[valid_relabel]

        batch["desired_goals"][batch_relabel_indices] = valid_new_goals

        # Update goal portion in obs and next_obs (goal is appended at the end)
        batch["obs"][batch_relabel_indices, obs_dim_without_goal:] = valid_new_goals
        batch["next_obs"][batch_relabel_indices, obs_dim_without_goal:] = valid_new_goals

        # Recompute sparse reward for relabeled transitions
        achieved = batch["next_achieved_goals"][batch_relabel_indices]
        desired = batch["desired_goals"][batch_relabel_indices]
        dist = torch.norm(achieved - desired, dim=-1)
        sparse_reward = (dist < self.goal_threshold).float().unsqueeze(-1)
        batch["rewards"][batch_relabel_indices] = sparse_reward

        return batch
