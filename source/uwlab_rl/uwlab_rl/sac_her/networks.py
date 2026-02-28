# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network architectures for SAC: GaussianActor and TwinQCritic."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, use_layer_norm: bool = True) -> nn.Sequential:
    """Build an MLP with optional LayerNorm after each hidden layer."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy for SAC.

    Outputs mean and log_std, then samples via reparameterization trick
    and applies tanh squashing. Returns (action, log_prob).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], use_layer_norm: bool = True):
        super().__init__()
        self.trunk = _build_mlp(obs_dim, hidden_dims, hidden_dims[-1], use_layer_norm)
        # Remove the final Linear from trunk — we split into mean and log_std heads
        # Actually, _build_mlp already adds a final linear, so let's build differently
        self.backbone = _build_mlp(obs_dim, hidden_dims, hidden_dims[-1], use_layer_norm)
        # Override: remove the final Linear layer from backbone, keep up to last ReLU
        backbone_layers: list[nn.Module] = []
        last_dim = obs_dim
        for h in hidden_dims:
            backbone_layers.append(nn.Linear(last_dim, h))
            if use_layer_norm:
                backbone_layers.append(nn.LayerNorm(h))
            backbone_layers.append(nn.ReLU())
            last_dim = h
        self.backbone = nn.Sequential(*backbone_layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns (squashed_action, log_prob)."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparameterization trick
        x_t = dist.rsample()
        # Squash through tanh
        action = torch.tanh(x_t)
        # Log-prob with tanh correction
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (tanh of mean) for evaluation."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        return torch.tanh(mean)


class TwinQCritic(nn.Module):
    """Twin Q-network for SAC (clipped double-Q trick).

    Two independent Q-networks that each map (obs, action) -> scalar Q-value.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], use_layer_norm: bool = True):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q1 = _build_mlp(input_dim, hidden_dims, 1, use_layer_norm)
        self.q2 = _build_mlp(input_dim, hidden_dims, 1, use_layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1(s,a), Q2(s,a))."""
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 (used for actor loss to save compute)."""
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa)
