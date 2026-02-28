# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclass for SAC+HER runner."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class SacHerRunnerCfg:
    """Configuration for the SAC+HER off-policy runner."""

    seed: int = 42
    """Random seed."""

    device: str = "cuda:0"
    """Device for training."""

    # -- Network architecture
    actor_hidden_dims: list[int] = [256, 256, 256]
    """Hidden layer dimensions for the actor network."""

    critic_hidden_dims: list[int] = [256, 256, 256]
    """Hidden layer dimensions for each Q-network in the twin critic."""

    use_layer_norm: bool = True
    """Whether to use LayerNorm in networks."""

    # -- SAC hyperparameters
    actor_lr: float = 3e-4
    """Learning rate for the actor."""

    critic_lr: float = 3e-4
    """Learning rate for the twin Q-critic."""

    alpha_lr: float = 3e-4
    """Learning rate for the entropy temperature (auto_alpha mode)."""

    gamma: float = 0.99
    """Discount factor."""

    tau: float = 0.005
    """Soft target update coefficient."""

    auto_alpha: bool = True
    """Whether to automatically tune the entropy coefficient."""

    init_alpha: float = 0.2
    """Initial entropy coefficient (used if auto_alpha=False, or as starting value)."""

    batch_size: int = 2048
    """Mini-batch size for gradient updates."""

    # -- Off-policy settings
    buffer_capacity: int = 1_000_000
    """Maximum number of transitions in the replay buffer."""

    utd_ratio: int = 2
    """Update-to-data ratio: number of gradient steps per env step."""

    warmup_steps: int = 20_000
    """Number of random-action steps before training begins."""

    # -- HER settings
    use_her: bool = False
    """Whether to use Hindsight Experience Replay."""

    her_k: int = 4
    """Number of HER relabeled goals per original transition."""

    her_strategy: Literal["future"] = "future"
    """HER relabeling strategy. Only 'future' is supported."""

    goal_dim: int = 6
    """Dimensionality of the goal vector (achieved_goal / desired_goal)."""

    goal_threshold: float = 0.02
    """L2 distance threshold for sparse goal-conditioned reward."""

    # -- Training control
    max_steps: int = 10_000_000
    """Maximum environment steps for training."""

    save_interval: int = 100_000
    """Save checkpoint every N env steps."""

    log_interval: int = 5_000
    """Log metrics every N env steps."""

    # -- Experiment
    experiment_name: str = "sac_her"
    """Name for the experiment (used for log directory)."""

    run_name: str | None = None
    """Optional run name suffix."""

    logger: str = "wandb"
    """Logger to use: 'wandb' or 'tensorboard'."""

    wandb_project: str = "omnireset"
    """W&B project name."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    load_run: str | None = None
    """Run directory to load from."""

    load_checkpoint: str | None = None
    """Specific checkpoint file to load."""

    # -- Misc
    clip_actions: bool = True
    """Whether to clip actions to [-1, 1]."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
