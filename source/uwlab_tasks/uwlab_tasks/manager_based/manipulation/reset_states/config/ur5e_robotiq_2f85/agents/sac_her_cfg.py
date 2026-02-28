# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC+HER agent configuration for OmniReset UR5e Robotiq 2F85 task."""

from isaaclab.utils import configclass

from uwlab_rl.sac_her.runner_cfg import SacHerRunnerCfg


@configclass
class SacHerOmniResetCfg(SacHerRunnerCfg):
    """SAC+HER configuration tuned for OmniReset peg insertion."""

    seed = 42
    experiment_name = "ur5e_robotiq_2f85_sac_her"

    # Networks — matching scale to PPO baseline
    actor_hidden_dims = [256, 256, 256]
    critic_hidden_dims = [256, 256, 256]
    use_layer_norm = True

    # SAC
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    auto_alpha = True
    init_alpha = 0.2
    batch_size = 2048

    # Off-policy
    buffer_capacity = 1_000_000
    utd_ratio = 2
    warmup_steps = 20_000

    # HER — Phase 1: off (dense reward), Phase 2: on (sparse reward)
    use_her = False
    her_k = 4
    her_strategy = "future"
    goal_dim = 6
    goal_threshold = 0.02

    # Training
    max_steps = 10_000_000
    save_interval = 100_000
    log_interval = 5_000

    # Logging
    logger = "wandb"
    wandb_project = "omnireset"

    # Misc
    clip_actions = True
    max_grad_norm = 1.0
