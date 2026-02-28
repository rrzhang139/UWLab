# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI argument helpers for SAC+HER training scripts."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uwlab_rl.sac_her import SacHerRunnerCfg


def add_sac_her_args(parser: argparse.ArgumentParser):
    """Add SAC+HER-specific arguments to the parser."""
    arg_group = parser.add_argument_group("sac_her", description="Arguments for SAC+HER agent.")
    # -- experiment
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load/resume
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb."
    )


def update_sac_her_cfg(agent_cfg: SacHerRunnerCfg, args_cli: argparse.Namespace) -> SacHerRunnerCfg:
    """Update SAC+HER config with CLI arguments."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.log_project_name is not None:
        agent_cfg.wandb_project = args_cli.log_project_name
    return agent_cfg
