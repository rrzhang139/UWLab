# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl SAC agent configuration for OmniReset UR5e Robotiq 2F85 task."""

SAC_RUNNER_CFG = {
    "seed": 42,
    # -- models
    "models": {
        "separate": True,
        "policy": {
            "class": "GaussianMixin",
            "clip_actions": False,
            "clip_log_std": True,
            "initial_log_std": 0.0,
            "min_log_std": -20.0,
            "max_log_std": 2.0,
            "input_shape": "Shape.STATES",
            "hiddens": [256, 256, 256],
            "hidden_activation": ["relu", "relu", "relu"],
            "output_shape": "Shape.ACTIONS",
            "output_activation": "tanh",
            "output_scale": 1.0,
        },
        "critic_1": {
            "class": "DeterministicMixin",
            "clip_actions": False,
            "input_shape": "Shape.STATES_ACTIONS",
            "hiddens": [256, 256, 256],
            "hidden_activation": ["relu", "relu", "relu"],
            "output_shape": "Shape.ONE",
            "output_activation": "",
            "output_scale": 1.0,
        },
        "critic_2": {
            "class": "DeterministicMixin",
            "clip_actions": False,
            "input_shape": "Shape.STATES_ACTIONS",
            "hiddens": [256, 256, 256],
            "hidden_activation": ["relu", "relu", "relu"],
            "output_shape": "Shape.ONE",
            "output_activation": "",
            "output_scale": 1.0,
        },
    },
    # -- memory (replay buffer)
    "memory": {
        "class": "RandomMemory",
        "memory_size": 1000000,
    },
    # -- agent
    "agent": {
        "class": "SAC",
        "gradient_steps": 2,
        "batch_size": 2048,
        "discount_factor": 0.99,
        "polyak": 0.005,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": {},
        "state_preprocessor": "RunningStandardScaler",
        "state_preprocessor_kwargs": None,
        "random_timesteps": 1000,
        "learning_starts": 1000,
        "grad_norm_clip": 1.0,
        "learn_entropy": True,
        "entropy_learning_rate": 3e-4,
        "initial_entropy_value": 1.0,
        "target_entropy": None,
        "rewards_shaper_scale": 1.0,
        "experiment": {
            "directory": "ur5e_robotiq_2f85_sac",
            "experiment_name": "",
            "write_interval": 100,
            "checkpoint_interval": 1000,
            "wandb": True,
            "wandb_kwargs": {
                "project": "omnireset",
            },
        },
    },
    # -- trainer
    "trainer": {
        "class": "SequentialTrainer",
        "timesteps": 500000,
        "environment_info": "log",
    },
}
