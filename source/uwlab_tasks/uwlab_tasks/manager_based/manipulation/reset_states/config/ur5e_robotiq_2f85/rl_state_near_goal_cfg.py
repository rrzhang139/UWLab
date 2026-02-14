# Near-goal training config: only 2 resets closest to goal
# 1. ObjectPartiallyAssembledEEGrasped (near-goal, 50%)
# 2. ObjectRestingEEGrasped (resting + grasping, 50%)

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .rl_state_cfg import (
    BaseEventCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
    Ur5eRobotiq2f85RelCartesianOSCEvalCfg,
)
from .actions import Ur5eRobotiq2f85RelativeOSCAction
from isaaclab.managers import SceneEntityCfg

from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85


@configclass
class NearGoalTrainEventCfg(BaseEventCfg):
    """Training events with only 2 near-goal resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectPartiallyAssembledEEGrasped",
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectRestingEEGrasped",
            ],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCNearGoalTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Training with only 2 near-goal resets, from scratch."""

    events: NearGoalTrainEventCfg = NearGoalTrainEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Re-add the actuator randomization that parent sets in __post_init__
        self.events.randomize_robot_actuator_parameters = EventTerm(
            func=task_mdp.randomize_operational_space_controller_gains,
            mode="reset",
            params={
                "action_name": "arm",
                "stiffness_distribution_params": (0.7, 1.3),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
