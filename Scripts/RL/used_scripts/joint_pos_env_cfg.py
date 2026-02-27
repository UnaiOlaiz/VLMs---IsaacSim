# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# I will place my code to modify the initial position of both the arm (franka) and the cube

import os
import json
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

vlm_json_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_initial_state.json")  # path of end coordinates


def load_vlm_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


vlm_data = load_vlm_json(vlm_json_path)


# helper functions
def _finger_closed_reward(env, robot_cfg: SceneEntityCfg, closed_thresh: float = 0.01):
    """1.0 if the gripper is (nearly) closed, else 0.0."""
    robot = env.scene[robot_cfg.name]
    finger_q = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, 2) typically
    closed = (finger_q < closed_thresh).all(dim=1)
    return closed.float()


def close_when_near(
    env,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_finger_cfg: SceneEntityCfg,
    std: float = 0.05,
    closed_thresh: float = 0.01,
):
    """
    Reward is high only when:
      - EE is close to object (shaped by exp distance via mdp.object_ee_distance)
      - AND gripper is closed.
    """
    near = mdp.object_ee_distance(env, std=std, object_cfg=object_cfg, ee_frame_cfg=ee_frame_cfg)
    closed = _finger_closed_reward(env, robot_finger_cfg, closed_thresh=closed_thresh)
    return near * closed


def object_is_lifted_and_closed(
    env,
    object_cfg: SceneEntityCfg,
    robot_finger_cfg: SceneEntityCfg,
    minimal_height: float = 0.05,
    closed_thresh: float = 0.01,
):
    """
    1.0 if object z > minimal_height AND gripper is closed, else 0.0
    (this prevents 'lift reward' being paid when you haven't grasped).
    """
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    lifted = (z > minimal_height).float()
    closed = _finger_closed_reward(env, robot_finger_cfg, closed_thresh=closed_thresh)
    return lifted * closed


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.physx.found_lost_aggregate_pairs_capacity = 262144
        self.sim.physx.total_aggregate_pairs_capacity = 262144
        self.sim.physx.gpu_found_lost_pairs_capacity = 1048576
        self.sim.physx.gpu_total_pairs_capacity = 1048576
        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Initialization of VLM data joint positions
        if vlm_data:
            joint_positions = vlm_data["joint_positions"]
            self.scene.robot.init_state.joint_positions = {
                "panda_joint1": joint_positions[0],
                "panda_joint2": joint_positions[1],
                "panda_joint3": joint_positions[2],
                "panda_joint4": joint_positions[3],
                "panda_joint5": joint_positions[4],
                "panda_joint6": joint_positions[5],
                "panda_joint7": joint_positions[6],
                "panda_finger_joint.*": 0.04,
            }

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1.00, use_default_offset=False
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])

        # Code for cube VLM coordinates initialization
        cube_position = [0.50, 0.0, 0.055]  # abitrary
        if vlm_data and "cube_target" in vlm_data:
            cube_position = vlm_data["cube_target"]

        if cube_position[2] < 0.03:
            cube_position[2] = 0.055

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=cube_position, rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Rewards design
        self.rewards.reaching_object.weight = 25.0

        self.rewards.close_when_near = RewTerm(
            func=close_when_near,
            weight=80.0,
            params={
                "std": 0.06,
                "closed_thresh": 0.01,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.low_lift_bonus = RewTerm(
            func=object_is_lifted_and_closed,
            weight=200.0,
            params={
                "minimal_height": 0.01,
                "closed_thresh": 0.01,
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.high_lift_bonus = RewTerm(
            func=object_is_lifted_and_closed,
            weight=600.0,
            params={
                "minimal_height": 0.05,
                "closed_thresh": 0.01,
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.object_gripping = None

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
