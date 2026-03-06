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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg

# path of the 'old' experiment json vlm_json_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_initial_state.json")  # path of end coordinates
vlm_json_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_start_near_cube_v2.json")


def load_vlm_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


vlm_data = load_vlm_json(vlm_json_path)


# helper functions
def _finger_closed_reward(env, robot_cfg: SceneEntityCfg, closed_thresh):
    """1.0 if the gripper is (nearly) closed, else 0.0."""
    robot = env.scene[robot_cfg.name]
    finger_q = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, 2) typically
    closed = (finger_q < closed_thresh).all(dim=1)
    return closed.float()


def ee_height_penalty(env, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, margin: float = 0.03):
    obj_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    ee_z = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, 2]
    return torch.clamp(ee_z - obj_z - margin, min=0.0)


def xy_alignment_reward(env, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, std: float = 0.015):
    obj_xy = env.scene[object_cfg.name].data.root_pos_w[:, :2]
    ee_xy = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :2]
    dist = torch.norm(obj_xy - ee_xy, dim=1)
    return torch.exp(-0.5 * (dist / std) ** 2)


def close_when_near(
    env,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_finger_cfg: SceneEntityCfg,
    std: float = 0.05,
    closed_thresh: float = 0.02,
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
    minimal_height: float = 0.04,
    closed_thresh: float = 0.025,
):
    obj = env.scene[object_cfg.name]
    z_dist = obj.data.root_pos_w[:, 2] - 0.055
    lifted = (z_dist > minimal_height).float()
    robot = env.scene[robot_finger_cfg.name]
    finger_q = robot.data.joint_pos[:, robot_finger_cfg.joint_ids]
    closed = (finger_q.sum(dim=1) < 0.05).float()
    return lifted * closed


def ee_above_object_reward(
    env,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    target_clearance: float = 0.02,
    std: float = 0.02,
):
    obj = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]

    # EE position from frame transformer
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]  # (N,3) first target frame
    ee_z = ee_pos[:, 2]

    z_err = ee_z - (obj_z + target_clearance)
    return torch.exp(-0.5 * (z_err / std) ** 2)


def close_far_penalty(
    env,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_finger_cfg: SceneEntityCfg,
    std: float = 0.06,
    closed_thresh: float = 0.02,
):
    near = mdp.object_ee_distance(env, std=std, object_cfg=object_cfg, ee_frame_cfg=ee_frame_cfg)
    closed = _finger_closed_reward(env, robot_finger_cfg, closed_thresh=closed_thresh)
    # penalty when closed but not near
    return closed * (1.0 - near)


def ee_position_w_obs(env, ee_frame_cfg: SceneEntityCfg):
    """End-effector position in world frame. Returns (num_envs, 3)."""
    ee_tf = env.scene[ee_frame_cfg.name]
    # FrameTransformer stores target frame positions as (N, num_targets, 3)
    ee_pos = ee_tf.data.target_pos_w[:, 0, :]
    return ee_pos


def ee_to_object_w_obs(env, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    """Vector from EE -> object in world frame. Returns (num_envs, 3)."""
    ee_pos = ee_position_w_obs(env, ee_frame_cfg)
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w
    return obj_pos - ee_pos


def gripper_opening_obs(env, robot_cfg):
    robot = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return q.sum(dim=1, keepdim=True)


def ee_height_error_obs(env, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, target_clearance: float = 0.0):
    obj = env.scene[object_cfg.name]
    obj_z_center = obj.data.root_pos_w[:, 2]
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    ee_z = ee_pos[:, 2]
    err = (ee_z - (obj_z_center + target_clearance)).unsqueeze(1)
    return err


def open_after_grasp_penalty(env, object_cfg: SceneEntityCfg, robot_finger_cfg: SceneEntityCfg):
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    lifted = z > 0.01

    robot = env.scene[robot_finger_cfg.name]
    finger_q = robot.data.joint_pos[:, robot_finger_cfg.joint_ids]
    open_amount = finger_q.mean(dim=1)

    return lifted.float() * open_amount


def strong_upward_reward(
    env, object_cfg: SceneEntityCfg, robot_finger_cfg: SceneEntityCfg, closed_thresh: float = 0.01
):
    obj = env.scene[object_cfg.name]
    vz = obj.data.root_lin_vel_w[:, 2]

    closed = _finger_closed_reward(env, robot_finger_cfg, closed_thresh=closed_thresh)
    return closed * torch.clamp(vz, min=0.0)


def shaped_object_height_reward(env, object_cfg: SceneEntityCfg, table_height: float = 0.055):
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    return torch.clamp(z - table_height, min=0.0)


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
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)

        self.scene.robot.spawn.activate_contact_sensors = True

        self.scene.robot.spawn.rigid_props = RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        )

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger", update_period=0.0, history_length=3, debug_vis=False
        )

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
                "panda_finger_joint1": joint_positions[7],
                "panda_finger_joint2": joint_positions[8],
            }
            self.scene.robot.default_joint_pos = self.scene.robot.init_state.joint_positions

        cube_position = [0.4054, 0.0, 0.055]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=False
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )

        finger_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])

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

        self.scene.object.init_state.pos = cube_position

        # Rewards design
        self.rewards.reaching_object.weight = 100.0
        self.rewards.reaching_object.params["std"] = 0.05

        self.rewards.close_when_near = RewTerm(
            func=close_when_near,
            weight=200.0,
            params={
                "std": 0.02,
                "closed_thresh": 0.02,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.low_lift_bonus = RewTerm(
            func=object_is_lifted_and_closed,
            weight=500.0,
            params={
                "minimal_height": 0.01,
                "closed_thresh": 0.03,
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.high_lift_bonus = RewTerm(
            func=object_is_lifted_and_closed,
            weight=1000.0,
            params={
                "minimal_height": 0.08,
                "closed_thresh": 0.025,
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.object_gripping = RewTerm(
            func=mdp.contact_forces,
            weight=300.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["panda_leftfinger", "panda_rightfinger"]),
                "threshold": 0.20,
            },
        )

        """
        self.rewards.ee_above_object = RewTerm(
            func=ee_above_object_reward,
            weight=200.0,
            params={
                "target_clearance": 0.005,
                "std": 0.02,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        """

        self.rewards.close_far_penalty = RewTerm(
            func=close_far_penalty,
            weight=-100.0,
            params={
                "std": 0.06,
                "closed_thresh": 0.02,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.open_after_grasp_penalty = RewTerm(
            func=open_after_grasp_penalty,
            weight=-200.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
            },
        )

        self.rewards.strong_upward = RewTerm(
            func=strong_upward_reward,
            weight=150.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_finger_cfg": finger_cfg,
                "closed_thresh": 0.02,
            },
        )

        self.rewards.ee_height_penalty = RewTerm(
            func=ee_height_penalty,
            weight=-5.0,
            params={
                "margin": 0.022,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        """
        self.rewards.xy_alignment = RewTerm(
            func=xy_alignment_reward,
            weight=250.0,
            params={
                "std": 0.015,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        """

        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

        self.rewards.shape_lift = RewTerm(
            func=shaped_object_height_reward,
            weight=5000.0,
            params={"object_cfg": SceneEntityCfg("object"), "table_height": 0.055},
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "{ENV_REGEX_NS}/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_frame",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.observations.policy.gripper_opening = ObsTerm(
            func=gripper_opening_obs,
            params={
                "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            },
        )

        self.observations.policy.ee_height_error = ObsTerm(
            func=ee_height_error_obs,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "target_clearance": 0.0,
            },
        )

        self.observations.policy.ee_position = ObsTerm(
            func=ee_position_w_obs,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        self.observations.policy.ee_to_object = ObsTerm(
            func=ee_to_object_w_obs,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("object")},
        )

        self.observations.policy.finger_dist = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])}
        )

        # self.events.reset_object_position = None
        # self.events.reset_robot_joints = None
        # if hasattr(self.events, "randomize_object"):
        #    self.events.randomize_object = None
        #
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger_joint.*"]),
            },
        )
        if hasattr(self.events, "randomize_object"):
            self.events.randomize_object = None

        self.commands.object_pose = None
        if hasattr(self.observations, "policy") and hasattr(self.observations.policy, "target_object_position"):
            self.observations.policy.target_object_position = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None

        if hasattr(self.events, "reset_all"):
            self.events.reset_all = None


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
