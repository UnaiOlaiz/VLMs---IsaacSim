# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import isaaclab.envs.mdp as core_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# --- VLM Data Loading ---
vlm_json_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_start_near_cube_v2.json")

def load_vlm_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

vlm_data = load_vlm_json(vlm_json_path)

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # 1. Run parent post-init first to load default rewards and observations
        super().__post_init__()

        # GPU capacity config
        self.sim.physx.found_lost_aggregate_pairs_capacity = 262144
        self.sim.physx.total_aggregate_pairs_capacity = 262144
        self.sim.physx.gpu_found_lost_pairs_capacity = 1048576
        self.sim.physx.gpu_total_pairs_capacity = 1048576

        # 2. Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # This enables collisions so your manual Convex Decomposition is used
        self.scene.robot.spawn.collision_props = CollisionPropertiesCfg(collision_enabled=True)

        # 3. Set actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        self.commands.object_pose.body_name = "panda_hand"

        # 4. Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # 5. End Effector Frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        # 6. INTEGRATE VLM LOGIC
        if vlm_data:
            print(f"[VLM INFO] Loading initialization from JSON")
            jp = vlm_data["joint_positions"]
            self.scene.robot.init_state.joint_positions = {
                f"panda_joint{i+1}": jp[i] for i in range(7)
            }
            self.scene.robot.init_state.joint_positions["panda_finger_joint1"] = jp[7]
            self.scene.robot.init_state.joint_positions["panda_finger_joint2"] = jp[8]
            self.scene.robot.default_joint_pos = self.scene.robot.init_state.joint_positions

            if "cube_world_pos" in vlm_data:
                cp = vlm_data["cube_world_pos"]
                self.scene.object.init_state.pos = (-cp[0], -cp[1], 0.06)

        # Disable randomization for VLM starts
        self.events.reset_object_position = None


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False