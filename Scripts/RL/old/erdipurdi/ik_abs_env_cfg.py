from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

import isaaclab.envs.mdp as core_mdp
import os
import json
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.controllers import DifferentialIKControllerCfg


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
        super().__post_init__()

        # GPU capacity config
        self.sim.physx.found_lost_aggregate_pairs_capacity = 262144
        self.sim.physx.total_aggregate_pairs_capacity = 262144
        self.sim.physx.gpu_found_lost_pairs_capacity = 1048576
        self.sim.physx.gpu_total_pairs_capacity = 1048576

        # Franka config
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.rigid_props = RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        )

        self.scene.robot.actuators["panda_shoulder"].stiffness = 400.0
        self.scene.robot.actuators["panda_shoulder"].damping = 60.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 800.0
        self.scene.robot.actuators["panda_forearm"].damping = 60.0

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
        )

        # IK absolute controller
        self.actions.arm_action = core_mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )

        self.actions.gripper_action = core_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )

        # Observations
        self.observations.policy.ee_position = ObsTerm(
            func=core_mdp.body_pose_w,  # Mantenemos esta para el EE
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        self.observations.policy.obj_position = ObsTerm(
            func=core_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
            scale=(-1.0, -1.0, 1.0),
        )

        self.observations.policy.gripper_opening = ObsTerm(
            func=core_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])},
        )

        # Rewards
        self.rewards.object_gripping = RewTerm(
            func=core_mdp.contact_forces,
            weight=10000.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["panda_.*finger"]),
                "threshold": 0.1,
            },
        )

        self.rewards.lift_bonus = RewTerm(
            func=mdp.object_is_lifted,
            weight=25000.0,
            params={"minimal_height": 0.05, "object_cfg": SceneEntityCfg("object")},
        )

        self.rewards.reaching_object = RewTerm(
            func=mdp.object_ee_distance,
            weight=5000.0,
            params={
                "std": 0.05,
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        self.rewards.object_height_reward = RewTerm(
            func=mdp.object_is_lifted,
            weight=5000.0,
            params={"minimal_height": 0.01, "object_cfg": SceneEntityCfg("object")},
        )

        self.rewards.action_rate.weight = -0.05

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                )
            ],
        )

        # Cube init
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.commands.object_pose.body_name = "panda_hand"

        # Disable cube randomization
        self.events.reset_object_position = None

        if vlm_data:
            print(f"[VLM INFO] Loading JSON: {vlm_json_path}")

            # joint pos init
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
            """
            # joint velocities init
            if "get_joint_velocities" in vlm_data:
                joint_vels = vlm_data["get_joint_velocities"]
                self.scene.robot.init_state.joint_velocities = {
                    "panda_joint1": joint_vels[0],
                    "panda_joint2": joint_vels[1],
                    "panda_joint3": joint_vels[2],
                    "panda_joint4": joint_vels[3],
                    "panda_joint5": joint_vels[4],
                    "panda_joint6": joint_vels[5],
                    "panda_joint7": joint_vels[6],
                    "panda_finger_joint1": joint_vels[7],
                    "panda_finger_joint2": joint_vels[8],
                }
            """
            # cube pos init
            if "cube_world_pos" in vlm_data:
                v_pos = vlm_data["cube_world_pos"]
                cor_x, cor_y = -v_pos[0], -v_pos[1]
                self.scene.object.init_state.pos = (cor_x, cor_y, 0.06)
                print(f"[VLM INFO] Cube positioned at: {self.scene.object.init_state.pos}")
                print("[VLM INFO] Raw cube_world_pos:", v_pos)

            print("[VLM INFO] Robot joints:", self.scene.robot.init_state.joint_positions)
            # if "get_joint_velocities" in vlm_data:
            #    print("[VLM INFO] Robot joint velocities:", self.scene.robot.init_state.joint_velocities)

        # Disable cube pos randomization again, explicitly
        self.events.reset_object_position = None
