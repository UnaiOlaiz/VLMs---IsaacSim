from __future__ import annotations
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from . import mdp
from isaaclab_assets.robots.franka import PANDA_CFG
import isaaclab.envs.mdp as isaaclab_mdp


@configclass
class FrankaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        # Fixed: Viewer and Observation function names
        self.viewer.eye = (3.0, 3.0, 3.0)

        self.observations.policy.obj_obs = ObsTerm(
            func=mdp.object_obs, params={"eef_link_name": "panda_hand"}
        )

        self.observations.policy.arm_joints = ObsTerm(
            func=mdp.get_robot_joint_state, params={"joint_names": ["panda_joint.*"]}
        )

        self.events.reset_scenario = EventTerm(
            func=mdp.reset_to_predefined_pose, mode="reset"
        )

        self.terminations.success = DoneTerm(
            func=mdp.task_done_pick_place,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "task_link_name": "panda_hand",
            },
        )
        # Rewards
        self.rewards.reaching_object = RewTerm(
            func=isaaclab_mdp.rewards.object_ee_distance,  # Added .rewards sub-module
            weight=-1.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )
