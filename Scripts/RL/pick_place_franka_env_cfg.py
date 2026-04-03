from __future__ import annotations

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Import your newly created MDP functions
from . import mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import PANDA_CFG  # Standard Franka Panda

@configclass
class FrankaPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(env_spacing=2.5) # I will set the num_envs in the training call
    robot: AssetBaseCfg = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
            prim_path = "{ENV_REGEX_NS}/Object",
            spawn=UsdFileCfg(
                usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/dex_cube_dynamic.usd",
                scale=(.8, .8, .8),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(.5, .0, .05)),
    )

    def __post_init__(self):
        self.viewer_eye = (3.0, 3.0, 3.0)
        
        # observation space
        self.observations.policy.obj_pos = ObsTerm(
                func=mdp.object_pos,
                params={"eef_link_name": "panda_hand"}
        )

        self.observations.policy.arm_join_pos = ObsTerm(
                func=mdp.get_robot_joint_state,
                params={"joint_names": ["panda_joint.*"]}
        )

        # terminations
        self.terminations.success = DoneTerm(
                func=mdp.task_done_pick_place,
                params={
                    "object_cfg": SceneEntityCfg("object"),
                    "min_x": .4, "max_x": .7,
                    "min_y": -.15, "max_y": .15
                }
        )

        # object position reset event
        self.events.reset_object = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                    "asset_cfg": SceneEntityCfg("object"),
                },
        )

        # rewards system
        self.rewards.reaching_object = RewTerm(
                func=mdp.object__ee_distance,
                weight=-1.0,
                params={"object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("robot", body_names="panda_hand")},
        )
