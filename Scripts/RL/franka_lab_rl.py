import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka RL - Isaaclab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. GROUP B: Everything else (Must come AFTER simulation_app)
# # 2. GROUP B: Everything else
import torch
import os
import json
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg

# This is the direct path for 5.1 spawners
# Use 'shapes' instead of 'meshes' for primitives like Cubes
import isaaclab.sim.spawners.shapes as shape_spawners
import isaaclab.sim.spawners.materials as mat_spawners

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

# Robotics and RL imports
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper


@configclass
class FrankaPickCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, device="cuda")
    observation_space = 13
    action_space = 7
    episode_length_s = 10.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=2.5
    )

    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=shape_spawners.CuboidCfg(  # Changed from MeshCubeCfg to CuboidCfg
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=mat_spawners.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            ),
        ),
    )

    decimation = 2


class FrankaRLEnv(DirectRLEnv):
    cfg: FrankaPickCfg

    def __init__(self, cfg: FrankaPickCfg, render_mode: str | None = None, **kwargs):
        json_path = os.path.join(
            os.path.dirname(__file__), "../Control/rl_initial_state.json"
        )
        with open(json_path, "r") as f:
            self.vlm_data = json.load(f)

        self.init_joint_pos = torch.tensor(
            self.vlm_data["joint_positions"], device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.scene.objects["cube"] = self.cube

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, :7]
        robot_pos = self.robot.data.root_pos_w[:, :3]
        cube_pos = self.cube.data.root_pos_w[:, :3]
        to_cube = cube_pos - robot_pos
        obs = torch.cat([joint_pos, to_cube, robot_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        hand_pos = self.robot.data.root_pos_w[:, :3]
        cube_pos = self.cube.data.root_pos_w[:, :3]
        dist = torch.norm(cube_pos - hand_pos, dim=-1)
        reaching_reward = 1.0 / (1.0 + dist * dist)
        cube_height = cube_pos[:, 2]
        lift_bonus = torch.where(cube_height > 0.03, 10.0 + (cube_height * 50.0), 0.0)
        return (reaching_reward * 2.0) + lift_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self.cube.data.root_pos_w[:, :3]
        died = cube_pos[:, 2] > 0.5
        time_out = self.episode_sums["steps"] >= (
            self.cfg.episode_length_s / self.cfg.sim.dt / self.cfg.decimation
        )
        return died, torch.tensor(time_out, device=self.device, dtype=torch.bool)

    def _reset_idx(self, env_ids: torch.Tensor):
        super().reset_idx(env_ids)
        joint_pos = self.init_joint_pos.repeat(len(env_ids), 1)
        if joint_pos.shape[1] == 7:
            joint_pos = torch.cat(
                [joint_pos, torch.zeros((len(env_ids), 2), device=self.device)], dim=-1
            )
        self.robot.write_joint_state_to_sim(joint_pos, env_ids=env_ids)

    def _step_impl(self, actions: torch.Tensor):
        current_joints = self.robot.data.joint_pos.clone()
        current_joints[:, :7] += actions
        self.robot.set_joint_position_target(current_joints)


if __name__ == "__main__":
    env_cfg = FrankaPickCfg()
    env = FrankaRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="pfg_project/Scripts/RL/results/tensorboard_franka/ppo_franka",
    )

    print("Starting training...")
    try:
        model.learn(total_timesteps=100000)
    finally:
        model.save("pfg_project/Scripts/RL/results/models_franka/model_pick")
        simulation_app.close()
