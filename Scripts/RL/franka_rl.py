import os
import sys

conda_env_path = "/home/unaiolaizolaosa/miniconda3/envs/PFG/lib/python3.11/site-packages"
if conda_env_path not in sys.path:
    sys.path.append(conda_env_path)

try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
print("Starting RL Setup!")

# Dependencies for the franka rl task
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import PPO # algorithm we will use
import os
# from isaacsim.gym.vec_env import TaskStopWatch
import asyncio
from omni.isaac.franka import Franka
from omni.isaac.core.world import World
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.scenes.scene import Scene
import torch
import argparse
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.types import ArticulationAction

# In case parallelization is specified from the command, I will add the functionality to spawn multiple environments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args, _ = parser.parse_known_args()

class FrankaPickRL(gym.Env):
    # RL environment creation class
    def __init__(self, num_envs):
        super().__init__()

        self.num_envs = num_envs
        self.world = World(stage_units_in_meters=1.0)

        isaac_root = "/home/unaiolaizolaosa/isaac-sim-5.1.0"
        asset_path = "/home/unaiolaizolaosa/isaac-sim-5.1.0/assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        # load the end state of the previous VLM task (joint positions + obtained coordinates)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "../Control/", "rl_initial_state.json")

        with open(json_path, "r") as f:
            self.vlm_data = json.load(f)

        # code for parallelization
        cloner = GridCloner(spacing=2.0)
        target_paths = [f"/World/Env_{i}" for i in range(self.num_envs)]
        cloner.define_base_env(base_env_path="/World/Env_0")

        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Env_0/Franka")
        cube_pos = np.array(self.vlm_data["cube_target"])
        self.cube_initial_pos = cube_pos

        DynamicCuboid(
                prim_path = "/World/Env_0/Cube",
                name="cube_0",
                position=cube_pos,
                size=.05,
                color=np.array([1,0,0]) # red xd
        )

        cloner.clone(source_prim_path="/World/Env_0", prim_paths=target_paths)

        self.robots = ArticulationView(prim_paths_expr="/World/Env_*/Franka", name="franka_view")
        self.cubes = RigidPrimView(prim_paths_expr="/World/Env_*/Cube", name="cube_view")

        self.world.scene.add(self.robots)
        self.world.scene.add(self.cubes)
        self.world.scene.add_default_ground_plane()
        
        # action space
        self.action_space = gym.spaces.Box(low=-.05, high=.05, shape=(7,), dtype=np.float32)

        # observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def get_observations(self):
        joints = self.robots.get_joint_positions()[0][:7]
        cube_pos, _ = self.cubes.get_world_poses()
        robot_pos, _ = self.robots.get_world_poses()
        vec = cube_pos[0] - robot_pos[0]
        return np.concatenate([joints, vec, [0,0]]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
            
        init_pos = np.array(self.vlm_data["joint_positions"])
        if len(init_pos) == 7:
            init_pos = np.concatenate([init_pos, [0,0]]) # padding if necessary

        all_init_joints=np.tile(init_pos, (self.num_envs, 1))
        self.robots.set_joint_positions(all_init_joints)
        self.robots.set_joint_velocities(np.zeros((self.num_envs, 9)))
        self.world.step(render=True)
        return self.get_observations(), {}

    def step(self, action):
        current_joints = self.robots.get_joint_positions()
        new_joints = current_joints.copy()
        
        action_batch = np.array(action).reshape(-1, 7)
        new_joints[:, :7] += action_batch
        
        self.robots.set_joint_positions(new_joints)
        self.world.step(render=True)

        obs = self.get_observations()
        cube_poses, _ = self.cubes.get_world_poses()
        cube_height = cube_poses[0][2]

        reward = 0
        if cube_height > 0.03:
            reward += 10.0 + (cube_height * 50)

        robot_pos, _ = self.robots.get_world_poses()
        dist = np.linalg.norm(cube_poses[0] - robot_pos[0])
        reward -= dist

        done = cube_height > 0.2
        return obs, reward, bool(done), False, {}

if __name__ == "__main__":
    env = FrankaPickRL(num_envs=args.num_envs)

    # Mlpolicy for numbers + verbose + storing the tensorboard logs in the corresponding path
    model = PPO("MlpPolicy",
                env,
                device="cuda",
                n_steps=1024,
                n_epochs=10,
                learning_rate=3e-4,
                verbose=1,
                batch_size=512,
                tensorboard_log="/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/tensorboard_franka/ppo_franka")

    steps = 100000
    print(f"Starting training for {steps} steps.")

    # If I were to manually stop (interrupt the process) -> save the results anyway
    try:
        model.learn(total_timesteps=steps)
    except KeyboardInterrupt:
        print("Training process interrupted -> Saving progress anyway")

    # And we now save the model
    model.save("/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/models_franka/model_pick")
    print("Model saved in the results directory!")

    # I will save the model both in pickle format and in onnx in case there is needed in the future
    dummy_input = torch.randn(1,12)
    torch.onnx.export(model.policy, dummy_input, "/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/models_franka/model_pick_onnx",
                      opset_version=11)

    # Finally close the simultion
    simulation_app.close()
