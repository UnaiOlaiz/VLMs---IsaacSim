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

class FrankaPickRL(gym.Env):
    # RL environment creation class
    def __init__(self):
        super().__init__()

        isaac_root = "/home/unaiolaizolaosa/isaac-sim-5.1.0"
        asset_path = "/home/unaiolaizolaosa/isaac-sim-5.1.0/assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        # set up the world inside Isaac
        self.world = World(stage_units_in_meters = 1.0)
        self.robot = Franka(prim_path="/World/Franka_Robot", name="franka_rl", usd_path = asset_path)
        self.world.scene.add(self.robot)
        self.world.scene.add_default_ground_plane()

        # load the end state of the previous VLM task (joint positions + obtained coordinates)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "..", "rl_initial_state.json")
        with open(json_path, "r") as f:
            self.vlm_data = json.load(f)

        self.target_pos = np.array(self.vlm_data["cube_target"])

        # action space -> 7 joints
        self.action_space = gym.spaces.Box(low=-.05, high=0.05, shape=(7,), dtype=np.float32)

        # observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def get_observations(self):
        joints = self.robot.get_joint_positions()
        end_pos, _ = self.robot.end_effector.get_world_pose()
        vec = self.target_pos - end_pos
        return np.concatenate([joints, vec]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.robot.set_joint_positions(np.array(self.vlm_data["joint_positions"]))
            
        init_pos = np.array(self.vlm_data["joint_positions"])
        if len(init_pos) == 7:
            init_pos = np.concatenate([init_pos, [0,0]])
        self.robot.set_joint_positions(init_pos)
        self.robot.set_joint_velocities(np.zeros(9))
        self.world.step(render=True)
        return self.get_observations(), {}

    def step(self,action):
        current_joints = self.robot.get_joint_positions()
        new_joints = current_joints.copy()
        new_joints[:7] += action
        self.robot.apply_action(ArticulationAction(joint_positions=new_joints))
        self.world.step(render=True)

        obs = self.get_observations()
        end_pos = self.robot.end_effector.get_world_pose()[0]
        dist = np.linalg.norm(self.target_pos - end_pos)

        reward = -dist
        done = dist < 0.02
        truncated = False

        return obs, reward, done, truncated, {}

if __name__ == "__main__":
    print(">>> CHECKPOINT 1: Initializing Environment...")
    env = FrankaPickRL()

    print(">>> CHECKPOINT 2: Environment Initialized. Setting up PPO...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka")

    steps = 100000
    print(f">>> CHECKPOINT 3: Starting training for {steps} steps.")
    model.learn(total_timesteps=steps)
