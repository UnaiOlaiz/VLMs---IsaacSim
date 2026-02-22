# File to test training NNs
# Dependencies
from omni.isaac.kit import SimulationApp

simulationApp = SimulationApp({"headless": True})

import os
from stable_baselines3 import PPO
from franka_rl import FrankaPickRL


env = FrankaPickRL(num_envs=1)
model_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts/RL/results/models_franka/model_pick.zip"

if os.path.exists(model_path):
    model = PPO.load(model_path, env)
    print("Model correctly loaded! Test time!")

    obs, _ = env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

else:
    print(f"No model found at specified path: {model_path}")

simulationApp.close()
