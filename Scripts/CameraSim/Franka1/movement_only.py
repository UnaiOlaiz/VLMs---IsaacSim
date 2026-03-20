# Script 3: Movement only — reads coords from JSON and moves Franka_1
# Run inside Isaac Sim AFTER killing BentoML
# BentoML must NOT be running when this script executes

import numpy as np
import asyncio
import omni.usd
from pxr import UsdGeom
from omni.physx.scripts import utils
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.articulations import Articulation
import sys
import os
import json


scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"

if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Control.franka_stop import execute_movement
    print(f"Movement scripts correctly loaded from path: '{scripts_path}'!")
except ImportError as e:
    print(f"Error loading scripts from given path: {scripts_path}")
    raise e

# Collision setup — skip fingers
stage = omni.usd.get_context().get_stage()
robot_path = "/World/Franka_1"
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if prim.IsA(UsdGeom.Mesh) and robot_path in path_str:
        if "finger" in path_str.lower():
            continue
        utils.setCollider(prim, approximationShape="convexHull")

# Path to the coordinates saved by run_vlm_standalone.py
JSON_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_first_franka.json")
GRASP_HEIGHT = 0.045  # end effector height above cube


async def run():
    # Load coordinates
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: JSON not found at {JSON_PATH}")
        print("Run save_camera_data.py then run_vlm_standalone.py first!")
        return

    with open(JSON_PATH) as f:
        data = json.load(f)

    target_pos = np.array(data["target_pos"])
    print(f"Loaded target position: {target_pos}")

    # Move to pre-grasp position
    final_coords = [target_pos[0], target_pos[1], GRASP_HEIGHT]
    print(f"Moving to pre-grasp position: {final_coords}")

    await execute_movement(final_coords)
    print("Arm located at pre-grasp position! Ready for RL task!")

    # Capture joint state for RL initialization
    franka = Articulation("/World/Franka_1")
    franka.initialize()

    joint_pos = franka.get_joint_positions().tolist()
    joint_vel = franka.get_joint_velocities().tolist()

    # Update JSON with joint state
    data["joint_positions"] = joint_pos
    data["joint_velocities"] = joint_vel

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"--- SUCCESS: Full state stored at {JSON_PATH} ---")


asyncio.ensure_future(run())