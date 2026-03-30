import numpy as np
import asyncio
import sys
import os
import json

scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from Scripts.Control.franka_controller import execute_movement

JSON_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_first_franka.json")
GRASP_HEIGHT = 0.045

async def run():
    with open(JSON_PATH) as f:
        data = json.load(f)

    target_pos = np.array(data["target_pos"])
    final_coords = [target_pos[0], target_pos[1], GRASP_HEIGHT]

    print(f"Moving to: {final_coords}")
    await execute_movement(final_coords)
    print("Done!")

asyncio.ensure_future(run())