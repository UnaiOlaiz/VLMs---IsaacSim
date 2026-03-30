# Script to move the franka arm (the first one on the left) to the closest coordinates (predicted by the VLM) to save the end position -> json (for RL start)
# this script will just be used to get close to the cubes
# Dependencies
import asyncio
import json
import os 
import sys 
import numpy as np
import omni 
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.xforms import get_world_pose

# Paths to load the data/paths
CUBE_POS_DATA = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results") # where the predicted coordinates are
END_POS_DIR = os.path.expanduser("~/Documents/PFG/Scripts/Control/RL_START") # where the results will be stored
SCRIPTS_PATH = os.path.expanduser("~/Documents/PFG/Scripts/") # directory where the franka controller script will be called from
if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)
from Control.franka_controller import execute_movement # function which will used to send the franka to some given coordinates
FRANKA_PATH = "/World/Franka_Robot" # franka arm at the left
COLORS = ["red", "green", "blue"] # cube colors

async def move_franka_to_cube(target_color):
    """
    this function will read the vlm coordinates stored in the corresponding json file,
    move franka there stopped at a given height
    and save the end robot's joints' positions onto a new json file (for RL starting purposes)
    """
    json_path = os.path.join(CUBE_POS_DATA, f"detection_cube_{target_color.lower()}.json")
    with open(json_path, "r") as f:
        data = json.load(f) 
        cube_position = np.array(data["world_pos"]) # (x,y,z)
        print(f"##### VLM TARGETED {target_color.upper()} CUBE AT POSITION: {cube_position}")

        # pre-defined height to stop the franka arm movement
        grasp_height = .065 # variable depending on the needs
        target_pos = [cube_position[0], cube_position[1], grasp_height]

        await execute_movement(target_pos) # send franka there
        await asyncio.sleep(2.0) # so the physics engine relaxes a bit

        # instantiate the franka arm
        franka_articulations = Articulation(FRANKA_PATH)
        franka_articulations.initialize()
        joint_positions = franka_articulations.get_joint_positions() # important as they will be saved later on

        # save the json with the saved data
        out_path = os.path.join(END_POS_DIR, f"rl_ready_{target_color}_cube_franka.json")
        rl_data_save = {
            "target_color": target_color.lower(),
            "cube_world_pos": cube_position.tolist(),
            "joint_positions": joint_positions.tolist()
        }
        with open(out_path, "w") as f:
            json.dump(rl_data_save, f, indent=4)

        print(f"##### DATA FOR CUBE {target_color.upper()} SAVED FOR RL IN PATH: {out_path} #####")

async def run_all_colors():
    for color in COLORS:
        await move_franka_to_cube(color)

asyncio.ensure_future(run_all_colors())


