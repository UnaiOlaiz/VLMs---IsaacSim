# ##### CONTROLLED BY THE CONTROLLER #####

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
FRANKA_PATH_1 = "/World/Franka_Robot" # franka arm at the left
FRANKA_PATH_2 = "/World/Franka_Robot_01" # franka on the right

COLORS = ["red", "green", "blue"] # cube colors used for iteration

# JETBOT COORDS
JETBOT_COORDS = np.array([1.28030509, -0.00182721,  0.06731872])

# hardcoded coordinates of the red cube post movement:
# /World/Cubes/Red_Cube: [ 2.88283682 -0.05328616  0.28445387]
CUBE_POST_MOVEMENT_COORDS = np.array([ 2.88283682, -0.05328616,  0.28445387])


async def move_first_franka(target_color):
    json_path = os.path.join(CUBE_POS_DATA, f"detection_cube_{target_color.lower()}.json")
    with open(json_path, "r") as f:
        data = json.load(f) 
        cube_position = np.array(data["world_pos"])

        # APPROACH TO THE TARGET
        safe_height = 0.15 
        await execute_movement([cube_position[0], cube_position[1], safe_height], keep_gripper_closed=False, franka_path=FRANKA_PATH_1)
        # await asyncio.sleep(1.0)
        print(f"##### APPROACHING VLM PREDICTED COORDS: {cube_position} #####")

        # DESCENT INTO TARGET
        contact_height = 0.025 
        await execute_movement([cube_position[0], cube_position[1], contact_height], keep_gripper_closed=False, franka_path=FRANKA_PATH_1)
        # await asyncio.sleep(1.0)
        print(f"##### DESCENDING TO GROUND #####")

        # CLOSING GRIPS
        print(f"##### CLOSING GRIPS ON {target_color.upper()} CUBE")
        await execute_movement([cube_position[0], cube_position[1], contact_height], keep_gripper_closed=True, max_steps=100, franka_path=FRANKA_PATH_1)
        # await asyncio.sleep(2.0) 

        # LIFT
        lift_height = 0.3
        await execute_movement([cube_position[0], cube_position[1], lift_height], keep_gripper_closed=True, franka_path=FRANKA_PATH_1)
        
        print("##### LIFT + PLACE BY CONTROLLER COMPLETED #####")

"""
async def grab_color_cube():
    await move_franka_to_cube("red")
    '''
    for color in COLORS:
        await move_franka_to_cube(color)
    '''
"""


# Function to place the lifted cube on top of the jetbot (still hardcoded + controller)
async def place_on_jetbot():
    # Place the arm on top of the jetbot
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.45], keep_gripper_closed=True, franka_path=FRANKA_PATH_1) # z=.25 offset to not collapse
    print(f"##### MOVING TO THE TOP OF THE JETBOT AT COORDINATES: {JETBOT_COORDS} (+z offset) ###")

    # Descend a little
    # await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.35], keep_gripper_closed=True) # lower the offset
    # print("##### DESCENDING A LITTLE #####")

    # Open the grips
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.45], keep_gripper_closed=False, franka_path=FRANKA_PATH_1) # release the grips
    print("##### OPENING THE GRIPS! #####")

    # Go back to a higher spot to not interfere with the jetbot navigation
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.95], keep_gripper_closed=False, franka_path=FRANKA_PATH_1) # release the grips

# function to pick the cube from the top of the jetbot (after first move)
async def pick_from_jetbot():
    await execute_movement([CUBE_POST_MOVEMENT_COORDS[0], CUBE_POST_MOVEMENT_COORDS[1], .45], keep_gripper_closed=False, franka_path=FRANKA_PATH_2) # a little higher than the actual z
    
    # descend to the actual coordinates
    await execute_movement(CUBE_POST_MOVEMENT_COORDS, keep_gripper_closed=False, franka_path=FRANKA_PATH_2) # a little higher than the actual z

    # then close the grips
    await execute_movement(CUBE_POST_MOVEMENT_COORDS, keep_gripper_closed=True, max_steps=100, franka_path=FRANKA_PATH_2)

    # MOVE TO SECURE PLACE
    await execute_movement([CUBE_POST_MOVEMENT_COORDS[0], CUBE_POST_MOVEMENT_COORDS[1], .65], keep_gripper_closed=True, max_steps=100, franka_path=FRANKA_PATH_2)

# function to place the lifted cube from jetbot -> to target platform
async def place_on_palett(target_palett):
    json_path = os.path.join(CUBE_POS_DATA, f"detection_pallet_{target_palett.lower()}.json")
    with open(json_path, "r") as f:
        data = json.load(f) 
        platform_position = np.array(data["world_pos"])

        # APPROACH TO THE TARGET
        safe_height = 0.15 
        await execute_movement([platform_position[0], platform_position[1], safe_height], keep_gripper_closed=True, franka_path=FRANKA_PATH_2)
        # await asyncio.sleep(1.0)
        print(f"##### APPROACHING VLM PREDICTED COORDS: {platform_position} #####")

        # OPENING GRIPS
        print(f"##### OPENING GRIPS ON {target_palett.upper()} PALETT #####")
        await execute_movement([platform_position[0], platform_position[1], safe_height], keep_gripper_closed=False, max_steps=100, franka_path=FRANKA_PATH_2)
        # await asyncio.sleep(2.0) 

        # LIFT
        lift_height = 0.3
        await execute_movement([platform_position[0], platform_position[1], lift_height], keep_gripper_closed=False, franka_path=FRANKA_PATH_2)
        
        print("##### AAAAA #####")



'''
# Main function 
async def run_main():
    print("##### RUNNING FULL PIPELINE! #####")
    print("##### GRABBING CUBE! #####")
    await move_franka_to_cube("red")
    print("##### CUBE LIFTED, PLACING ON JETBOT! #####")
    await place_on_jetbot()
    print("##### PLACED ON JETBOT! #####")
    print("##### PIPELINE SUCCESSFULLY COMPLETED! #####")

# asyncio.ensure_future(run_main())
'''





