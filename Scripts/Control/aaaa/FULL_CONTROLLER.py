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
POSITION_DATA = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results") # where the predicted coordinates are
END_POS_DIR = os.path.expanduser("~/Documents/PFG/Scripts/Control/RL_START") # where the results will be stored
SCRIPTS_PATH = os.path.expanduser("~/Documents/PFG/Scripts/") # directory where the franka controller script will be called from

if SCRIPTS_PATH not in sys.path:
    sys.path.append(SCRIPTS_PATH)
from Scripts.Control.aaaa.franka_controller import execute_movement # function which will used to send the franka to some given coordinates

FRANKA_1_PATH = "/World/Franka_Robot" # franka arm at the left
FRANKA_2_PATH = "/World/Franka_Robot_01" # right franka (2)

COLORS = ["red", "green", "blue"] # cube colors used for iteration

# JETBOT COORDS (at the start)
JETBOT_COORDS = np.array([1.28030509, -0.00182721,  0.06731872])

async def move_first_franka(target_color):
    # function to move the first franka
    cube_json_path = os.path.join(POSITION_DATA, f"detection_cube_{target_color.lower()}.json")
    with open(cube_json_path, "r") as f:
        data = json.load(f) 
        cube_position = np.array(data["world_pos"])

        # APPROACH TO THE TARGET
        safe_height = 0.15 
        await execute_movement([cube_position[0], cube_position[1], safe_height], keep_gripper_closed=False)
        await asyncio.sleep(1.0)
        print(f"##### APPROACHING VLM PREDICTED COORDS: {cube_position} #####")

        # DESCENT INTO TARGET
        contact_height = 0.025 
        await execute_movement([cube_position[0], cube_position[1], contact_height], keep_gripper_closed=False)
        await asyncio.sleep(1.0)
        print(f"##### DESCENDING TO GROUND #####")

        # CLOSING GRIPS
        print(f"##### CLOSING GRIPS ON {target_color.upper()} CUBE")
        await execute_movement([cube_position[0], cube_position[1], contact_height], keep_gripper_closed=True, max_steps=500)
        await asyncio.sleep(4.0) 

        # LIFT
        lift_height = 0.3
        print(f"##### LIFTING CUBE #####")
        await execute_movement([cube_position[0], cube_position[1], lift_height], keep_gripper_closed=True, max_steps=500)
        
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
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.45], keep_gripper_closed=True) # z=.25 offset to not collapse
    print(f"##### MOVING TO THE TOP OF THE JETBOT AT COORDINATES: {JETBOT_COORDS} (+z offset) ###")

    # Descend a little
    # await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.35], keep_gripper_closed=True) # lower the offset
    # print("##### DESCENDING A LITTLE #####")

    # Open the grips
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.45], keep_gripper_closed=False) # release the grips
    print("##### OPENING THE GRIPS! #####")

    # Go back to a higher spot to not interfere with the jetbot navigation
    await execute_movement([JETBOT_COORDS[0], JETBOT_COORDS[1], 0.95], keep_gripper_closed=False) # release the grips

async def move_second_franka(target_platform):
    # script to move second franka (pick cube from jetbot -> leave it in target platform)
    palett_json_path = os.path.join(POSITION_DATA, f"detection_pallet_{target_platform.lower()}.json")
    with open(palett_json_path, "r") as f:
        data = json.load(f) 
        palett_position = np.array(data["world_pos"])

    cube_hardcoded_pos = np.array([0,0,0])

    print("##### SEGUNDO BRAZO: Bajando al Jetbot #####")
    await execute_movement(cube_hardcoded_pos, 
                           keep_gripper_closed=False, 
                           robot_path=FRANKA_2_PATH) # IMPORTANTE: Pasar el path del 2º robot
    
    # 2. CERRAR GRIPS
    print("##### SEGUNDO BRAZO: Sujetando cubo #####")
    await execute_movement(cube_hardcoded_pos, 
                           keep_gripper_closed=True, 
                           robot_path=FRANKA_2_PATH)
    await asyncio.sleep(1.0)

    # 3. MOVER A LA PLATAFORMA (PLACE)
    print("##### SEGUNDO BRAZO: Moviendo a plataforma final #####")
    # Subir primero (Safe height)
    await execute_movement([palett_position[0], palett_position[1], 0.3], 
                           keep_gripper_closed=True, 
                           robot_path=FRANKA_2_PATH)
    # Bajar a la plataforma
    await execute_movement(palett_position, 
                           keep_gripper_closed=True, 
                           robot_path=FRANKA_2_PATH)
    
    # 4. SOLTAR
    print("##### SEGUNDO BRAZO: Cubo entregado #####")
    await execute_movement(palett_position, 
                           keep_gripper_closed=False, 
                           robot_path=FRANKA_2_PATH)


# Main function (kept for legacy compatibility)
async def run_main():
    print("##### RUNNING FULL PIPELINE! #####")
    print("##### GRABBING CUBE! #####")
    await move_first_franka("red")
    print("##### CUBE LIFTED, PLACING ON JETBOT! #####")
    await place_on_jetbot()
    print("##### PLACED ON JETBOT! #####")
    await move_second_franka("black")
    print("##### CUBE PICKED AND PLACED ON TOP OF TARGET PALETT #####")
    print("##### PIPELINE SUCCESSFULLY COMPLETED! #####")

# Individual step functions for new workflow
async def step_1_move_first_franka(target_color="red"):
    print("##### STEP 1: GRABBING CUBE! #####")
    await move_first_franka(target_color)
    print("##### CUBE LIFTED! #####")

async def step_2_place_on_jetbot():
    print("##### STEP 2: PLACING ON JETBOT! #####")
    await place_on_jetbot()
    print("##### PLACED ON JETBOT! #####")

async def step_4_move_second_franka(target_platform="black"):
    print("##### STEP 4: MOVING SECOND FRANKA #####")
    await move_second_franka(target_platform)
    print("##### CUBE PICKED AND PLACED ON TOP OF TARGET PALETT #####")

# asyncio.ensure_future(run_main())
    






