# Main orchestrating function where the corresponding functions will be called in order to complete the whole movement
# Dependencies
import asyncio
import sys 
import os
# Import of the necessary function from other scripts:
CURRENT_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control")
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)
from Scripts.Control.aaaa.FULL_CONTROLLER import (
    step_1_move_first_franka, 
    step_2_place_on_jetbot, 
    step_4_move_second_franka
)
from Scripts.Control.jetbot_controller import main as move_jetbot

# files of the position of the frankas
right_json = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_right_white.json")
left_json = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_left_white.json")

async def main():
    print("##### STARTING FULL PIPELINE (LIFT+PLACE+NAVIGATION) STILL WITH CONTROLLER #####")
    
    # Step 1: Move first franka to grab cube
    await step_1_move_first_franka("red")
    await asyncio.sleep(1.0)
    
    # Step 2: Place cube on jetbot
    await step_2_place_on_jetbot()
    await asyncio.sleep(1.0)
    
    # Step 3: Move jetbot
    print("##### STEP 3: MOVING JETBOT #####")
    await move_jetbot(right_json)
    await asyncio.sleep(1.0)
    
    # Step 4: Move second franka to pick from jetbot and place on platform
    # await step_4_move_second_franka("black")
    
    print("##### PIPELINE FINISHED! #####")

asyncio.ensure_future(main())