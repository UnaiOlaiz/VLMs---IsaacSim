# Main orchestrating function where the corresponding functions will be called in order to complete the whole movement
# Dependencies
import asyncio
import sys
import os

# Import of the necessary function from other scripts:
CURRENT_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control")
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)
from PROGRESSION.FULL_CONTROLLER import (
    move_first_franka,
    place_on_jetbot,
    pick_from_jetbot,
    place_on_palett,
)
from jetbot_controller import main as move_jetbot

# files
right_json = os.path.expanduser(
    "~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_right_white.json"
)
left_json = os.path.expanduser(
    "~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_left_white.json"
)


async def main():
    print(
        "##### STARTING FULL PIPELINE (LIFT+PLACE+NAVIGATION) STILL WITH CONTROLLER #####"
    )
    await move_first_franka("blue")
    await place_on_jetbot()
    await asyncio.sleep(1.0)
    await move_jetbot(right_json)
    await pick_from_jetbot()
    await place_on_palett("red")
    print("##### PIPELINE FINISHED! #####")


asyncio.ensure_future(main())

# hardcoded coordinates of the red cube post movement:
# /World/Cubes/Red_Cube: [ 2.88283682 -0.05328616  0.28445387]

