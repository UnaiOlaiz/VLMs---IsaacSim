# Main orchestrating function where the corresponding functions will be called in order to complete the whole movement
# Dependencies
import asyncio
import sys 
import os
# Import of the necessary function from other scripts:
CURRENT_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control")
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)
from PROGRESSION.FULL_CONTROLLER import run_main as lift_place_cube
from jetbot_controller import main as move_jetbot

async def main():
    print("##### STARTING FULL PIPELINE (LIFT+PLACE+NAVIGATION) STILL WITH CONTROLLER #####")
    await lift_place_cube()
    await asyncio.sleep(1.0)
    await move_jetbot()
    print("##### PIPELINE FINISHED! #####")

asyncio.ensure_future(main())