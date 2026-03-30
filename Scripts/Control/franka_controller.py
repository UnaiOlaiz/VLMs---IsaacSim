# Script for the franka 'custom' controller (modifying RMPFlow), this will be called whenever it is needed to manually move the arm
# Dependencies
import numpy as np
import json
import asyncio
from omni.isaac.franka import Franka
import omni.timeline
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.franka.controllers import RMPFlowController

# Controller class
class FrankaControl:
    def __init__(self, prim_path="/World/Franka_Robot"):
        if not is_prim_path_valid(prim_path):
            print("##### ERROR GETTING THE FRANKA PATH, CHECK BOTH THE PATH AND NAME OF THE ROBOT IN THE STAGE #####")
            prim_path = "/Franka_Robot"

        # init of the robot
        self.robot = Franka(prim_path=prim_path)
        self.robot.initialize()

    # function to move the franka arm to a given position (x,y,z)
    def move_to_cube_top(self, target_pos, keep_gripper_closed=False):
        top_pos = np.array(target_pos)

        if not hasattr(self, "rmp_controller"):
            # use of RMPFlowController (already implemented)
            self.rmp_controller = RMPFlowController(
                name="target_hover", robot_articulation=self.robot
            )
        
        actions = self.rmp_controller.forward(
            target_end_effector_position=top_pos,
            target_end_effector_orientation=None,
        )
        self.robot.apply_action(actions)

        # logic to control the grippers
        if keep_gripper_closed:
            self.robot.gripper.close()
        else:
            self.robot.gripper.open()

        end_pos, _ = self.robot.end_effector.get_world_pose()
        distance = np.linalg.norm(end_pos - top_pos)
        print(f"##### DISTANCE TO DESTINATION: {distance:.4f}m ######", end="\r") # where to stop ("\r") to over-write the line
        return distance < 0.015 # variable

    # function to open the grippers
    def open_gripper(self):
        self.robot.gripper.open()

    # close it
    def close_gripper(self):
        self.robot.gripper.close()

# function to start the movement
async def execute_movement(final_coords, keep_gripper_closed=False, max_steps = 500):

    timeline = omni.timeline.get_timeline_interface()

    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(2.0)

    try:
        manager = FrankaControl()
        steps = 0 # I will add steps max limit so it does not get stuck
        stop = False 
        while not stop and steps < max_steps:
            await asyncio.sleep(0.01)
            stop = manager.move_to_cube_top(final_coords)
            steps += 1
            try:
                stop = manager.move_to_cube_top(
                    final_coords, keep_gripper_closed=keep_gripper_closed
                )
                if steps % 100 == 0:
                    print(f"##### STEPS: {steps/max_steps} #####", end="\r")
            except Exception as e:
                continue


        print("##### MOVEMENT COMPLETED ######")

    except Exception as e:
        print(f"Execution error: {e}")
