import numpy as np
import json
import asyncio
from omni.isaac.franka import Franka
from omni.isaac.core.world import World
import omni.timeline
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.franka.controllers import RMPFlowController

class FrankaControl:
    def __init__(self, prim_path="/World/Franka_Robot", name="franka_pfg"):
        if not is_prim_path_valid(prim_path):
            print("Error verifying prim path")
            prim_path = "/Franka_Robot"
            
        self.robot = Franka(prim_path=prim_path, name=name)
        self.robot.initialize() 

    def move_to_cube_top(self, target_pos):
        top_pos = np.array(target_pos)
        top_pos[2] = 0.20 
        
        if not hasattr(self, "rmp_controller"):
            self.rmp_controller = RMPFlowController(name="target_hover", robot_articulation=self.robot)

        actions = self.rmp_controller.forward(
            target_end_effector_position=top_pos,
            target_end_effector_orientation=None
        )

        self.robot.apply_action(actions)
        self.robot.gripper.open()

        end_pos, _ = self.robot.end_effector.get_world_pose()
        distance = np.linalg.norm(end_pos - top_pos)

        print(f"Current Distance: {distance:.4f}m", end="\r")

        return distance < 0.0575 # established arbitrary distance record, it works! 

    def save_robot_state(self, cube_pos):
        '''
        Function to save end robot's joints' positions to then transfer it to an RL task environment
        '''
        state_data = {
                "joint_positions": self.robot.get_joint_positions().tolist(),
                "joint_velocities": self.robot.get_joint_velocities().tolist(),
                "cube_target": cube_pos.tolist()
        }   

        with open("rl_initial_state.json", "w") as f:
            json.dump(state_data, f)
        print("\nEnd state saved to 'rl_initial_state.json'")

async def execute_movement(final_coords):
    timeline = omni.timeline.get_timeline_interface()
    
    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(2.0) 

    try:
        manager = FrankaControl()
        print(f"Franka ready! Moving to {final_coords}")
        
        done = False
        while not done:
            await asyncio.sleep(0.01)
            try:
                done = manager.move_to_cube_top(final_coords)
            except Exception as e:
                # Catch specific move errors if they occur
                continue
        print("\nMovement phase completed! The arm is located on the top of the target!")
        
        # save the end state
        manager.save_robot_state(np.array(final_coords))
        
    except Exception as e:
        print(f"Execution error: {e}")
