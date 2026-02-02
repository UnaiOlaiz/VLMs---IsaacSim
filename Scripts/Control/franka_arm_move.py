# Dependencies
import asyncio
import numpy as np
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api import World 
from isaacsim.robot.manipulators.examples.franka import Franka 
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController 
from isaacsim.core.api.objects import VisualSphere

# Coordinates, pre-fixed for now, the ones that the VLM returns later
coordinates_xyz = np.array([0.4, 0.0, 0.5])

async def move_franka():
    '''
    Asynchronous function that will be running to move the arm to the given coordinates
    '''
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
    
    await world.initialize_simulation_context_async()

    robot_path = "/World/Franka_Robot" # path to the robot module in the environment
    if world.scene.object_exists("franka_arm"):
        franka_arm = world.scene.get_object("franka_arm")
    else:
        franka_arm = world.scene.add(Franka(prim_path=robot_path, name="franka_arm"))

    # I will create a visual marker to visualize where the arm is trying to go, I will create it manually using this script
    if prim_utils.is_prim_path_valid("/World/target_marker"):
        prim_utils.delete_prim("/World/target_marker")

    VisualSphere(
        prim_path="/World/target_marker",
        name="target",
        position=coordinates_xyz,
        radius=.03,
        color=np.array([0,0,1]) # blue
    ) 

    # We "restart" the env and initialize the robot
    await world.reset_async()
    franka_arm.initialize()

    # Controller inizialization
    controller = RMPFlowController(
        name="franka_controller",
        robot_articulation=franka_arm
    )
    controller.reset()
    print(f"Moving franka arm to given coordinates: {coordinates_xyz}")

    try:
        for i in range(500):
            if not world.is_playing():
                world.play()

            actions = controller.forward(
                target_end_effector_position=coordinates_xyz,
                target_end_effector_orientation=np.array([0, 1, 0, 0]) 
            )
            
            if actions is not None:
                franka_arm.apply_action(actions)
            
            await world.step_async()

            # Distance check (euclidean)
            position, _ = franka_arm.end_effector.get_world_pose()
            distance = np.linalg.norm(position - coordinates_xyz)
            
            if distance < 0.01:
                print(f"Success! Final Distance: {distance:.4f}m")
                break 
    except Exception as e:
        print(f"An error occurred during movement: {e}")


asyncio.ensure_future(move_franka())