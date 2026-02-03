import asyncio
import numpy as np
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, clear_stage, get_current_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdPhysics, Sdf, Gf
from isaacsim.robot.wheeled_robots.robots import WheeledRobot

if World.instance():
    World.instance().clear_instance()
clear_stage()

world = World()
app = omni.kit.app.get_app()

async def run_simulation():
    stage = get_current_stage()
    
    scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(scene_path):
        UsdPhysics.Scene.Define(stage, Sdf.Path(scene_path))

    world.scene.add_default_ground_plane()
    
    assets_root = get_assets_root_path()
    asset_path = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    print("Loading Jetbot USD...")
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Jetbot")
    
    for i in range(30):
        await app.next_update_async()
    
    robot_path = "/World/Jetbot/chassis"
    jetbot = Robot(prim_path=robot_path, name="my_jetbot")
    world.scene.add(jetbot)
    print("Robot added to scene.")

    world.play()
    print("Simulation started.")
    
    jetbot.initialize()
    
    for i in range(10):
        await app.next_update_async()

    controller = jetbot.get_articulation_controller()
    
    print("Starting movement loop...")
    for i in range(500):
        action = ArticulationAction(joint_velocities=np.array([2.0, 10.0]))
        
        if controller:
            controller.apply_action(action)
        
        await app.next_update_async()
        if i % 100 == 0:
            print(f"Driving... Iteration {i}")

    print("Done!")

asyncio.ensure_future(run_simulation())