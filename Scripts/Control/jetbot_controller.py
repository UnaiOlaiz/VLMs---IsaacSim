# Jetbot script controller, we will use the franka positions obtained by the vlm prediction to travel between them
# Dependencies
import asyncio
import json
import math
import numpy as np
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
import omni.timeline
import omni.usd
import omni.physx
from pxr import UsdPhysics
import os

# path where the results obtained are stored.
FRANKA_COORDINATES = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results")

# helper functions (pure math)
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# controller class for the jetbot
class JetbotControl:
    def __init__(self, prim_path="/World/Jetbot"):
        if not is_prim_path_valid(prim_path):
            prim_path = "/Jetbot"

        self.prim_path = prim_path

        stage = omni.usd.get_context().get_stage()
        self.left_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{prim_path}/chassis/left_wheel_joint"), "angular"
        )
        self.right_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{prim_path}/chassis/right_wheel_joint"), "angular"
        )

        if not self.left_drive or not self.right_drive:
            raise RuntimeError("Drive API not found — run drive configuration first")

        self.chassis_xform = XFormPrim(f"{prim_path}/chassis")
        self.chassis_xform.initialize()

        self._pos = np.array([0.0, 0.0, 0.03])
        self._orient = np.array([1.0, 0.0, 0.0, 0.0])

        self._physics_sub = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step
            )
        )
        print("JetbotControl ready.")

    def _on_physics_step(self, dt):
        try:
            pos, ori = self.chassis_xform.get_world_poses()
            self._pos = pos[0]
            self._orient = ori[0]
        except Exception:
            pass

    def get_pose(self):
        return self._pos.copy(), self._orient.copy()

    def set_wheels(self, left_deg_s, right_deg_s):
        self.left_drive.GetTargetVelocityAttr().Set(left_deg_s)
        self.right_drive.GetTargetVelocityAttr().Set(right_deg_s)

    def stop(self):
        self.set_wheels(0.0, 0.0)

    def unsubscribe(self):
        self._physics_sub = None

    def get_state(self):
        pos, orient = self.get_pose()
        return {
            "chassis_position": pos.tolist(),
            "chassis_orientation": orient.tolist(),
        }

    def save_robot_state(self, target_pos):
        state = self.get_state()
        state_data = {
            "chassis_position": state["chassis_position"],
            "chassis_orientation": state["chassis_orientation"],
            "target_position": np.array(target_pos).tolist(),
        }
        with open("jetbot_state.json", "w") as f:
            json.dump(state_data, f)
        print("End state saved to 'jetbot_state.json'")


async def execute_movement(json_filename: str, turn_speed=300.0, drive_speed=300.0, heading_tolerance=0.05, position_tolerance=0.3): # the parameters can be tweaked
    # Read the json coordinates
    json_path = os.path.join(FRANKA_COORDINATES, json_filename) # the filename will be different for left/right
    if not json_path or not os.path.exists(json_path):
        print(f"##### Cannot find the JSON file, please check the given path: {json_path} #####")
    with open(json_path, "r") as f:
        data = json.load(f)
        gx, gy, _ = data["world_pos"] # go x, go y (z does not matter)
        side = data.get("side", "unknown")
        print(f"##### Loaded objective: Side {side} | in coordinates {gx:.3f}, {gy:.3f} #####")
    
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(1.0)

    manager = None
    try:
        manager = JetbotControl()
        await asyncio.sleep(0.1)

        # rotate to goal destination
        print(f"##### PHASE 1: Rotating to destination #####")
        for step in range(3000):
            await asyncio.sleep(0.01)
            pos, orient = manager.get_pose()
            dx, dy = gx - pos[0], gy - pos[1]
            goal_heading = math.atan2(dy, dx)
            yaw = quat_wxyz_to_yaw(orient)
            heading_error = wrap_to_pi(goal_heading - yaw)

            if abs(heading_error) < heading_tolerance:
                manager.stop()
                print("##### ROTATION COMPLETED! #####")
                break

            v_turn = turn_speed if heading_error > 0 else -turn_speed
            manager.set_wheels(-v_turn, v_turn)

        manager.stop()
        await asyncio.sleep(0.2)

        # travel to destination
        print(f"##### PHASE 2: Driving forward to destination #####")
        for step in range(5000):
            await asyncio.sleep(0.01)
            pos, orient = manager.get_pose()
            dist = math.sqrt((gx - pos[0]) ** 2 + (gy - pos[1]) ** 2) # euclidean distance

            # print(f"[Drive] pos=({pos[0]:.3f},{pos[1]:.3f}) dist={dist:.3f}m")

            if dist < position_tolerance:
                print(f"##### DESTINATION REACHED: Final distance: {dist:.3f}m #####")
                break

            # orientation correction while going forward
            goal_heading = math.atan2(gy - pos[1], gx - pos[0])
            yaw = quat_wxyz_to_yaw(orient)
            heading_error = wrap_to_pi(goal_heading - yaw)

            if abs(heading_error) > 0.8: # so it does not deviate much
                v_reorient = 150.0 if heading_error > 0 else -150.0
                manager.set_wheels(-v_reorient, v_reorient)
                continue

            if abs(heading_error) < .035:
                correction = 0
            else: 
                correction = np.clip(heading_error * 12.0, -60, 60)
            
            v_base = drive_speed * (1.0 - math.exp(-4.0 * dist))
            manager.set_wheels(v_base - correction, v_base + correction)

        manager.stop()
        manager.unsubscribe()


    except Exception as e:
        print(f"Execution error: {e}")
        import traceback

        traceback.print_exc()

    return manager

# Main function to run inside sim
async def main():
    print("##### STARTING NAVIGATION #####")
    
    # files
    right_json = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_right_white.json")
    left_json = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results/detection_robot_left_white.json")

    # first go to right robot, then left
    print("##### HEADING TO RIGHT FRANKA ROBOT #####")
    await execute_movement(right_json)

    # when it gets there, wait for a little to calm down the physics engine
    await asyncio.sleep(3.0)

    print("##### HEADING TO LEFT FRANKA ROBOT #####")
    await execute_movement(left_json)

    print("##### MOVEMENT COMPLETED, LET'S GO!!! #####")

# asyncio.ensure_future(main())