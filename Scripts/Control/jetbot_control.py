import asyncio
import json
import math
import numpy as np
from isaacsim.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
import omni.timeline
import omni.usd
import omni.physx
from pxr import UsdPhysics


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class JetbotControl:
    def __init__(self, prim_path="/World/Jetbot"):
        if not is_prim_path_valid(prim_path):
            prim_path = "/Jetbot"

        self.prim_path = prim_path

        stage = omni.usd.get_context().get_stage()
        self.left_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{prim_path}/chassis/left_wheel_joint"), "angular")
        self.right_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{prim_path}/chassis/right_wheel_joint"), "angular")

        if not self.left_drive or not self.right_drive:
            raise RuntimeError("Drive API not found — run drive configuration first")

        self.chassis_xform = XFormPrim(f"{prim_path}/chassis")
        self.chassis_xform.initialize()

        self._pos    = np.array([0.0, 0.0, 0.03])
        self._orient = np.array([1.0, 0.0, 0.0, 0.0])

        self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )
        print("JetbotControl ready.")

    def _on_physics_step(self, dt):
        try:
            pos, ori = self.chassis_xform.get_world_poses()
            self._pos    = pos[0]
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

    def save_robot_state(self, target_pos):
        pos, orient = self.get_pose()
        state_data = {
            "chassis_position":    pos.tolist(),
            "chassis_orientation": orient.tolist(),
            "target_position":     np.array(target_pos).tolist()
        }
        with open("jetbot_state.json", "w") as f:
            json.dump(state_data, f)
        print("End state saved to 'jetbot_state.json'")


async def execute_movement(final_coords, 
                            turn_speed=300.0,    # deg/s for rotation phase
                            drive_speed=200.0,   # deg/s for forward phase
                            heading_tolerance=0.05,   # rad (~3°)
                            position_tolerance=0.10): # metres
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(2.0)

    manager = None
    try:
        manager = JetbotControl()
        gx, gy = final_coords[0], final_coords[1]

        await asyncio.sleep(0.1)

        print(f"Phase 1: Rotating to face ({gx:.3f}, {gy:.3f})")
        for step in range(3000):
            await asyncio.sleep(0.01)
            pos, orient = manager.get_pose()
            dx, dy = gx - pos[0], gy - pos[1]
            goal_heading  = math.atan2(dy, dx)
            yaw           = quat_wxyz_to_yaw(orient)
            heading_error = wrap_to_pi(goal_heading - yaw)

            print(f"  [Rotate] yaw={math.degrees(yaw):.1f}° "
                  f"goal={math.degrees(goal_heading):.1f}° "
                  f"err={math.degrees(heading_error):.1f}°")

            if abs(heading_error) < heading_tolerance:
                manager.stop()
                print("  Rotation complete.")
                break

            if heading_error > 0:
                manager.set_wheels(-turn_speed, turn_speed)
            else:
                manager.set_wheels(turn_speed, -turn_speed)

        manager.stop()
        await asyncio.sleep(0.2)  # brief pause between phases

        print(f"Phase 2: Driving forward to ({gx:.3f}, {gy:.3f})")
        for step in range(5000):
            await asyncio.sleep(0.01)
            pos, orient = manager.get_pose()
            dist = math.sqrt((gx - pos[0])**2 + (gy - pos[1])**2)

            print(f"  [Drive] pos=({pos[0]:.3f},{pos[1]:.3f}) dist={dist:.3f}m")

            if dist < position_tolerance:
                print("  Target reached!")
                break

            # Small heading correction while driving
            dx, dy = gx - pos[0], gy - pos[1]
            goal_heading  = math.atan2(dy, dx)
            yaw           = quat_wxyz_to_yaw(orient)
            heading_error = wrap_to_pi(goal_heading - yaw)

            # Differential correction: slight speed difference to stay on course
            correction = heading_error * 50.0  # scale to deg/s
            correction = float(np.clip(correction, -turn_speed * 0.5, turn_speed * 0.5))

            manager.set_wheels(drive_speed - correction, drive_speed + correction)

        manager.stop()
        manager.unsubscribe()

        if dist < position_tolerance:
            print("Movement complete! Jetbot is at the target.")
        else:
            print(f"Movement did NOT complete. Final dist={dist:.3f}m")

        manager.save_robot_state(final_coords)

    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()

    return manager