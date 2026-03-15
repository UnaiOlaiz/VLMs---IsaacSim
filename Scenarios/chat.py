import asyncio
import math
import numpy as np

from pxr import UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.robots import WheeledRobot


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_wxyz_to_yaw(q):
    """
    Quaternion expected as [w, x, y, z].
    Returns planar yaw.
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def add_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/SunLight")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.53)


class GoToXYController(BaseController):
    """
    Custom closed-loop differential-drive controller.

    Input:
        current_position: np.array([x, y, z])
        current_orientation: np.array([w, x, y, z])
        goal_position: np.array([x_goal, y_goal])

    Output:
        ArticulationAction with wheel joint velocities
    """

    def __init__(
        self,
        name="go_to_xy_controller",
        wheel_radius=0.03,
        wheel_base=0.1125,
        kp_linear=1.2,
        kp_angular=3.0,
        max_linear=0.5,
        max_angular=2.0,
        position_tolerance=0.03,
    ):
        super().__init__(name=name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.position_tolerance = position_tolerance

    def forward(self, current_position, current_orientation, goal_position):
        x, y = current_position[0], current_position[1]
        gx, gy = goal_position[0], goal_position[1]

        dx = gx - x
        dy = gy - y
        distance = math.sqrt(dx * dx + dy * dy)

        yaw = quat_wxyz_to_yaw(current_orientation)
        goal_heading = math.atan2(dy, dx)
        heading_error = wrap_to_pi(goal_heading - yaw)

        # Stop if close enough
        if distance < self.position_tolerance:
            return ArticulationAction(joint_velocities=[0.0, 0.0])

        # Simple proportional control
        linear_cmd = self.kp_linear * distance
        angular_cmd = self.kp_angular * heading_error

        # If badly misaligned, prioritize turning over forward motion
        if abs(heading_error) > 0.35:
            linear_cmd *= 0.25

        linear_cmd = float(np.clip(linear_cmd, -self.max_linear, self.max_linear))
        angular_cmd = float(np.clip(angular_cmd, -self.max_angular, self.max_angular))

        # Differential drive inverse kinematics
        left_w = ((2.0 * linear_cmd) - (angular_cmd * self.wheel_base)) / (
            2.0 * self.wheel_radius
        )
        right_w = ((2.0 * linear_cmd) + (angular_cmd * self.wheel_base)) / (
            2.0 * self.wheel_radius
        )

        return ArticulationAction(joint_velocities=[left_w, right_w])


async def spawn_and_drive():
    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0)
    if world is None:
        print("Failed to create world")
        return

    await world.initialize_simulation_context_async()
    if world is None or world.stage is None:
        print("Failed to initialize simulation context")
        return

    world.scene.add_default_ground_plane()
    add_lights(world.stage)

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find Isaac Sim assets root path.")

    jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"

    robot = world.scene.add(
        WheeledRobot(
            prim_path="/World/CarrierBot",
            name="carrier_bot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=jetbot_usd,
            position=np.array([0.0, 0.0, 0.03]),
        )
    )

    # Payload box placed on top of the robot
    payload = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Payload",
            name="payload",
            position=np.array([0.0, 0.0, 0.16]),
            scale=np.array([0.10, 0.10, 0.08]),
            color=np.array([0.9, 0.2, 0.2]),
            mass=0.2,
        )
    )

    # Add three colored platforms
    red_platform = world.scene.add(
        DynamicCuboid(
            prim_path="/World/RedPlatform",
            name="red_platform",
            position=np.array([2.0, 0.0, 0.01]),
            scale=np.array([0.3, 0.3, 0.02]),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.0,
        )
    )

    green_platform = world.scene.add(
        DynamicCuboid(
            prim_path="/World/GreenPlatform",
            name="green_platform",
            position=np.array([0.0, 2.0, 0.01]),
            scale=np.array([0.3, 0.3, 0.02]),
            color=np.array([0.0, 1.0, 0.0]),
            mass=0.0,
        )
    )

    blue_platform = world.scene.add(
        DynamicCuboid(
            prim_path="/World/BluePlatform",
            name="blue_platform",
            position=np.array([-2.0, 0.0, 0.01]),
            scale=np.array([0.3, 0.3, 0.02]),
            color=np.array([0.0, 0.0, 1.0]),
            mass=0.0,
        )
    )

    await world.reset_async()
    await world.play_async()

    controller = GoToXYController()

    # List of waypoints in XY
    goals = [
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    ]

    current_goal_idx = 0
    max_steps = 3000

    for step in range(max_steps):
        pos, quat = robot.get_world_pose()
        goal_xy = goals[current_goal_idx]

        action = controller.forward(
            current_position=pos,
            current_orientation=quat,
            goal_position=goal_xy,
        )
        robot.apply_action(action)

        # Change waypoint when close
        dist = np.linalg.norm(pos[:2] - goal_xy)
        if dist < 0.05:
            current_goal_idx += 1
            if current_goal_idx >= len(goals):
                robot.apply_action(ArticulationAction(joint_velocities=[0.0, 0.0]))
                print("Finished all goals.")
                break
            else:
                print(
                    f"Reached goal {current_goal_idx}, switching to {goals[current_goal_idx]}"
                )

        if step % 60 == 0:
            print(f"step={step} pos={pos[:2]} goal={goal_xy} dist={dist:.3f}")

        step_result = world.step_async()
        if step_result is not None:
            await step_result
        else:
            print("World step returned None, breaking simulation loop")
            break

    print("Done.")


asyncio.create_task(spawn_and_drive())
