# Main vision script for Franka_1 cube detection and pre-grasp positioning

# Dependencies needed
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import omni.replicator.core as rep
import asyncio
from omni.isaac.core.utils.xforms import get_world_pose
from pxr import UsdGeom
from omni.isaac.core.objects import VisualSphere
from omni.physx.scripts import utils
import sys
import os
import json
from omni.isaac.core.articulations import Articulation
import subprocess


scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"

if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Control.franka_stop import execute_movement
    print(f"Movement scripts correctly loaded from path: '{scripts_path}'!")
except ImportError as e:
    print(f"Error loading scripts from given path: {scripts_path}")
    raise e


# Code in order to avoid franka collisions
# Skip fingers — they are already set to convexDecomposition
stage = omni.usd.get_context().get_stage()
robot_path = "/World/Franka_1"

for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if prim.IsA(UsdGeom.Mesh) and robot_path in path_str:
        if "finger" in path_str.lower():  # keep convex decomposition for the fingers
            continue
        utils.setCollider(prim, approximationShape="convexHull")

# URL where the BentoML service is running
URL = "http://127.0.0.1:8000/ground"

# FIXED RESOLUTION OF THE ENVIRONMENT INSIDE ISAACSIM
RESOLUTION = (1280, 720)

# The instruction — change to switch target cube
INSTRUCTION = "red cube"


def get_prediction(instruction, rgb_image):
    """
    Function that will perform the prediction given the instruction+environment image pair.
    """
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)

    buffered = BytesIO()
    img.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "instruction": instruction,
        "image_b64": img_str,
    }

    response = requests.post(URL, json=payload, timeout=10)
    return response.json()


def get_3d_target_calibrated(u, v, depth_map, cam_matrix):
    """
    Calibrated unprojection using verified camera convention.
    """
    z_depth = depth_map[v, u]

    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return np.array([0.0, 0.0, -1.0])

    # f_pixel = (focal_length * image_width) / horizontal_aperture
    f_pixel = (18.14 * 1280) / 20.955  # 20.955 is the default Isaac horizontal aperture
    cx, cy = 640, 360

    x_cam = (u - cx) * z_depth / f_pixel
    y_cam = (v - cy) * z_depth / f_pixel
    z_cam = -z_depth

    target_pos_local = np.array([x_cam, y_cam, z_cam, 1.0])
    # Transform to world
    target_pos_world = np.dot(cam_matrix, target_pos_local)
    return target_pos_world[:3]


async def main_vision():
    """
    Captures Camera_01 frames, queries the VLM for the target cube,
    and returns stable world-space XYZ of the detected cube.
    """
    print("-" * 50 + "INITIALIZING RENDERER" + "-" * 50)

    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except Exception as e:
        print(f"Cleanup skipped (normal for first run): {e}")

    rp = rep.create.render_product("/World/Camera_01", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_01")
    if not camera_prim.IsValid():
        print("-" * 50 + "CAMERA NOT FOUND" + "-" * 50)
        return

    loop = asyncio.get_event_loop()

    # Parameters for coordinate finding
    consecutive_detections = 0
    stability_count = 3  # lock after 3 consecutive consistent detections
    last_stable_xyz = np.array([0.0, 0.0, 0.0])

    print(f"STARTING COORDINATE SEARCHING for: '{INSTRUCTION}'")

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        # Crop to cube workspace — left side of camera view, excluding robot arm
        y_start, y_end = 200, 500
        x_start, x_end = 0, 550
        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            try:
                result = await loop.run_in_executor(
                    None, get_prediction, INSTRUCTION, cropped_img
                )

                if result and result.get("target") and result["target"].get("found"):
                    raw_bbox = result["target"]["bbox_xyxy"]

                    # Map bbox from crop space → full image pixel coords
                    crop_h, crop_w = (y_end - y_start), (x_end - x_start)
                    v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                    u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                    u_final = int(np.clip(u_crop + x_start, 0, RESOLUTION[0] - 1))
                    v_final = int(np.clip(v_crop + y_start, 0, RESOLUTION[1] - 1))

                    world_transform = UsdGeom.Xformable(
                        camera_prim
                    ).ComputeLocalToWorldTransform(0)
                    cam_matrix = np.array(world_transform).reshape(4, 4).T
                    current_xyz = get_3d_target_calibrated(
                        u_final, v_final, depth_data, cam_matrix
                    )

                    # Simple filter — same as working script
                    # Reject if Z is below floor or too high, or X is out of scene
                    if current_xyz[2] < -0.1 or current_xyz[0] > 2.0:
                        print(f"Skipping hallucination: {current_xyz}")
                        continue

                    distance = np.linalg.norm(current_xyz - last_stable_xyz)
                    if distance < 0.05:
                        consecutive_detections += 1
                        print(
                            f"Detections stable: {consecutive_detections}/{stability_count}"
                        )
                    else:
                        consecutive_detections = 1
                        last_stable_xyz = current_xyz
                        print(
                            f"VLM jitter detected. Resetting stability at: {current_xyz}"
                        )

                    if consecutive_detections >= stability_count:
                        print(f"--- FINAL TARGET LOCKED: {last_stable_xyz} ---")

                        VisualSphere(
                            prim_path="/World/green_target",
                            name="green_target",
                            position=last_stable_xyz,
                            radius=0.02,
                            color=np.array([0, 1, 0]),
                        )

                        return {"world_xyz": last_stable_xyz, "raw_json": result}

            except Exception as e:
                print(f"Connection error: {e}")

        await asyncio.sleep(0.1)


async def run():
    target_data = await main_vision()
    target_pos = target_data["world_xyz"]

    # I will add this to not crash the GPU
    # Stop BentoML and free GPU memory before physics movement
    # Stop BentoML by killing the process on port 8000
    result = subprocess.run(["lsof", "-t", "-i", ":8000"], capture_output=True, text=True)
    pid = result.stdout.strip()
    if pid:
        subprocess.run(["kill", "-9", pid])
        print(f"Killed BentoML process {pid}")
    await asyncio.sleep(8.0)

    # Get robot position to compute relative cube position for RL
    robot_pos, _ = get_world_pose("/World/Franka_1")
    relative_cube_pos = target_pos - robot_pos

    # Offset to place end effector just above the cube
    grasp_height = 0.045  # THIS IS THE DISTANCE! VERY IMPORTANT
    final_coords = [target_pos[0], target_pos[1], grasp_height]

    print(f"Final target locked at: {target_pos}. Moving to position: {final_coords}")
    print(f"Robot position saved: {robot_pos}, will be used for the RL task.")
    await execute_movement(final_coords)
    print("Arm located at pre-grasp position! Ready for RL task!")

    # Capture Section — save joint state for RL initialization
    franka = Articulation("/World/Franka_1")
    franka.initialize()

    joint_pos = franka.get_joint_positions().tolist()
    joint_vel = franka.get_joint_velocities().tolist()

    data_save = {
        "joint_positions": joint_pos,
        "get_joint_velocities": joint_vel,
        "cube_world_pos": relative_cube_pos.tolist(),
    }

    json_folder = os.path.expanduser("~/Documents/PFG/Scripts/Control/")
    json_filename = "rl_first_franka.json"
    full_path = os.path.join(json_folder, json_filename)

    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    with open(full_path, "w") as f:
        json.dump(data_save, f, indent=4)

    print(f"--- SUCCESS: Joint positions stored in {full_path} ---")


asyncio.ensure_future(run())