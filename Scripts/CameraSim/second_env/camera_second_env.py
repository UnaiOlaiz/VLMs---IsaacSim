# File for second env platform detection + navigation.
# Uses USD matrix with .T and +Z forward camera convention.

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
import sys
import os
import json
import importlib


scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"

if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    if "Control.jetbot_control" in sys.modules:
        importlib.reload(sys.modules["Control.jetbot_control"])
    from Control.jetbot_control import execute_movement

    print(f"Movement scripts correctly loaded from path: '{scripts_path}'!")
except ImportError as e:
    print(f"Error loading scripts from given path: {scripts_path}")
    raise e


# URL where the BentoML service is running
URL = "http://127.0.0.1:8000/ground"

# FIXED RESOLUTION OF THE ENVIRONMENT INSIDE ISAACSIM
RESOLUTION = (1280, 720)

# Target object instruction for the VLM
INSTRUCTION = "red platform"


def get_prediction(instruction, rgb_image):
    """
    Sends instruction + image to the BentoML VLM service and returns the result.
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
    distance = depth_map[v, u]

    if distance == 0 or np.isnan(distance) or np.isinf(distance):
        return np.array([0.0, 0.0, -1.0])

    f_pixel = (18.14756 * 1280) / 20.955
    cx, cy = 640, 360

    x_norm = (u - cx) / f_pixel
    y_norm = (v - cy) / f_pixel
    z_depth = distance / np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    # Verified: sx=+1, sy=-1, sz=-1 gives correct world position
    point_cam = np.array([x_norm * z_depth, -y_norm * z_depth, -z_depth])

    R = cam_matrix[:3, :3]
    point_world = R @ point_cam + cam_matrix[:3, 3]
    return point_world


async def main_vision():
    """
    Captures Camera_02 frames, queries the VLM, and returns the stable
    world-space XYZ of the detected target platform.
    """
    print("-" * 50 + "INITIALIZING RENDERER" + "-" * 50)

    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except Exception as e:
        print(f"Cleanup skipped (normal for first run): {e}")

    rp = rep.create.render_product("/World/Camera_02", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_02")
    if not camera_prim.IsValid():
        print("-" * 50 + "CAMERA NOT FOUND" + "-" * 50)
        return

    # Compute camera matrix ONCE before the loop
    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    cam_matrix = np.array(world_transform).reshape(4, 4).T
    print(f"Camera position: {cam_matrix[:3, 3]}")  # should be ~[0.270, -4.171, 3.945]

    loop = asyncio.get_event_loop()

    # Stability tracking
    consecutive_detections = 0
    stability_count = 3  # lock after 3 consecutive consistent detections
    last_stable_xyz = np.array([0.0, 0.0, 0.0])

    print("STARTING COORDINATE SEARCHING!")

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        y_start, y_end = 100, 280
        x_start, x_end = 550, 900
        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            try:
                result = await loop.run_in_executor(
                    None, get_prediction, INSTRUCTION, cropped_img
                )
                # Add this right after result is received:
                if result and result.get("target") and result["target"].get("found"):
                    raw_bbox = result["target"]["bbox_xyxy"]
                    print(f"VLM bbox: {result['target']['bbox_xyxy']}")

                    crop_h = y_end - y_start
                    crop_w = x_end - x_start
                    v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                    u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                    u_final = int(u_crop + x_start)
                    v_final = int(v_crop + y_start)

                    current_xyz = get_3d_target_calibrated(
                        u_final, v_final, depth_data, cam_matrix
                    )

                    print(f"Pixel u={u_final}, v={v_final}")
                    print(f"Raw depth at pixel: {depth_data[v_final, u_final]:.4f}m")
                    print(f"Computed world XYZ: {current_xyz}")
                    print(f"VLM bbox: {raw_bbox}")
                    print(f"Crop center pixel: u={u_final}, v={v_final}")
                    print(f"Crop region: x={x_start}-{x_end}, y={y_start}-{y_end}")

                    # Filter: platform is on the floor so Z must be near 0
                    if current_xyz[2] < -0.05 or current_xyz[2] > 0.15:
                        print(f"Skipping bad Z={current_xyz[2]:.3f}: {current_xyz}")
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

    robot_pos, _ = get_world_pose("/World/Jetbot")
    relative_platform_pos = target_pos - robot_pos

    grasp_height = 0.015
    final_coords = [target_pos[0], target_pos[1], grasp_height]

    print(f"Final target locked at: {target_pos}. Moving to: {final_coords}")
    print(f"Robot position saved: {robot_pos}, will be used for the RL task.")

    manager = await execute_movement(final_coords)

    # Get state from manager directly — no re-initialization needed
    if manager:
        state = manager.get_state()
        data_save = {
            "joint_positions": state["joint_positions"],
            "get_joint_velocities": state["joint_velocities"],
            "cube_world_pos": relative_platform_pos.tolist(),
        }

        json_folder = os.path.expanduser("~/Documents/PFG/Scripts/Control/")
        os.makedirs(json_folder, exist_ok=True)
        full_path = os.path.join(json_folder, "jetbot_state.json")

        with open(full_path, "w") as f:
            json.dump(data_save, f, indent=4)

        print(f"--- SUCCESS: Joint positions stored in {full_path} ---")


asyncio.ensure_future(run())
