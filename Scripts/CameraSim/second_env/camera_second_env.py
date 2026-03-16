# File for second env platform detection + navigation.
# Uses per-platform crops so the VLM only sees one platform at a time.

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import omni.replicator.core as rep
import asyncio
from omni.isaac.core.utils.xforms import get_world_pose
from pxr import UsdGeom
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


URL = "http://127.0.0.1:8000/ground"
RESOLUTION = (1280, 720)

# Change this to switch targets: "green platform", "red platform", "blue platform"
INSTRUCTION = "red platform"

# Per-platform crops — each crop isolates one platform so the VLM
# can't confuse colors. Tune these values by checking Camera_02 view.
# Format: (y_start, x_start, y_end, x_end)
PLATFORM_CROPS = {
    "green platform": (100, 500, 280, 800),  # top-center
    "blue platform": (250, 100, 480, 450),  # left
    "red platform": (250, 800, 480, 1180),  # right
}

# All platforms are within 2.5m of origin
MAX_PLATFORM_DIST = 2.5


def get_prediction(instruction, rgb_image):
    """Sends cropped image + instruction to the VLM service."""
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"instruction": instruction, "image_b64": img_str}
    response = requests.post(URL, json=payload, timeout=10)
    return response.json()


def get_3d_target_calibrated(u, v, depth_map, cam_matrix):
    """Unprojects pixel (u,v) + straight-line depth into world XYZ."""
    distance = depth_map[v, u]
    if distance == 0 or np.isnan(distance) or np.isinf(distance):
        return np.array([0.0, 0.0, -1.0])

    f_pixel = (18.14756 * 1280) / 20.955
    cx, cy = 640, 360

    x_norm = (u - cx) / f_pixel
    y_norm = (v - cy) / f_pixel
    z_depth = distance / np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    point_cam = np.array([x_norm * z_depth, -y_norm * z_depth, -z_depth])
    R = cam_matrix[:3, :3]
    return R @ point_cam + cam_matrix[:3, 3]


async def main_vision():
    print("-" * 50 + "INITIALIZING RENDERER" + "-" * 50)

    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except Exception as e:
        print(f"Cleanup skipped: {e}")

    rp = rep.create.render_product("/World/Camera_02", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_02")
    if not camera_prim.IsValid():
        print("CAMERA NOT FOUND")
        return

    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    cam_matrix = np.array(world_transform).reshape(4, 4).T
    print(f"Camera position: {cam_matrix[:3, 3]}")

    # Get crop for this instruction
    if INSTRUCTION not in PLATFORM_CROPS:
        print(f"Unknown instruction '{INSTRUCTION}'. Using full image.")
        y_start, x_start, y_end, x_end = 0, 0, RESOLUTION[1], RESOLUTION[0]
    else:
        y_start, x_start, y_end, x_end = PLATFORM_CROPS[INSTRUCTION]

    print(f"Using crop: x={x_start}-{x_end}, y={y_start}-{y_end} for '{INSTRUCTION}'")

    loop = asyncio.get_event_loop()

    consecutive_detections = 0
    stability_count = 3
    last_stable_xyz = np.array([0.0, 0.0, 0.0])

    print(f"STARTING COORDINATE SEARCHING for: '{INSTRUCTION}'")

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            try:
                result = await loop.run_in_executor(
                    None, get_prediction, INSTRUCTION, cropped_img
                )

                if result and result.get("target") and result["target"].get("found"):
                    raw_bbox = result["target"]["bbox_xyxy"]
                    print(f"VLM bbox: {raw_bbox}")

                    # Map 0-1000 bbox → crop pixel coords → full image pixel coords
                    crop_h = y_end - y_start
                    crop_w = x_end - x_start
                    v_final = int(
                        np.clip(
                            (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000 + y_start,
                            0,
                            RESOLUTION[1] - 1,
                        )
                    )
                    u_final = int(
                        np.clip(
                            (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000 + x_start,
                            0,
                            RESOLUTION[0] - 1,
                        )
                    )

                    current_xyz = get_3d_target_calibrated(
                        u_final, v_final, depth_data, cam_matrix
                    )

                    print(f"Pixel u={u_final}, v={v_final}")
                    print(f"Depth: {depth_data[v_final, u_final]:.4f}m")
                    print(f"World XYZ: {current_xyz}")

                    # Filter 1: Z must be near floor
                    if current_xyz[2] < -0.05 or current_xyz[2] > 0.2:
                        print(f"Skipping bad Z={current_xyz[2]:.3f}")
                        continue

                    # Filter 2: must be within scene bounds
                    dist_from_origin = np.linalg.norm(current_xyz[:2])
                    if dist_from_origin > MAX_PLATFORM_DIST:
                        print(
                            f"Skipping hallucination — {dist_from_origin:.2f}m from origin"
                        )
                        consecutive_detections = 0
                        last_stable_xyz = np.array([0.0, 0.0, 0.0])
                        continue

                    distance = np.linalg.norm(current_xyz - last_stable_xyz)
                    if distance < 0.03:
                        consecutive_detections += 1
                        print(
                            f"Detections stable: {consecutive_detections}/{stability_count}"
                        )
                    else:
                        consecutive_detections = 1
                        last_stable_xyz = current_xyz
                        print(f"VLM jitter. Resetting at: {current_xyz}")

                    if consecutive_detections >= stability_count:
                        print(f"--- FINAL TARGET LOCKED: {last_stable_xyz} ---")
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

    print(f"Final target: {target_pos} → moving to {final_coords}")

    manager = await execute_movement(final_coords)

    if manager:
        state = manager.get_state()
        data_save = {
            "chassis_position": state["chassis_position"],
            "chassis_orientation": state["chassis_orientation"],
            "platform_world_pos": relative_platform_pos.tolist(),
        }

        json_folder = os.path.expanduser("~/Documents/PFG/Scripts/Control/")
        os.makedirs(json_folder, exist_ok=True)
        full_path = os.path.join(json_folder, "jetbot_state.json")

        with open(full_path, "w") as f:
            json.dump(data_save, f, indent=4)

        print(f"--- SUCCESS: State stored in {full_path} ---")


asyncio.ensure_future(run())
