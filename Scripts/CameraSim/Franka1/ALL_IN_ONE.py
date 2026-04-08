# All-in-one script: captures camera, calls VLM, moves Franka_1
# Run inside Isaac Sim Script Editor WITH BentoML running on GPU

import numpy as np
import omni.replicator.core as rep
import omni.usd
import asyncio
from pxr import UsdGeom
from omni.physx.scripts import utils
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import VisualSphere
from PIL import Image
import requests
import base64
import os
import json
import sys
from io import BytesIO

# --- Config ---
scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from Scripts.Control.aaaa.franka_controller import execute_movement

RESOLUTION = (1280, 720)
URL = "http://127.0.0.1:8000/ground"
INSTRUCTION = "red cube"  # change to "green cube" or "blue cube" as needed
GRASP_HEIGHT = 0.055
ROBOT_PATH = "/World/Franka_Robot"
JSON_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_first_franka.json")

# Camera constants — verified for Camera_01
F_PIXEL = (18.14756 * 1280) / 20.955
CX, CY = 640, 360

# Calibration offset — computed from known cube positions vs VLM output
CALIB_OFFSET = np.array([-0.99, 0.20, 0.0])


# --- Collision setup ---
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if prim.IsA(UsdGeom.Mesh) and ROBOT_PATH in path_str:
        if "finger" in path_str.lower():
            continue
        utils.setCollider(prim, approximationShape="convexHull")


# --- Helper functions ---
def encode_image(rgb_array):
    """Encode full RGB frame as base64 PNG for VLM."""
    rgb_clean = np.ascontiguousarray(rgb_array[..., :3], dtype=np.uint8)
    # Save debug image
    Image.fromarray(rgb_clean).save(
        os.path.expanduser("~/Documents/PFG/Scripts/Control/camera_data/debug_crop.png")
    )
    img = Image.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_vlm(image_b64, timeout=30):
    """Call BentoML VLM service."""
    payload = {"instruction": INSTRUCTION, "image_b64": image_b64}
    response = requests.post(URL, json=payload, timeout=timeout)
    return response.json()


def unproject_to_world(u, v, depth_map, cam_matrix):
    """Convert pixel + depth to world XYZ."""
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return None
    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth
    local = np.array([x_cam, y_cam, z_cam, 1.0])
    world = np.dot(cam_matrix, local)
    return world[:3] + CALIB_OFFSET


async def main():
    print("=" * 60)
    print("PHASE 1: Initializing camera")
    print("=" * 60)

    try:
        rep.orchestrator.stop()
    except:
        pass

    rp = rep.create.render_product("/World/Camera_01", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_01")
    if not camera_prim.IsValid():
        print("ERROR: Camera_01 not found!")
        return

    # Stabilize camera
    print("Waiting for camera to stabilize...")
    for _ in range(5):
        await rep.orchestrator.step_async()

    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    cam_matrix = np.array(world_transform).reshape(4, 4).T

    robot_pos, _ = get_world_pose(ROBOT_PATH)
    print(f"Robot position: {robot_pos}")

    print("=" * 60)
    print(f"PHASE 2: VLM detection — looking for '{INSTRUCTION}'")
    print("=" * 60)

    loop = asyncio.get_event_loop()

    consecutive_detections = 0
    stability_count = 3
    last_stable_xyz = np.array([0.0, 0.0, 0.0])
    target_pos = None

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        image_b64 = encode_image(rgb_data)

        try:
            result = await loop.run_in_executor(None, call_vlm, image_b64)
            print(f"VLM result: {result}")

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]

                # Map normalized bbox to full image pixel coords
                v_norm = (raw_bbox[0] + raw_bbox[2]) / 2 / 1000
                u_norm = (raw_bbox[1] + raw_bbox[3]) / 2 / 1000
                u_final = int(np.clip(u_norm * RESOLUTION[0], 0, RESOLUTION[0] - 1))
                v_final = int(np.clip(v_norm * RESOLUTION[1], 0, RESOLUTION[1] - 1))

                print(f"Pixel u={u_final}, v={v_final}, depth={depth_data[v_final, u_final]:.4f}m")

                xyz = unproject_to_world(u_final, v_final, depth_data, cam_matrix)
                if xyz is None:
                    print("Invalid depth, skipping")
                    continue

                print(f"World XYZ (calibrated): {xyz}")

                # Pure geometric filter — reject physically impossible detections
                dist_from_franka = np.linalg.norm(xyz[:2] - robot_pos[:2])
                if xyz[2] < -0.1 or xyz[2] > 0.8 or dist_from_franka > 1.5:
                    print(f"Rejected by filter: z={xyz[2]:.3f}, dist={dist_from_franka:.3f}")
                    continue

                # Stability check
                dist = np.linalg.norm(xyz - last_stable_xyz)
                if dist < 0.05:
                    consecutive_detections += 1
                    print(f"Stable: {consecutive_detections}/{stability_count}")
                else:
                    consecutive_detections = 1
                    last_stable_xyz = xyz
                    print(f"Jitter reset at: {xyz}")

                if consecutive_detections >= stability_count:
                    print(f"--- TARGET LOCKED: {last_stable_xyz} ---")
                    target_pos = last_stable_xyz

                    VisualSphere(
                        prim_path="/World/vlm_target",
                        name="vlm_target",
                        position=target_pos,
                        radius=0.02,
                        color=np.array([0, 1, 0]),
                    )
                    break

            else:
                print("VLM found nothing")

        except Exception as e:
            print(f"VLM error: {e}")

        await asyncio.sleep(0.1)

    # Save coordinates to JSON
    relative_cube_pos = target_pos - robot_pos
    data_save = {
        "target_pos": target_pos.tolist(),
        "robot_pos": robot_pos.tolist(),
        "cube_world_pos": relative_cube_pos.tolist(),
    }
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data_save, f, indent=4)
    print(f"Coordinates saved to {JSON_PATH}")

    print("=" * 60)
    print("PHASE 3: Moving arm to pre-grasp position")
    print("=" * 60)

    final_coords = [target_pos[0], target_pos[1], GRASP_HEIGHT]
    print(f"Moving to: {final_coords}")
    await execute_movement(final_coords)
    print("Arm at pre-grasp position! Ready for RL task!")

    # Save joint state for RL initialization
    franka = Articulation(ROBOT_PATH)
    franka.initialize()
    joint_pos = franka.get_joint_positions().tolist()
    joint_vel = franka.get_joint_velocities().tolist()
    data_save["joint_positions"] = joint_pos
    data_save["joint_velocities"] = joint_vel
    with open(JSON_PATH, "w") as f:
        json.dump(data_save, f, indent=4)

    print(f"--- SUCCESS: Full state stored at {JSON_PATH} ---")


asyncio.ensure_future(main())