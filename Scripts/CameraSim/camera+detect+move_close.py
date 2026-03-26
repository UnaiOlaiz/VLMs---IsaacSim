# =============================================================================
#  VLM CUBE DETECTION — Multi-prompt median, configurable color
#  Robot stays in DEFAULT pose, only saves coordinates for RL
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
import omni
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
from omni.isaac.core.utils.prims import get_prim_at_path

scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Change this to detect different cubes: "red", "green", "blue"
CUBE_COLOR  = "blue"

URL_MULTI   = "http://127.0.0.1:8000/ground_multi"
RESOLUTION  = (1280, 720)
ROBOT_PATH  = "/World/Franka_Robot"
CAMERA_PATH = "/World/Cameras/Camera_01"

# Stability logic
STABILITY_COUNT     = 3
STABILITY_THRESHOLD = 0.05

# Camera intrinsics
F_PIXEL = (18.14 * 1280) / 20.955
CX, CY  = 640, 360

# Debug image save path
DEBUG_PATH = os.path.expanduser("~/Documents/PFG/Scripts/Control/camera_data/")

# ── COLLISION SETUP ───────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) and ROBOT_PATH in str(prim.GetPath()):
        try:
            utils.setCollider(prim, approximationShape="convexHull")
        except:
            pass


# =============================================================================
#  HELPERS
# =============================================================================

def call_multi(color, rgb_image, timeout=60):
    """Call ground_multi endpoint — sends FULL image."""
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = PILImage.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"color": color, "image_b64": img_str}
    response = requests.post(URL_MULTI, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def unproject(u, v, depth_map, cam_matrix):
    """Unproject pixel + depth to world XYZ."""
    u = int(np.clip(u, 0, RESOLUTION[0] - 1))
    v = int(np.clip(v, 0, RESOLUTION[1] - 1))
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return None
    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth
    local = np.array([x_cam, y_cam, z_cam, 1.0])
    world = np.dot(cam_matrix, local)
    return world[:3]


def is_valid(xyz):
    """Reject physically impossible detections."""
    if xyz is None or not np.all(np.isfinite(xyz)):
        return False
    if xyz[2] < -0.3:  # relaxed — floor level can give slightly negative Z
        return False
    if xyz[0] > 2.0:   # too far in X
        return False
    return True


def save_debug_image(rgb_data):
    """Save full image for inspection."""
    os.makedirs(DEBUG_PATH, exist_ok=True)
    PILImage.fromarray(
        np.ascontiguousarray(rgb_data[..., :3], dtype=np.uint8)
    ).save(os.path.join(DEBUG_PATH, "debug_full.png"))


def remove_marker():
    prim = get_prim_at_path("/World/detection_marker")
    if prim and prim.IsValid():
        try:
            omni.kit.commands.execute(
                "DeletePrims", paths=["/World/detection_marker"], destructive=False
            )
        except:
            pass


def spawn_marker(position, color_rgb):
    remove_marker()
    try:
        VisualSphere(
            prim_path="/World/detection_marker",
            name="detection_marker",
            position=np.array(position, dtype=np.float32),
            radius=0.025,
            color=np.array(color_rgb),
        )
    except Exception as e:
        print(f"Marker spawn failed: {e}")


# =============================================================================
#  MAIN VISION LOOP
# =============================================================================

async def main_vision():
    print("-" * 50 + " INITIALIZING RENDERER " + "-" * 50)

    try:
        rep.orchestrator.stop()
    except:
        pass

    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid():
        print(f"ERROR: Camera not found at {CAMERA_PATH}!")
        return None

    # Stabilize camera
    for _ in range(5):
        await rep.orchestrator.step_async()

    loop        = asyncio.get_event_loop()
    consecutive = 0
    last_xyz    = None
    debug_saved = False

    print(f"SEARCHING FOR: {CUBE_COLOR.upper()} CUBE (multi-prompt, full image)")

    while True:
        await rep.orchestrator.step_async()
        rgb_data   = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is None or depth_data is None:
            await asyncio.sleep(0.1)
            continue

        # Save debug image once
        if not debug_saved:
            save_debug_image(rgb_data)
            print(f"Debug image saved to {DEBUG_PATH}debug_full.png")
            debug_saved = True

        try:
            # Send FULL image to VLM
            result = await loop.run_in_executor(
                None, call_multi, CUBE_COLOR, rgb_data
            )

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox  = result["target"]["bbox_xyxy"]
                num_valid = result["target"].get("num_valid", 1)
                all_bboxes = result["target"].get("all_bboxes", [])

                print(f"Agreement: {num_valid}/3 | bbox: {raw_bbox} | all: {all_bboxes}")

                # Skip very low confidence
                if num_valid < 2:
                    print("Only 1/3 prompts agreed — skipping")
                    await asyncio.sleep(0.1)
                    continue

                # Map bbox to full image pixel coords
                v_norm  = (raw_bbox[0] + raw_bbox[2]) / 2.0 / 1000.0
                u_norm  = (raw_bbox[1] + raw_bbox[3]) / 2.0 / 1000.0
                u_final = int(u_norm * RESOLUTION[0])
                v_final = int(v_norm * RESOLUTION[1])

                world_transform = UsdGeom.Xformable(
                    camera_prim
                ).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_transform).reshape(4, 4).T

                xyz       = unproject(u_final, v_final, depth_data, cam_matrix)
                depth_val = depth_data[v_final, u_final]

                print(f"Pixel: ({u_final}, {v_final}) | Depth: {depth_val:.4f}m | XYZ: {xyz}")

                if not is_valid(xyz):
                    print(f"Rejected invalid: {xyz}")
                    await asyncio.sleep(0.1)
                    continue

                # Stability check
                if last_xyz is None:
                    consecutive = 1
                    last_xyz    = xyz
                    print(f"First valid detection: {xyz}")
                else:
                    dist = np.linalg.norm(xyz - last_xyz)
                    if dist < STABILITY_THRESHOLD:
                        consecutive += 1
                        print(f"Stable: {consecutive}/{STABILITY_COUNT}")
                    else:
                        consecutive = 1
                        last_xyz    = xyz
                        print(f"Jitter reset: {xyz}")

                if consecutive >= STABILITY_COUNT:
                    print(f"--- FINAL TARGET LOCKED: {last_xyz} ---")
                    marker_colors = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1]}
                    spawn_marker(last_xyz, marker_colors.get(CUBE_COLOR, [1,1,0]))
                    return {"world_xyz": last_xyz, "pixel_coords": [u_final, v_final]}

            else:
                print(f"No valid detection for '{CUBE_COLOR}' cube")

        except Exception as e:
            print(f"Detection error: {e}")

        await asyncio.sleep(0.1)


# =============================================================================
#  RUN
# =============================================================================

async def run():
    target_data = await main_vision()
    if target_data is None:
        print("Vision failed.")
        return

    target_pos = np.array(target_data["world_xyz"], dtype=np.float64)
    robot_pos, _ = get_world_pose(ROBOT_PATH)
    robot_pos  = np.array(robot_pos, dtype=np.float64)
    relative   = target_pos - robot_pos

    print(f"\nDetected {CUBE_COLOR} cube at:  {target_pos.tolist()}")
    print(f"Robot position:               {robot_pos.tolist()}")
    print(f"Relative cube pos (for RL):   {relative.tolist()}")

    # Save joint state in default pose
    franka    = Articulation(ROBOT_PATH)
    franka.initialize()
    joint_pos = franka.get_joint_positions().tolist()
    joint_vel = franka.get_joint_velocities().tolist()

    data_save = {
        "joint_positions":  joint_pos,
        "joint_velocities": joint_vel,
        "cube_world_pos":   relative.tolist(),
        "cube_world_abs":   target_pos.tolist(),
        "robot_world_pos":  robot_pos.tolist(),
        "cube_color":       CUBE_COLOR,
        "pixel_coords":     target_data.get("pixel_coords"),
        "reset_type":       "default_pose",
    }

    out_dir  = os.path.expanduser("~/Documents/PFG/Scripts/Control/")
    out_path = os.path.join(out_dir, "rl_start_default_pose.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data_save, f, indent=4)

    print(f"\n--- SUCCESS: Saved to {out_path} ---")


asyncio.ensure_future(run())