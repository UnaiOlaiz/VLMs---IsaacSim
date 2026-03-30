# =============================================================================
#  VLM-GUIDED JETBOT NAVIGATION — Camera_02 → Franka_2 (CALIBRATED)
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import omni.replicator.core as rep
import omni.usd
import asyncio
import sys
import os

from pxr import UsdGeom
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.xforms import get_world_pose

# ── PATH SETUP ────────────────────────────────────────────────────────────────
scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Scripts.Control.jetbot_controller import execute_movement as jetbot_move

    print(f"Jetbot control loaded from: '{scripts_path}'")
except ImportError as e:
    print(f"Error loading jetbot control: {e}")
    raise e

# ── CONSTANTS & CALIBRATION ───────────────────────────────────────────────────
URL = "http://127.0.0.1:8000/ground"
RESOLUTION = (1280, 720)
CAMERA_PATH = "/World/Cameras/Camera_03"
JETBOT_PATH = "/World/Jetbot"

# Camera_02 intrinsics
F_PIXEL = (18.14756 * 1280) / 20.955
CX, CY = 640, 360

# Camera_02 calibration offset - Applied to world coordinates
CAM02_OFFSET = np.array([3.01, 0.88, 0.112])

# VLM instruction — detect the marker sphere near Franka_2
FRANKA2_INSTRUCTION = "magenta sphere"

# Stop this far before Franka_2 so Jetbot doesn't collide
STOP_DISTANCE = 0.6  # meters


# =============================================================================
#  CAMERA HELPERS
# =============================================================================


def setup_camera():
    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except:
        pass
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])
    return rgb_annot, depth_annot


def unproject(u, v, depth_map, cam_matrix):
    """Converts pixel to World XYZ with Calibration Offset."""
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return None

    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth

    local = np.array([x_cam, y_cam, z_cam, 1.0])
    world = np.dot(cam_matrix, local)

    # Apply the calibration offset to the final world position
    return world[:3] + CAM02_OFFSET


# =============================================================================
#  VLM HELPERS
# =============================================================================


def call_vlm(instruction, rgb_image, timeout=30):
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"instruction": instruction, "image_b64": img_str}
    response = requests.post(URL, json=payload, timeout=timeout)
    return response.json()


async def vlm_detect(
    rgb_annot, depth_annot, camera_prim, instruction, label, stability_count=3
):
    print(f"\n{'=' * 20} VLM: Finding '{label}' {'=' * 20}")
    loop = asyncio.get_event_loop()
    consecutive = 0
    last_xyz = np.zeros(3)

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is None or rgb_data.size == 0:
            continue

        try:
            result = await loop.run_in_executor(None, call_vlm, instruction, rgb_data)

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]

                # Convert normalized VLM coordinates back to pixels
                v_norm = (raw_bbox[0] + raw_bbox[2]) / 2 / 1000
                u_norm = (raw_bbox[1] + raw_bbox[3]) / 2 / 1000
                u_final = int(np.clip(u_norm * RESOLUTION[0], 0, RESOLUTION[0] - 1))
                v_final = int(np.clip(v_norm * RESOLUTION[1], 0, RESOLUTION[1] - 1))

                # Get Camera Transform
                world_transform = UsdGeom.Xformable(
                    camera_prim
                ).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_transform).reshape(4, 4).T

                # Unproject with OFFSET
                xyz = unproject(u_final, v_final, depth_data, cam_matrix)

                if xyz is None:
                    continue

                if np.linalg.norm(xyz - last_xyz) < 0.15:
                    consecutive += 1
                else:
                    consecutive = 1
                    last_xyz = xyz

                if consecutive >= stability_count:
                    print(f"--- {label.upper()} LOCKED AT (Calibrated): {last_xyz} ---")
                    # Spawn visual marker to verify calibration
                    VisualSphere(
                        prim_path=f"/World/detected_{label.replace(' ', '_')}",
                        name=f"detected_{label.replace(' ', '_')}",
                        position=last_xyz,
                        radius=0.2,
                        color=np.array([1, 0, 1]),  # Magenta
                    )
                    return last_xyz
            else:
                print(f"VLM: Searching for {label}...")

        except Exception as e:
            print(f"VLM error: {e}")

        await asyncio.sleep(0.1)


# =============================================================================
#  MAIN EXECUTION
# =============================================================================


async def run():
    print("\n" + "=" * 60)
    print("  JETBOT NAVIGATION: Camera_02 VLM (Calibrated) → Franka_2")
    print("=" * 60)

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid():
        print("ERROR: Camera_02 not found!")
        return

    rgb_annot, depth_annot = setup_camera()

    # --- PHASE 1: Target Marker ---
    marker_path = "/World/Franka2_Marker"
    if not stage.GetPrimAtPath(marker_path).IsValid():
        VisualSphere(
            prim_path=marker_path,
            name="franka2_marker",
            position=np.array([5.46, 0.1, 0.15]),
            radius=0.15,
            color=np.array([1, 0, 0]),  # Red
        )

    # Wait for visuals to initialize
    for _ in range(10):
        await rep.orchestrator.step_async()

    # --- PHASE 2: Detection ---
    franka2_xyz = await vlm_detect(
        rgb_annot, depth_annot, camera_prim, FRANKA2_INSTRUCTION, "franka2 marker"
    )

    # --- PHASE 3: Navigation Planning ---
    jetbot_pos, _ = get_world_pose(JETBOT_PATH)

    # Direction vector from Jetbot to detected/calibrated Franka_2
    direction = franka2_xyz[:2] - jetbot_pos[:2]
    dist_total = np.linalg.norm(direction)
    direction_norm = direction / (dist_total + 1e-6)

    # Calculate stop point STOP_DISTANCE meters before the base
    nav_target_xy = franka2_xyz[:2] - direction_norm * STOP_DISTANCE
    nav_target = [float(nav_target_xy[0]), float(nav_target_xy[1]), 0.0]

    print(f"Current Jetbot Pos: {jetbot_pos[:2]}")
    print(f"Navigating to: {nav_target}")

    # --- PHASE 4: Execute Movement ---
    await jetbot_move(nav_target)

    print("\n" + "=" * 60)
    print("  SUCCESS: Jetbot reached Franka_2 docking zone.")
    print("=" * 60)


asyncio.ensure_future(run())
