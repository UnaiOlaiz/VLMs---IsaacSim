# Script 1: Save camera data from Isaac Sim (NO BentoML needed)
# Run inside Isaac Sim Script Editor BEFORE starting BentoML

import numpy as np
import omni.replicator.core as rep
import omni.usd
import asyncio
from pxr import UsdGeom
from omni.physx.scripts import utils
from omni.isaac.core.utils.xforms import get_world_pose
from PIL import Image
import os
import json
import pickle

RESOLUTION = (1280, 720)
SAVE_DIR = os.path.expanduser("~/Documents/PFG/Scripts/Control/camera_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# Collision setup
stage = omni.usd.get_context().get_stage()
robot_path = "/World/Franka_1"
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if prim.IsA(UsdGeom.Mesh) and robot_path in path_str:
        if "finger" in path_str.lower():
            continue
        utils.setCollider(prim, approximationShape="convexHull")

async def save_data():
    print("--- INITIALIZING CAMERA ---")

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

    # Wait a few frames for the camera to stabilize
    print("Waiting for camera to stabilize...")
    for _ in range(5):
        await rep.orchestrator.step_async()

    rgb_data = rgb_annot.get_data()
    depth_data = depth_annot.get_data()

    # Get camera world transform
    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    cam_matrix = np.array(world_transform).reshape(4, 4).T

    # Get robot position
    robot_pos, _ = get_world_pose("/World/Franka_1")

    # Save full RGB image
    rgb_clean = np.ascontiguousarray(rgb_data[..., :3], dtype=np.uint8)
    full_img = Image.fromarray(rgb_clean)
    full_img.save(os.path.join(SAVE_DIR, "camera01_full.png"))

    # Save cropped image (cube workspace area)
    y_start, y_end = 200, 500
    x_start, x_end = 0, 550
    cropped = rgb_clean[y_start:y_end, x_start:x_end]
    crop_img = Image.fromarray(cropped)
    crop_img.save(os.path.join(SAVE_DIR, "camera01_crop.png"))

    # Save depth data as numpy array
    np.save(os.path.join(SAVE_DIR, "depth_data.npy"), depth_data)

    # Save metadata
    metadata = {
        "cam_matrix": cam_matrix.tolist(),
        "robot_pos": robot_pos.tolist(),
        "resolution": list(RESOLUTION),
        "crop": {
            "y_start": y_start, "y_end": y_end,
            "x_start": x_start, "x_end": x_end
        }
    }
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"--- DATA SAVED TO: {SAVE_DIR} ---")
    print(f"  Full image:  camera01_full.png")
    print(f"  Crop image:  camera01_crop.png  (send this to VLM)")
    print(f"  Depth data:  depth_data.npy")
    print(f"  Metadata:    metadata.json")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Start BentoML in terminal")
    print("  2. Run run_vlm_standalone.py in terminal (NOT Isaac Sim)")
    print("=" * 60)

asyncio.ensure_future(save_data())