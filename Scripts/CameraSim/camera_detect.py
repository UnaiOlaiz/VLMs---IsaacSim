# Dependencies
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
import os
import json
import sys
import cv2 

TARGET_TYPE  = "pallet"   # cube / pallet
TARGET_COLOR = "black"  # red, green, blue, black
RESOLUTION   = (1280, 720) # screen resolution
ROBOT_PATH   = "/World/Franka_Robot" # franka name in environment
URL_MULTI = "http://127.0.0.1:8000/ground_multi" # endpoint of the function we will use inside our bentoml server

# Camera config: Camera_01 for cubes, Camera_02 for pallets
CAMERA_PATH = "/World/Cameras/Camera_02" if TARGET_TYPE == "pallet" else "/World/Cameras/Camera_01"

# We load the CV detection scripts
sys.path.append(os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV"))
try:
    from detect_cubes import detect as detect_cubes
    from detect_palettes import detect_palette as detect_palettes
except ImportError as e:
    print(f"Error loading the CV scripts: {e}")

# Camera properties
F_PIXEL = (18.14764 * 1280) / 20.955
CX, CY  = 640, 360

# stability + confidence parameters
STABILITY_COUNT = 3
STABILITY_THRESHOLD = 0.05 

# calibration offsets (they are not the ground truth coordinates, but rather guiding offsets)
OFFSETS = {
    "cube": {
        "red":   [0.070, -0.025, 0.0],
        "green": [0.197,  0.234, 0.0],
        "blue":  [0.071, -0.372, 0.0]
    },
    "pallet": {
        "red":   [0.427, -0.295, 0.0],
        "blue":  [0.220,  1.025, 0.0], 
        "black": [-0.084, 0.294, 0.0]
    }
}

def apply_calibration(xyz, color, t_type):
    """
    function to callibrate the coordinates depending of the color and material type (cube/pallet)
    """
    corrected = np.array(xyz)
    if t_type in OFFSETS and color in OFFSETS[t_type]:
        off = OFFSETS[t_type][color]
        corrected += np.array(off)
        print(f"##### APPLIED OFFSET TO {color} {t_type}: {off} #####")
    return corrected

# OPTIONAL
def spawn_marker(position, color_name):
    # function (optional) to create a visual sphere to represent the predicted coordinates
    c_map = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1], "black": [0.1,0.1,0.1]}
    rgb = c_map.get(color_name.lower(), [1,1,0])
    try:
        VisualSphere(
            prim_path="/World/detection_marker",
            name="detection_marker",
            position=np.array(position, dtype=np.float32),
            radius=0.03 if TARGET_TYPE == "cube" else 0.1,
            color=np.array(rgb),
        )
    except: pass

def unproject(u, v, depth_map, cam_matrix):
    """
    Function to translate the given coordinates to real world ones
    """
    u, v = int(np.clip(u, 0, RESOLUTION[0]-1)), int(np.clip(v, 0, RESOLUTION[1]-1))
    z_depth = depth_map[v, u]
    
    if not np.isfinite(z_depth) or z_depth <= 0:
        patch = depth_map[max(0,v-1):v+2, max(0,u-1):u+2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) == 0: return None
        z_depth = np.median(valid)

    x_c = (u - CX) * z_depth / F_PIXEL
    y_c = (v - CY) * z_depth / F_PIXEL
    z_c = -z_depth

    local_p = np.array([x_c, y_c, z_c, 1.0])
    world_p = np.dot(cam_matrix, local_p)
    return world_p[:3]

async def main_vision():
    print(f"##### LOOKING FOR {TARGET_COLOR.upper()} {TARGET_TYPE.upper()} #####")
    
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    dep_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_ann.attach([rp]); dep_ann.attach([rp])

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    
    # necessary so the camera physics don't explode xd
    for _ in range(20): await rep.orchestrator.step_async()

    last_xyz, consecutive = None, 0

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_ann.get_data()
        dep_data = dep_ann.get_data()
        if rgb_data is None: continue

        # condition to use one CV function or another
        if TARGET_TYPE == "pallet":
            proc_rgb, found = detect_palettes(rgb_data, TARGET_COLOR)
        else:
            proc_rgb, found = detect_cubes(rgb_data, TARGET_COLOR)

        if not found: continue

        # we send the image to the vlm
        img = PILImage.fromarray(proc_rgb)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # we send the request to the server
        payload = {"color": TARGET_COLOR, "image_b64": img_str, "target_type": TARGET_TYPE}
        try:
            res = await asyncio.get_event_loop().run_in_executor(None, 
                  lambda: requests.post(URL_MULTI, json=payload, timeout=10).json())
            
            if res and res.get("target", {}).get("found"):
                bbox = res["target"]["bbox_xyxy"]
                u_f = int(((bbox[1] + bbox[3]) / 2000.0) * RESOLUTION[0])
                v_f = int(((bbox[0] + bbox[2]) / 2000.0) * RESOLUTION[1])

                tf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
                cam_mat = np.array(tf).reshape(4, 4).T
                xyz_raw = unproject(u_f, v_f, dep_data, cam_mat)

                if xyz_raw is not None:
                    if TARGET_TYPE == "cube": xyz_raw[2] = 0.015
                    
                    xyz = apply_calibration(xyz_raw, TARGET_COLOR, TARGET_TYPE)

                    # stability logic
                    if last_xyz is not None and np.linalg.norm(xyz - last_xyz) < STABILITY_THRESHOLD:
                        consecutive += 1
                    else:
                        consecutive = 1
                        last_xyz = xyz

                    if consecutive >= STABILITY_COUNT:
                        print(f"##### TARGET LOCKED! COORDINATES: {xyz} #####")
                        spawn_marker(xyz, TARGET_COLOR)
                        return {"world_xyz": xyz}

        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(0.05)

async def run():
    # MAIN FUNCTION
    target_data = await main_vision()
    if not target_data: return

    t_pos = np.array(target_data["world_xyz"])
    r_pos, _ = get_world_pose(ROBOT_PATH)
    
    # json data we save
    data_save = {
        "target_type": TARGET_TYPE,
        "color": TARGET_COLOR,
        "relative_pos": (t_pos - np.array(r_pos)).tolist(),
        "world_pos": t_pos.tolist(),
        "robot_world_pos": r_pos.tolist(),
        "status": "calibrated"
    }

    path = os.path.expanduser(f"~/Documents/PFG/Scripts/Control/rl_{TARGET_TYPE}_start.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data_save, f, indent=4)
    
    print(f"\##### SUCCESS #####")

asyncio.ensure_future(run())