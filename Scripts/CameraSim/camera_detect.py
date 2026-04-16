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

TARGET_TYPE  = "cube"   # cube / pallet / robot
TARGET_COLOR = "blue"   # red, green, blue, black (for pallet) , white (for robots)
RESOLUTION   = (1280, 720) # screen resolution
ROBOT_PATH   = "/World/Franka_Robot" # franka name in environment
URL_MULTI = "http://127.0.0.1:8000/ground_multi" # endpoint of the function we will use inside our bentoml server

# Camera config: Camera_01 for cubes, Camera_02 for pallets
CAMERA_PATH = "/World/Cameras/Camera_02" if TARGET_TYPE == "pallet" else "/World/Cameras/Camera_01" if TARGET_TYPE == "cube" else "/World/Cameras/Camera_03" # if target equals robot

# We load the CV detection scripts
sys.path.append(os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV"))
try:
    from detect_cubes import detect as detect_cubes
    from detect_palettes import detect_palette 
    from detect_frankas import detect as detect_frankas
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
        "green": [0.222,  0.133, 0.0], 
        "blue":  [0.004, -0.005, 0.0]
    },
    "pallet": {
        "red":   [0.427, -0.295, 0.0],
        "blue":  [0.220,  1.025, 0.0], 
        "black": [-0.084, 0.294, 0.0]
    },
    "robot": {
        "left":  [0.56, -0.39, 0.0], 
        "right": [-1.62, -0.93, 0.0]
    }
}

def apply_calibration(xyz, color, t_type):
    """
    function to callibrate the coordinates depending of the color and material type (cube/pallet)
    """
    corrected = np.array(xyz)
    if t_type == "robot": 
        key = "left" if xyz[0] < 2.0 else "right"
        offset = OFFSETS["robot"][key]
        corrected += np.array(offset)
        print(f"##### APPLIED ROBOT OFFSET ({key}): {offset} #####")
    elif t_type in OFFSETS and color in OFFSETS[t_type]:
        offset = OFFSETS[t_type][color]
        corrected += np.array(offset)
        print(f"##### APPLIED OFFSET TO {color} {t_type}: {offset} #####")
    return corrected

# OPTIONAL
def spawn_marker(position, color_name, suffix=""):
    # function (optional) to create a visual sphere to represent the predicted coordinates
    c_map = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1], "black": [0.1,0.1,0.1], "white": [.0, 1, .0]} # Verde para robots
    rgb = c_map.get(color_name.lower(), [1,1,0])
    try:
        VisualSphere(
            prim_path=f"/World/detection_marker_{suffix}",
            name=f"detection_marker_{suffix}",
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
    print(f"##### BUSCANDO {TARGET_TYPE.upper()} {TARGET_COLOR.upper()} #####")
    
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    dep_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_ann.attach([rp]); dep_ann.attach([rp])

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    
    for _ in range(20): await rep.orchestrator.step_async()

    final_results = {}
    tracking_state = {}

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_ann.get_data()
        dep_data = dep_ann.get_data()
        
        if rgb_data is None or dep_data is None: 
            continue

        if TARGET_TYPE == "pallet":
            proc_rgb, found = detect_palette(rgb_data, TARGET_COLOR)
        elif TARGET_TYPE == "robot":
            proc_rgb, found = detect_frankas(rgb_data, "white")
        else:
            proc_rgb, found = detect_cubes(rgb_data, TARGET_COLOR)

        if not found: 
            continue

        img = PILImage.fromarray(proc_rgb)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = {
            "color": "white" if TARGET_TYPE == "robot" else TARGET_COLOR, 
            "image_b64": img_str, 
            "target_type": TARGET_TYPE
        }

        try:
            res = await asyncio.get_event_loop().run_in_executor(None, 
                  lambda: requests.post(URL_MULTI, json=payload, timeout=30).json())
            
            detections = res.get("targets", [])
            if not detections and res.get("target"): 
                detections = [res["target"]]
            
            for det in detections:
                if det.get("found"):
                    bbox = det["bbox_xyxy"]
                    u_f = int(((bbox[1] + bbox[3]) / 2000.0) * RESOLUTION[0])
                    v_f = int(((bbox[0] + bbox[2]) / 2000.0) * RESOLUTION[1])

                    tf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
                    cam_mat = np.array(tf).reshape(4, 4).T
                    xyz_raw = unproject(u_f, v_f, dep_data, cam_mat)

                    if xyz_raw is not None:
                        if TARGET_TYPE == "robot":
                            if 1.2 < xyz_raw[0] < 3.2:
                                print(f"##### DETECTION IGNORED, MAY BE JETBOT: {xyz_raw[0]:.2f} #####")
                                continue
                        
                        if TARGET_TYPE == "cube":
                            if TARGET_COLOR == "blue":
                                xyz_raw[2] = 0.315  # La nueva altura de tu cubo azul
                            else:
                                xyz_raw[2] = 0.015
                        
                        xyz = apply_calibration(xyz_raw, TARGET_COLOR, TARGET_TYPE)

                        obj_id = ("left" if xyz[0] < 2.0 else "right") if TARGET_TYPE == "robot" else "default"

                        if obj_id not in tracking_state:
                            tracking_state[obj_id] = [xyz, 1]
                        else:
                            last_xyz, count = tracking_state[obj_id]
                            if np.linalg.norm(xyz - last_xyz) < STABILITY_THRESHOLD:
                                tracking_state[obj_id] = [xyz, count + 1]
                            else:
                                tracking_state[obj_id] = [xyz, 1]

                        if tracking_state[obj_id][1] >= STABILITY_COUNT and obj_id not in final_results:
                            print(f"##### {TARGET_TYPE.upper()} {obj_id.upper()} DETECTED #####")
                            final_results[obj_id] = xyz
                            spawn_marker(xyz, TARGET_COLOR, obj_id)

            if TARGET_TYPE == "robot" and len(final_results) >= 2:
                return final_results
            elif TARGET_TYPE != "robot" and len(final_results) >= 1:
                return final_results

        except Exception as e:
            print(f"Vision error: {e}")
        
        await asyncio.sleep(0.05)

async def run():
    # MAIN FUNCTION
    all_targets = await main_vision()
    if not all_targets: return

    r_pos, _ = get_world_pose(ROBOT_PATH)
    results_dir = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results")
    os.makedirs(results_dir, exist_ok=True)

    for obj_id, t_pos in all_targets.items():
        # Custom filename if multiple robots
        suffix = f"_{obj_id}" if TARGET_TYPE == "robot" else ""
        filename = f"detection_{TARGET_TYPE}{suffix}_{TARGET_COLOR}.json"
        
        data_save = {
            "target_type": TARGET_TYPE,
            "target_color": TARGET_COLOR,
            "side": obj_id,
            "world_pos": t_pos.tolist(),
            "robot_world_pos": r_pos.tolist(),
            "relative_pos": (t_pos - np.array(r_pos)).tolist(),
            "camera_used": CAMERA_PATH,
            "status": "calibrated_success"
        }

        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(data_save, f, indent=4)
        
        print(f"##### SUCCESS: {TARGET_TYPE.upper()} {obj_id.upper()} SAVED AT {path} #####")

asyncio.ensure_future(run())