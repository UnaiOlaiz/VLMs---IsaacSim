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

# Available colors for each target type
AVAILABLE_COLORS = {
    "cube": ["red", "green", "blue"],
    "pallet": ["red", "blue", "black"],
    "robot": ["white"],
}

# Ask user for target type
print("\n##### SELECT TARGET TYPE #####")
print("1. cube")
print("2. pallet")
print("3. robot")
target_choice = input("Enter choice (1/2/3): ").strip()
target_map = {"1": "cube", "2": "pallet", "3": "robot"}
TARGET_TYPE = target_map.get(target_choice, "pallet")

RESOLUTION = (1280, 720)  # screen resolution
ROBOT_PATH = "/World/Franka_Robot"  # franka name in environment
URL_MULTI = "http://127.0.0.1:8000/ground_multi"  # endpoint of the function we will use inside our bentoml server

# Available colors for selected target
COLORS_TO_TEST = AVAILABLE_COLORS.get(TARGET_TYPE, ["white"])

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
CX, CY = 640, 360

# stability + confidence parameters
STABILITY_COUNT = 3
STABILITY_THRESHOLD = 0.05

# calibration offsets (they are not the ground truth coordinates, but rather guiding offsets)
OFFSETS = {
    "cube": {
        "red": [0.070, -0.025, 0.0],
        "green": [0.222, 0.133, 0.0],
        "blue": [0.004, -0.005, 0.0],
    },
    "pallet": {
        "red": [0.427, -0.295, 0.0],
        "blue": [0.220, 1.025, 0.0],
        "black": [-0.084, 0.294, 0.0],
    },
    "robot": {"left": [0.56, -0.39, 0.0], "right": [-1.62, -0.93, 0.0]},
}


def get_camera_path(target_type):
    """Get camera path based on target type"""
    if target_type == "pallet":
        return "/World/Cameras/Camera_02"
    elif target_type == "cube":
        return "/World/Cameras/Camera_01"
    else:  # robot
        return "/World/Cameras/Camera_03"


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
    c_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "black": [0.1, 0.1, 0.1],
        "white": [0.0, 1, 0.0],
    }  # Verde para robots
    rgb = c_map.get(color_name.lower(), [1, 1, 0])
    try:
        VisualSphere(
            prim_path=f"/World/detection_marker_{suffix}",
            name=f"detection_marker_{suffix}",
            position=np.array(position, dtype=np.float32),
            radius=0.03 if TARGET_TYPE == "cube" else 0.1,
            color=np.array(rgb),
        )
    except:
        pass


def unproject(u, v, depth_map, cam_matrix):
    """
    Function to translate the given coordinates to real world ones
    """
    u, v = int(np.clip(u, 0, RESOLUTION[0] - 1)), int(np.clip(v, 0, RESOLUTION[1] - 1))
    z_depth = depth_map[v, u]
    if not np.isfinite(z_depth) or z_depth <= 0:
        patch = depth_map[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) == 0:
            return None
        z_depth = np.median(valid)

    x_c = (u - CX) * z_depth / F_PIXEL
    y_c = (v - CY) * z_depth / F_PIXEL
    z_c = -z_depth

    local_p = np.array([x_c, y_c, z_c, 1.0])
    world_p = np.dot(cam_matrix, local_p)
    return world_p[:3]


def calculate_statistics(detections_list):
    """
    Calculate mean and std for a list of detections
    """
    if not detections_list:
        return None, None
    
    detections_array = np.array(detections_list)
    mean = np.mean(detections_array, axis=0)
    std = np.std(detections_array, axis=0)
    return mean, std


async def main_vision_with_tracking(detection_type="cv_offsets", target_color="black", camera_path=None):
    """
    Modified main_vision that supports 3 types of detection:
    - "raw": VLM only (no CV processing, no offsets)
    - "cv": CV detection without offsets
    - "cv_offsets": CV detection with calibration offsets
    """
    if camera_path is None:
        camera_path = get_camera_path(TARGET_TYPE)
    
    print(f"##### BUSCANDO {TARGET_TYPE.upper()} {target_color.upper()} ({detection_type.upper()}) #####")

    rp = rep.create.render_product(camera_path, resolution=RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    dep_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_ann.attach([rp])
    dep_ann.attach([rp])

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(camera_path)

    for _ in range(20):
        await rep.orchestrator.step_async()

    final_results = {}
    tracking_state = {}

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_ann.get_data()
        dep_data = dep_ann.get_data()

        if rgb_data is None or dep_data is None:
            continue

        # Process image based on detection type
        if detection_type == "raw":
            # VLM only - send raw RGB without CV preprocessing
            proc_rgb = rgb_data
            found = True
        else:
            # CV detection (both "cv" and "cv_offsets" use CV preprocessing)
            if TARGET_TYPE == "pallet":
                proc_rgb, found = detect_palette(rgb_data, target_color)
            elif TARGET_TYPE == "robot":
                proc_rgb, found = detect_frankas(rgb_data, "white")
            else:
                proc_rgb, found = detect_cubes(rgb_data, target_color)

        if not found:
            continue

        img = PILImage.fromarray(proc_rgb)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "color": "white" if TARGET_TYPE == "robot" else target_color,
            "image_b64": img_str,
            "target_type": TARGET_TYPE,
        }

        try:
            res = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.post(URL_MULTI, json=payload, timeout=30).json()
            )

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
                                print(
                                    f"##### DETECTION IGNORED, MAY BE JETBOT: {xyz_raw[0]:.2f} #####"
                                )
                                continue
                        
                        # Apply offsets only if detection_type is "cv_offsets"
                        if detection_type == "cv_offsets":
                            xyz = apply_calibration(xyz_raw, color=target_color, t_type=TARGET_TYPE)
                        else:
                            xyz = xyz_raw

                        obj_id = (
                            ("left" if xyz[0] < 2.0 else "right")
                            if TARGET_TYPE == "robot"
                            else "default"
                        )

                        if obj_id not in tracking_state:
                            tracking_state[obj_id] = [xyz, 1]
                        else:
                            last_xyz, count = tracking_state[obj_id]
                            if np.linalg.norm(xyz - last_xyz) < STABILITY_THRESHOLD:
                                tracking_state[obj_id] = [xyz, count + 1]
                            else:
                                tracking_state[obj_id] = [xyz, 1]

                        if (
                            tracking_state[obj_id][1] >= STABILITY_COUNT
                            and obj_id not in final_results
                        ):
                            print(f"##### {TARGET_TYPE.upper()} {obj_id.upper()} DETECTED ({detection_type.upper()}) #####")
                            final_results[obj_id] = xyz
                            spawn_marker(xyz, target_color, obj_id)

            if TARGET_TYPE == "robot" and len(final_results) >= 2:
                return final_results
            elif TARGET_TYPE != "robot" and len(final_results) >= 1:
                return final_results

        except Exception as e:
            print(f"Vision error: {e}")

        await asyncio.sleep(0.05)


async def run():
    # MAIN FUNCTION - Run 5 detections with 3 detection types for each color
    NUM_RUNS = 5
    DETECTION_TYPES = ["raw", "cv", "cv_offsets"]
    results_dir = os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV+VLM_results")
    os.makedirs(results_dir, exist_ok=True)
    
    r_pos, _ = get_world_pose(ROBOT_PATH)
    camera_path = get_camera_path(TARGET_TYPE)
    
    print(f"\n##### TESTING {TARGET_TYPE.upper()} WITH COLORS: {COLORS_TO_TEST} #####\n")
    
    # Iterate over each color
    for color_idx, target_color in enumerate(COLORS_TO_TEST, 1):
        print(f"\n\n{'='*60}")
        print(f"COLOR {color_idx}/{len(COLORS_TO_TEST)}: {target_color.upper()}")
        print(f"{'='*60}\n")
        
        # Storage for all runs: {obj_id: {detection_type: [positions]}}
        all_runs = {}
        
        print(f"##### STARTING {NUM_RUNS} DETECTION RUNS FOR {target_color.upper()} #####\n")
        
        for run_num in range(NUM_RUNS):
            print(f"\n========== RUN {run_num + 1}/{NUM_RUNS} ==========")
            
            for detection_type in DETECTION_TYPES:
                print(f"  - Running {detection_type.upper()} detection...")
                targets = await main_vision_with_tracking(
                    detection_type=detection_type,
                    target_color=target_color,
                    camera_path=camera_path
                )
                
                if targets:
                    for obj_id, pos in targets.items():
                        if obj_id not in all_runs:
                            all_runs[obj_id] = {dt: [] for dt in DETECTION_TYPES}
                        all_runs[obj_id][detection_type].append(pos)
        
        print(f"\n##### CALCULATING STATISTICS FOR {target_color.upper()} #####\n")
        
        # Calculate statistics for each object
        for obj_id in all_runs.keys():
            suffix = f"_{obj_id}" if TARGET_TYPE == "robot" else ""
            filename = f"detection_statistics_{TARGET_TYPE}{suffix}_{target_color}.json"
            
            data_save = {
                "target_type": TARGET_TYPE,
                "target_color": target_color,
                "side": obj_id,
                "num_runs": NUM_RUNS,
                "camera_used": camera_path,
                "robot_world_pos": r_pos.tolist(),
            }
            
            for detection_type in DETECTION_TYPES:
                detections_list = all_runs[obj_id].get(detection_type, [])
                mean, std = calculate_statistics(detections_list)
                
                data_save[detection_type] = {
                    "description": f"{detection_type.replace('_', ' ').title()} detection",
                    "detections": [det.tolist() for det in detections_list],
                    "mean": mean.tolist() if mean is not None else None,
                    "std": std.tolist() if std is not None else None,
                }
                
                print(f"{obj_id} - {detection_type.upper()}: Mean={mean}, Std={std}")

            path = os.path.join(results_dir, filename)
            with open(path, "w") as f:
                json.dump(data_save, f, indent=4)

            print(f"\n##### SUCCESS: {TARGET_TYPE.upper()} {obj_id.upper()} ({target_color.upper()}) STATISTICS SAVED #####")
    
    print(f"\n\n{'='*60}")
    print(f"##### ALL TESTS COMPLETED #####")
    print(f"Results saved in: {results_dir}")
    print(f"{'='*60}\n")


asyncio.ensure_future(run())
