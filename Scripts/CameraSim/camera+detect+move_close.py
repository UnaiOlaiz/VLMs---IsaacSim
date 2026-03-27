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

TARGET_TYPE  = "cube"  # "cube" o "pallet"
TARGET_COLOR = "green"     # "red", "green", "blue", "black"

URL_MULTI    = "http://127.0.0.1:8000/ground_multi"
RESOLUTION   = (1280, 720)
ROBOT_PATH   = "/World/Franka_Robot"

CAMERA_PATH = "/World/Cameras/Camera_02" if TARGET_TYPE == "pallet" else "/World/Cameras/Camera_01"

STABILITY_COUNT = 1 if TARGET_TYPE == "pallet" else 3
STABILITY_THRESHOLD = 0.5  # 

F_PIXEL = (18.14 * 1280) / 20.955
CX, CY  = 640, 360

def call_multi(color, target_type, rgb_image, timeout=60):
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = PILImage.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    payload = {"color": color, "image_b64": img_str, "target_type": target_type}
    response = requests.post(URL_MULTI, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

def unproject(u, v, depth_map, cam_matrix):
    u = int(np.clip(u, 0, RESOLUTION[0] - 1))
    v = int(np.clip(v, 0, RESOLUTION[1] - 1))
    
    p_size = 5 if TARGET_TYPE == "pallet" else 2
    v_min, v_max = max(0, v-p_size), min(RESOLUTION[1], v+p_size+1)
    u_min, u_max = max(0, u-p_size), min(RESOLUTION[0], u+p_size+1)
    
    depth_patch = depth_map[v_min:v_max, u_min:u_max]
    valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]
    
    if len(valid_depths) == 0: return None
    z_depth = np.median(valid_depths)

    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth

    local_point = np.array([x_cam, y_cam, z_cam, 1.0])
    world_point = np.dot(cam_matrix, local_point)
    return world_point[:3]

def is_valid(xyz):
    if xyz is None: return False
    
    z_ok = -0.50 < xyz[2] < 0.60
    
    dist_val = np.linalg.norm(xyz[:2])
    dist_ok = dist_val < 7.0
    
    if not z_ok: print(f"DEBUG RECHAZO: Z incorrecto ({xyz[2]:.3f}m)")
    if not dist_ok: print(f"DEBUG RECHAZO: Distancia excesiva ({dist_val:.3f}m)")
    
    return z_ok and dist_ok

def spawn_marker(position, color_rgb):
    try:
        VisualSphere(
            prim_path="/World/detection_marker",
            name="detection_marker",
            position=np.array(position, dtype=np.float32),
            radius=0.1, # Marcador grande para ser visible a 4m
            color=np.array(color_rgb),
        )
    except: pass


async def main_vision():
    print(f"--- LOOKING FOR {TARGET_COLOR.upper()} {TARGET_TYPE.upper()} ---")

    try: rep.orchestrator.stop()
    except: pass

    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    
    for _ in range(20): await rep.orchestrator.step_async()

    loop = asyncio.get_event_loop()
    last_xyz = None
    consecutive = 0

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is None: continue

        try:
            result = await loop.run_in_executor(None, call_multi, TARGET_COLOR, TARGET_TYPE, rgb_data)

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]
                conf = result["target"].get("num_valid", 0)
                print(f"Detected {TARGET_TYPE} (Confidence: {conf}/3)")

                v_norm, u_norm = (raw_bbox[0]+raw_bbox[2])/2000.0, (raw_bbox[1]+raw_bbox[3])/2000.0
                u_f, v_f = int(u_norm * RESOLUTION[0]), int(v_norm * RESOLUTION[1])

                world_tf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_tf).reshape(4, 4).T

                xyz = unproject(u_f, v_f, depth_data, cam_matrix)

                if is_valid(xyz):
                    if STABILITY_COUNT <= 1:
                        print(f"--- TARGET LOCKED: {xyz} ---")
                        spawn_marker(xyz, [1,0,0] if TARGET_COLOR=="red" else [0,0,1])
                        return {"world_xyz": xyz}
                    
                    if last_xyz is not None and np.linalg.norm(xyz - last_xyz) < STABILITY_THRESHOLD:
                        consecutive += 1
                        print(f"Estable: {consecutive}/{STABILITY_COUNT}")
                    else:
                        consecutive = 1
                        last_xyz = xyz

                    if consecutive >= STABILITY_COUNT:
                        print(f"--- TARGET LOCKED (STABLE): {xyz} ---")
                        spawn_marker(xyz, [1,0,0] if TARGET_COLOR=="red" else [0,0,1])
                        return {"world_xyz": xyz}

        except Exception as e:
            print(f"Error en inferencia: {e}")
        await asyncio.sleep(0.1)

async def run():
    target_data = await main_vision()
    if not target_data: return

    target_pos = np.array(target_data["world_xyz"])
    robot_pos, _ = get_world_pose(ROBOT_PATH)
    relative = target_pos - np.array(robot_pos)

    data_save = {
        "target_type": TARGET_TYPE,
        "color": TARGET_COLOR,
        "relative_pos": relative.tolist(),
        "world_pos": target_pos.tolist(),
        "robot_world_pos": robot_pos.tolist()
    }

    out_path = os.path.expanduser(f"~/Documents/PFG/Scripts/Control/rl_{TARGET_TYPE}_start.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data_save, f, indent=4)
    
    print(f"\n--- SUCCESS: Data saved in: {out_path} ---")

asyncio.ensure_future(run())