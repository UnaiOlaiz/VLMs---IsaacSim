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

TARGET_TYPE  = "pallet"  # "cube" o "pallet"
TARGET_COLOR = "black" # "red", "green", "blue", "black"
URL_MULTI    = "http://127.0.0.1:8000/ground_multi"
RESOLUTION   = (1280, 720)
ROBOT_PATH   = "/World/Franka_Robot"

CAMERA_PATH = "/World/Cameras/Camera_02" if TARGET_TYPE == "pallet" else "/World/Cameras/Camera_01"
sys.path.append(os.path.expanduser("~/Documents/PFG/Scripts/CameraSim/CV"))

try:
    from detect_cubes import detect as detect_cubes
    from detect_palettes import detect_palette as detect_palettes
    print("✅ Scripts CV cargados correctamente.")
except ImportError as e:
    print(f"❌ Error cargando scripts CV: {e}")

# Camera properties
F_PIXEL = (18.14764 * 1280) / 20.955
CX, CY  = 640, 360
STABILITY_COUNT = 3
STABILITY_THRESHOLD = 0.05 # 5cm para RL de alta precisión

def call_multi_with_cv(color, target_type, rgb_image):
    """
    Usa los scripts de CV locales para pre-procesar la imagen 
    y luego valida con el VLM.
    """
    # 1. Ejecutar detección CV local según el tipo
    if target_type == "pallet":
        processed_rgb, found = detect_palettes(rgb_image, color)
    else:
        processed_rgb, found = detect_cubes(rgb_image, color)
    
    if not found:
        return None

    # Guardar debug para ver qué está filtrando el CV
    cv2.imwrite("debug_cv_preprocessed.png", cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))

    # 2. Enviar la imagen filtrada al VLM para confirmación y BBox final
    img = PILImage.fromarray(processed_rgb)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    payload = {"color": color, "image_b64": img_str, "target_type": target_type}
    try:
        response = requests.post(URL_MULTI, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error en comunicación VLM: {e}")
        return None

def unproject(u, v, depth_map, cam_matrix):
    u, v = int(np.clip(u, 0, RESOLUTION[0] - 1)), int(np.clip(v, 0, RESOLUTION[1] - 1))
    
    # Parche pequeño para el centro del objeto
    p_size = 1
    depth_patch = depth_map[v-p_size:v+p_size+1, u-p_size:u+p_size+1]
    valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]
    
    if len(valid_depths) == 0: return None
    z_depth = np.nanmedian(valid_depths)

    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth

    local_point = np.array([x_cam, y_cam, z_cam, 1.0])
    world_point = np.dot(cam_matrix, local_point)
    return world_point[:3]

def spawn_marker(position, color):
    c_map = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1], "black": [0.1,0.1,0.1]}
    rgb = c_map.get(color.lower(), [1,1,0])
    try:
        VisualSphere(
            prim_path="/World/detection_marker",
            name="detection_marker",
            position=np.array(position, dtype=np.float32),
            radius=0.03 if TARGET_TYPE == "cube" else 0.1,
            color=np.array(rgb),
        )
    except: pass

async def main_vision():
    print(f"--- BUSCANDO {TARGET_COLOR.upper()} {TARGET_TYPE.upper()} (MODO HÍBRIDO CV+VLM) ---")

    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp]); depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    
    for _ in range(20): await rep.orchestrator.step_async()

    last_xyz, consecutive = None, 0

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()
        if rgb_data is None: continue

        # Llamada híbrida
        result = await asyncio.get_event_loop().run_in_executor(
            None, call_multi_with_cv, TARGET_COLOR, TARGET_TYPE, rgb_data
        )

        if result and result.get("target") and result["target"].get("found"):
            bbox = result["target"]["bbox_xyxy"] # [y1, x1, y2, x2] en escala 1000
            
            # Mapeo a píxeles reales
            u_f = int(((bbox[1] + bbox[3]) / 2000.0) * RESOLUTION[0])
            v_f = int(((bbox[0] + bbox[2]) / 2000.0) * RESOLUTION[1])

            tf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
            cam_mat = np.array(tf).reshape(4, 4).T
            xyz = unproject(u_f, v_f, depth_data, cam_mat)

            if xyz is not None:
                # Ajuste de altura centroide (Z=0.015 para cubos en mesa)
                if TARGET_TYPE == "cube": xyz[2] = 0.015
                
                # Lógica de estabilidad
                if last_xyz is not None and np.linalg.norm(xyz - last_xyz) < STABILITY_THRESHOLD:
                    consecutive += 1
                    print(f"Estable: {consecutive}/{STABILITY_COUNT} en {xyz}")
                else:
                    consecutive = 1
                    last_xyz = xyz

                if consecutive >= STABILITY_COUNT:
                    print(f"--- TARGET LOCKED: {xyz} ---")
                    spawn_marker(xyz, TARGET_COLOR)
                    return {"world_xyz": xyz}
        
        await asyncio.sleep(0.05)

async def run():
    target_data = await main_vision()
    if not target_data: return

    t_pos = np.array(target_data["world_xyz"])
    r_pos, _ = get_world_pose(ROBOT_PATH)
    
    data_save = {
        "target_type": TARGET_TYPE,
        "color": TARGET_COLOR,
        "relative_pos": (t_pos - np.array(r_pos)).tolist(),
        "world_pos": t_pos.tolist(),
        "robot_world_pos": r_pos.tolist()
    }

    path = os.path.expanduser(f"~/Documents/PFG/Scripts/Control/rl_{TARGET_TYPE}_start.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data_save, f, indent=4)
    
    print(f"\n--- SUCCESS: Datos guardados en {path} ---")

asyncio.ensure_future(run())