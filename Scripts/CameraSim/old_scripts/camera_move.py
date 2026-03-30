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
from omni.isaac.core.articulations import Articulation
import os
import json
import sys

# --- CONFIGURACIÓN DE RUTAS ---
scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Scripts.Control.franka_controller import execute_movement
    print("✅ Script de movimiento cargado correctamente.")
except ImportError as e:
    print(f"❌ Error al cargar execute_movement: {e}")
    raise e

# --- PARÁMETROS DE CÁMARA Y ESCENA ---
TARGET_TYPE  = "cube"
TARGET_COLOR = "blue"      # CAMBIAR A: "red", "green", "blue"
URL_MULTI    = "http://127.0.0.1:8000/ground_multi"
RESOLUTION   = (1280, 720)
ROBOT_PATH   = "/World/Franka_Robot"
CAMERA_PATH  = "/World/Cameras/Camera_01"

# Calibración Estándar Isaac Sim
FOCAL_LENGTH = 18.147 
HORIZ_APERTURE = 20.955
F_PIXEL = (FOCAL_LENGTH * RESOLUTION[0]) / HORIZ_APERTURE
CX, CY  = RESOLUTION[0] / 2, RESOLUTION[1] / 2

# --- LÓGICA DE VISIÓN ---

def call_multi(color, target_type, rgb_image):
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = PILImage.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prompt con guiado espacial relativo para mejorar precisión del VLM
    if color == "green":
        prompt = "the small green cube located to the right of the central red cube"
    elif color == "blue":
        prompt = "the small blue cube located to the left of the central red cube"
    else:
        prompt = "the red cube in the center of the workspace"
    
    payload = {
        "color": color, 
        "image_b64": img_str, 
        "target_type": target_type,
        "custom_prompt": prompt  
    }
    response = requests.post(URL_MULTI, json=payload, timeout=30)
    return response.json()

def unproject_with_median(u, v, depth_map, cam_matrix):
    u, v = int(np.clip(u, 0, RESOLUTION[0]-1)), int(np.clip(v, 0, RESOLUTION[1]-1))
    
    # Muestreo de profundidad en área 4x4 para mayor estabilidad
    depth_patch = depth_map[max(0, v-2):v+2, max(0, u-2):u+2]
    valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]
    
    if len(valid_depths) == 0: return None
    z_depth = np.median(valid_depths)

    # Conversión a coordenadas de cámara (Eje Y invertido para Isaac Sim)
    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = -(v - CY) * z_depth / F_PIXEL 
    z_cam = -z_depth

    local_p = np.array([x_cam, y_cam, z_cam, 1.0])
    world_p = np.dot(cam_matrix, local_p)
    return world_p[:3]

async def get_stable_target():
    print(f"\n--- 📡 BUSCANDO {TARGET_COLOR.upper()} ---")
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp]); depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    loop = asyncio.get_event_loop()
    
    consecutive = 0
    last_xyz = None

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()
        if rgb_data is None: continue

        try:
            res = await loop.run_in_executor(None, call_multi, TARGET_COLOR, TARGET_TYPE, rgb_data)
            if res and res.get("target") and res["target"].get("found"):
                bbox = res["target"]["bbox_xyxy"]
                u_f = int(((bbox[1] + bbox[3]) / 2000.0) * RESOLUTION[0])
                v_f = int(((bbox[0] + bbox[2]) / 2000.0) * RESOLUTION[1])

                world_tf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                cam_mat = np.array(world_tf).reshape(4, 4).T
                xyz = unproject_with_median(u_f, v_f, depth_data, cam_mat)

                if xyz is not None:
                    # Filtro de seguridad direccional (basado en el centro Y=0)
                    is_valid = True
                    if TARGET_COLOR == "green" and xyz[1] < 0.01: is_valid = False
                    if TARGET_COLOR == "blue" and xyz[1] > -0.01: is_valid = False

                    if is_valid:
                        if last_xyz is not None and np.linalg.norm(xyz - last_xyz) < 0.05:
                            consecutive += 1
                            print(f"📍 Estable: {consecutive}/3 en {xyz}")
                        else:
                            consecutive = 1
                            last_xyz = xyz
                        
                        if consecutive >= 3:
                            VisualSphere(prim_path=f"/World/marker_{TARGET_COLOR}", position=xyz, radius=0.015, color=np.array([0, 1, 0]))
                            return xyz
        except Exception as e:
            print(f"Error: {e}")
        await asyncio.sleep(0.1)

# --- FLUJO PRINCIPAL ---

async def run_perception_and_move():
    # 1. Fase de Visión
    target_xyz = await get_stable_target()

    # 2. Fase de Movimiento
    grasp_h = 0.045 # Altura segura sobre el cubo
    move_to = [target_xyz[0], target_xyz[1], grasp_h]
    
    print(f"\n🚀 MOVIENDO FRANKA A: {move_to}")
    await execute_movement(move_to)
    await asyncio.sleep(2.0)

    # 3. Captura de datos para RL
    print("💾 Guardando estado final...")
    franka_art = Articulation(ROBOT_PATH)
    franka_art.initialize()
    
    robot_world_pos, _ = get_world_pose(ROBOT_PATH)
    rel_pos = target_xyz - np.array(robot_world_pos)
    
    final_state = {
        "target_color": TARGET_COLOR,
        "world_cube_xyz": target_xyz.tolist(),
        "relative_cube_pos": rel_pos.tolist(),
        "joint_positions": franka_art.get_joint_positions().tolist(),
        "joint_velocities": franka_art.get_joint_velocities().tolist()
    }

    # 4. Guardado en JSON
    out_dir = os.path.expanduser("~/Documents/PFG/Scripts/Control/cube_jsons/")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rl_start_{TARGET_COLOR}_cube.json")

    with open(path, "w") as f:
        json.dump(final_state, f, indent=4)

    print(f"\n✨ PROCESO COMPLETADO. Datos guardados en:\n{path}")

asyncio.ensure_future(run_perception_and_move())