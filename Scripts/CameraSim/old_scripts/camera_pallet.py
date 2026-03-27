# =============================================================================
#  VLM PALLET DETECTION — Optimized for Large Objects (Camera_02)
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

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PALLET_COLOR = "red"  # Opciones: "red", "blue", "black"

URL_MULTI    = "http://127.0.0.1:8000/ground_multi"
RESOLUTION   = (1280, 720)
ROBOT_PATH   = "/World/Franka_Robot"
CAMERA_PATH  = "/World/Cameras/Camera_02"

# Stability logic
STABILITY_COUNT     = 3
STABILITY_THRESHOLD = 0.08 # Un poco más permisivo por el tamaño del objeto

# Intrínsecos Camera_02 (Ajusta si la focal es distinta a la 01, pero probemos con esta)
F_PIXEL = (18.14 * 1280) / 20.955
CX, CY  = 640, 360

# =============================================================================
#  HELPERS (ADAPTADOS PARA PALÉS)
# =============================================================================

def call_multi(color, rgb_image):
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = PILImage.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # IMPORTANTE: Cambiamos el prompt dinámicamente para buscar "pallets"
    payload = {"color": f"{color} pallet", "image_b64": img_str}
    
    response = requests.post(URL_MULTI, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def unproject(u, v, depth_map, cam_matrix):
    u = int(np.clip(u, 0, RESOLUTION[0] - 1))
    v = int(np.clip(v, 0, RESOLUTION[1] - 1))
    
    # Parche más grande (7x7) porque los palés tienen huecos/rejillas
    v_min, v_max = max(0, v-3), min(RESOLUTION[1], v+4)
    u_min, u_max = max(0, u-3), min(RESOLUTION[0], u+4)
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

def is_valid_pallet(xyz):
    """Filtros de validación para el palé en el suelo."""
    if xyz is None or not np.all(np.isfinite(xyz)): return False
    # El palé es más alto que el cubo, pero sigue en el suelo
    if xyz[2] < -0.05 or xyz[2] > 0.20: return False 
    return True

# =============================================================================
#  MAIN VISION LOOP
# =============================================================================

async def pallet_vision():
    print(f"--- BUSCANDO PALÉ {PALLET_COLOR.upper()} EN CAMERA_02 ---")

    try: rep.orchestrator.stop()
    except: pass

    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)

    for _ in range(10): await rep.orchestrator.step_async()

    loop = asyncio.get_event_loop()
    consecutive = 0
    last_xyz = None

    while True:
        print(">> Capturando frame...") # DEBUG
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is None: continue

        try:
            result = await loop.run_in_executor(None, call_multi, PALLET_COLOR, rgb_data)

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]
                
                # Coordenadas a píxeles
                v_norm, u_norm = (raw_bbox[0]+raw_bbox[2])/2000.0, (raw_bbox[1]+raw_bbox[3])/2000.0
                u_f, v_f = int(u_norm * RESOLUTION[0]), int(v_norm * RESOLUTION[1])

                world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_transform).reshape(4, 4).T

                xyz = unproject(u_f, v_f, depth_data, cam_matrix)

                if not is_valid_pallet(xyz): continue

                # Estabilidad
                if last_xyz is None:
                    last_xyz = xyz
                    consecutive = 1
                else:
                    dist = np.linalg.norm(xyz - last_xyz)
                    if dist < STABILITY_THRESHOLD:
                        consecutive += 1
                        print(f"Palé estable: {consecutive}/{STABILITY_COUNT}")
                    else:
                        consecutive = 1
                        last_xyz = xyz

                if consecutive >= STABILITY_COUNT:
                    return {"world_xyz": last_xyz}

        except Exception as e:
            print(f"Error: {e}")
        await asyncio.sleep(0.1)

# =============================================================================
#  RUN
# =============================================================================

async def run_pallet_task():
    target_data = await pallet_vision()
    if not target_data: return

    target_pos = np.array(target_data["world_xyz"])
    robot_pos, _ = get_world_pose(ROBOT_PATH)
    relative = target_pos - np.array(robot_pos)

    print(f"\n✅ PALÉ {PALLET_COLOR.upper()} LOCALIZADO")
    print(f"Mundo: {target_pos}")
    print(f"Relativo al Robot: {relative}")

    # Guardar para RL
    out_path = os.path.expanduser("~/Documents/PFG/Scripts/Control/pallet_start_pose.json")
    with open(out_path, "w") as f:
        json.dump({"pallet_pos_rel": relative.tolist(), "color": PALLET_COLOR}, f, indent=4)
    print(f"Configuración guardada en: {out_path}")

asyncio.ensure_future(run_pallet_task())