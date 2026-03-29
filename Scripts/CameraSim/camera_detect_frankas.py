import requests
import base64
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
import omni
import omni.replicator.core as rep
import asyncio
from pxr import UsdGeom
from omni.isaac.core.objects import VisualSphere
import os
import json
import cv2 

# --- CONFIGURACIÓN DE ESCENA ---
CAMERA_PATH = "/World/Cameras/Camera_01" 
RESOLUTION  = (1280, 720)
URL_MULTI   = "http://127.0.0.1:8000/ground_multi"

# Calibración Óptica (Basada en Camera_01)
F_PIXEL = (18.14764 * 1280) / 20.955
CX, CY  = 640, 360

# --- CAPA DE CALIBRACIÓN FRANKAS ---
# 'left' (cerca de cubos), 'right' (cerca de palets)
OFFSETS_FRANKAS = {
    "franka_left": [0.0, 0.0, 0.0], 
    "franka_right": [0.0, 0.0, 0.0]  
}

def preprocess_robots_advanced(rgb_image):
    """
    CV Avanzado: Aísla las bases blancas de los Frankas.
    Borra cubos, mesa y suaviza la esfera central de prueba.
    """
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # 2. Umbralización agresiva (Solo el blanco más puro del robot)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    
    # 3. Limpieza Morfológica (Opening con kernel 25x25)
    # Elimina ruidos pequeños (cubos) y fragmenta la esfera si no es sólida
    kernel = np.ones((25,25), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Guardar debug visual para tu memoria del PFG
    debug_path = "debug_franka_mask.png"
    cv2.imwrite(debug_path, cleaned)
    
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

def unproject(u, v, depth_map, cam_matrix):
    """Transformación de Píxel a Coordenadas de Mundo."""
    u, v = int(np.clip(u, 0, RESOLUTION[0]-1)), int(np.clip(v, 0, RESOLUTION[1]-1))
    
    # Parche de profundidad 5x5 para promediar la masa del robot
    patch = depth_map[max(0,v-2):v+3, max(0,u-2):u+3]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    
    if len(valid) == 0: return None
    z_depth = np.median(valid)

    x_c = (u - CX) * z_depth / F_PIXEL
    y_c = (v - CY) * z_depth / F_PIXEL
    z_c = -z_depth

    local_p = np.array([x_c, y_c, z_c, 1.0])
    return np.dot(cam_matrix, local_p)[:3]

async def detect_all_frankas():
    print("\n--- INICIANDO DETECCIÓN HÍBRIDA DE FRANKAS ---")
    
    # Configurar Replicator
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    dep_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_ann.attach([rp]); dep_ann.attach([rp])
    
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)

    # Warm-up de la cámara
    for _ in range(30): await rep.orchestrator.step_async()

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_ann.get_data()
        dep_data = dep_ann.get_data()
        
        if rgb_data is None: continue

        # 1. Capa CV: Crear máscara limpia
        clean_img = preprocess_robots_advanced(rgb_data)

        # 2. Capa VLM: Inferencia con validación Pydantic
        img_pil = PILImage.fromarray(clean_img)
        buf = BytesIO(); img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # PAYLOAD CORREGIDO: 'color' incluido para evitar ValidationError
        payload = {
            "image_b64": img_b64, 
            "target_type": "all_robots_franka",
            "color": "white" 
        }
        
        try:
            res = await asyncio.get_event_loop().run_in_executor(None, 
                  lambda: requests.post(URL_MULTI, json=payload, timeout=20).json())

            # Procesar lista de targets
            detections = res.get("targets", [])
            if not detections and res.get("target"): detections = [res["target"]]
            
            if not detections:
                print("Esperando detección de robots...")
                continue

            # Matriz de cámara
            tf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
            cam_mat = np.array(tf).reshape(4, 4).T
            
            raw_xyz_list = []
            for det in detections:
                if det.get("found"):
                    bbox = det["bbox_xyxy"]
                    u_f = int(((bbox[1] + bbox[3]) / 2000.0) * RESOLUTION[0])
                    v_f = int(((bbox[0] + bbox[2]) / 2000.0) * RESOLUTION[1])
                    
                    xyz = unproject(u_f, v_f, dep_data, cam_mat)
                    if xyz is not None:
                        xyz[2] = 0.0 # Proyección a suelo
                        raw_xyz_list.append(xyz)

            # 3. Capa de Lógica: Diferenciar por posición X
            if len(raw_xyz_list) >= 2:
                raw_xyz_list.sort(key=lambda p: p[0]) # El menor X es el de la izquierda
                
                xyz_l = raw_xyz_list[0] + np.array(OFFSETS_FRANKAS["franka_left"])
                xyz_r = raw_xyz_list[1] + np.array(OFFSETS_FRANKAS["franka_right"])

                print(f"FRANKS DETECTED -> LEFT: {xyz_l} | RIGHT: {xyz_r}")
                
                # Marcadores Visuales
                VisualSphere(prim_path="/World/Marker_Franka_L", name="M_L", 
                             position=np.array(xyz_l, dtype=np.float32), radius=0.12, color=np.array([1,1,1]))
                VisualSphere(prim_path="/World/Marker_Franka_R", name="M_R", 
                             position=np.array(xyz_r, dtype=np.float32), radius=0.12, color=np.array([1,1,1]))
                
        except Exception as e:
            print(f"Error de red/servidor: {e}")
        
        await asyncio.sleep(0.5)

# Lanzar proceso
asyncio.ensure_future(detect_all_frankas())