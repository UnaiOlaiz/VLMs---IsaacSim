import numpy as np
import omni
import asyncio
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import VisualSphere
import os
import json
import sys

# --- CONFIGURACIÓN DE RUTAS ---
scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Control.aaaa.franka_controller import execute_movement
    print("✅ Script de movimiento cargado.")
except ImportError as e:
    print(f"❌ Error al cargar execute_movement: {e}")
    raise e

# --- COORDENADAS REALES (Ground Truth) ---
REAL_POSITIONS = {
    "red":   [0.094, 0.0, 0.015],
    "green": [0.09460346, 0.16237024, 0.01499997],
    "blue":  [ 0.06924441, -0.333803,    0.115     ]
}

# --- PARÁMETROS DE EJECUCIÓN ---
color = "red"  # CAMBIAR A: "red", "green", "blue"
ROBOT_PATH   = "/World/Franka_Robot"
GRASP_HEIGHT = 0.15   # <--- Altura aumentada para evitar bloqueos por colisión
MAX_WAIT_STEPS = 500  # Pasos máximos para esperar el movimiento

async def run_ground_truth_collection(TARGET_COLOR):
    GRASP_HEIGHT = 0.15 if TARGET_COLOR == "blue" else 0.015
    if TARGET_COLOR not in REAL_POSITIONS:
        print(f"❌ Color {TARGET_COLOR} no reconocido.")
        return

    target_xyz = np.array(REAL_POSITIONS[TARGET_COLOR])
    move_to = [target_xyz[0], target_xyz[1], GRASP_HEIGHT]

    print(f"\n🎯 MODO GROUND TRUTH: {TARGET_COLOR.upper()}")
    print(f"📍 Objetivo Real: {target_xyz} | Aproximación: {move_to}")
    
    # 1. Marcador Visual
    VisualSphere(
        prim_path=f"/World/gt_marker_{TARGET_COLOR}", 
        position=target_xyz, 
        radius=0.02, 
        color=np.array([1, 1, 0]) 
    )

    # 2. Ejecutar movimiento con vigilancia de tiempo
    print(f"🚀 Moviendo Franka a posición de seguridad...")
    
    # Lanzamos el movimiento como una tarea asíncrona
    move_task = asyncio.ensure_future(execute_movement(move_to))
    
    steps = 0
    while not move_task.done() and steps < MAX_WAIT_STEPS:
        await omni.kit.app.get_app().next_update_async()
        steps += 1
        if steps % 100 == 0:
            print(f"   ... esperando movimiento ({steps}/{MAX_WAIT_STEPS})")

    if not move_task.done():
        print("⚠️ Tiempo de espera agotado o brazo bloqueado. Procediendo con captura...")
        move_task.cancel()

    # Pequeña pausa para que la física se estabilice
    await asyncio.sleep(1.0)

    # 3. Captura de estado del robot
    print("💾 Capturando joints y pose para el dataset...")
    franka_art = Articulation(ROBOT_PATH)
    franka_art.initialize()
    
    robot_world_pos, _ = get_world_pose(ROBOT_PATH)
    rel_pos = target_xyz - np.array(robot_world_pos)
    
    final_data = {
        "target_color": TARGET_COLOR,
        "method": "ground_truth_safe",
        "world_cube_xyz": target_xyz.tolist(),
        "relative_cube_pos": rel_pos.tolist(),
        "joint_positions": franka_art.get_joint_positions().tolist(),
        "joint_velocities": franka_art.get_joint_velocities().tolist(),
        "robot_base_pos": robot_world_pos.tolist(),
        "grasp_height_used": GRASP_HEIGHT
    }

    # 4. Guardar JSON
    out_dir = os.path.expanduser("~/Documents/PFG/Scripts/Control/cube_jsons/")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rl_start_{TARGET_COLOR}_cube.json")

    with open(path, "w") as f:
        json.dump(final_data, f, indent=4)

    print(f"\n✨ DATOS GUARDADOS EN:\n{path}")
    print(f"Status: Movimiento completado en {steps} pasos.")


async def run_all_colors():
    for color in ["red", "green", "blue"]:
        await run_ground_truth_collection(color)


asyncio.ensure_future(run_all_colors())