# =============================================================================
#  VLM + PPO + RMPFLOW FULL PIPELINE — Isaac Sim (PFG FINAL VERSION)
#  Flujo: VLM → RMPFlow pre-grasp → PPO grasp (sin lift)
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
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
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from stable_baselines3 import PPO

# ── PATH SETUP ────────────────────────────────────────────────────────────────
scripts_path = "/home/unaiolaizolaosa/Documents/PFG/Scripts"
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

try:
    from Control.franka_stop import execute_movement
    print(f"Movement scripts correctly loaded from: '{scripts_path}'")
except ImportError as e:
    print(f"Error loading scripts from: {scripts_path}")
    raise e

# ── COLLISION SETUP ───────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
robot_path = "/World/Franka_Robot"
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) and robot_path in str(prim.GetPath()):
        utils.set_collider_approximation(prim, "convexHull")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
URL         = "http://127.0.0.1:8000/ground"
RESOLUTION  = (1280, 720)
INSTRUCTION = "red cube"
JSON_FOLDER = os.path.expanduser("~/Documents/PFG/Scripts/Control/")
JSON_FILE   = "rl_start_near_cube_v2.json"

# ⚠️ Actualizar con la ruta del modelo nuevo cuando termine el entrenamiento
MODEL_PATH = "/home/unaiolaizolaosa/Documents/IsaacLab/logs/sb3/Isaac-Lift-Cube-Franka-IK-Abs-VLM-v0/FECHA/model.zip"


# =============================================================================
#  VLM HELPERS
# =============================================================================

def get_prediction(instruction, rgb_image):
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"instruction": instruction, "image_b64": img_str}
    response = requests.post(URL, json=payload, timeout=10)
    return response.json()


def get_3d_target_calibrated(u, v, depth_map, cam_matrix):
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return np.array([0.0, 0.0, -1.0])
    f_pixel = (18.14 * 1280) / 20.955
    cx, cy = 640, 360
    x_cam = (u - cx) * z_depth / f_pixel
    y_cam = (v - cy) * z_depth / f_pixel
    z_cam = -z_depth
    target_pos_local = np.array([x_cam, y_cam, z_cam, 1.0])
    target_pos_world = np.dot(cam_matrix, target_pos_local)
    return target_pos_world[:3]


# =============================================================================
#  PPO HELPERS
# =============================================================================

def build_obs(joint_pos, joint_vel, ee_pos, ee_quat,
              obj_pos_world, gripper, target_pos, target_quat, prev_action):
    """
    Observation space (48,) — replica exacta de ik_abs_env_cfg.py:
     0-8   joint_pos       9
     9-17  joint_vel       9
    18-25  prev_action     8
    26-28  target_pos      3
    29-32  target_quat     4
    33-35  ee_pos          3
    36-39  ee_quat         4
    40-42  obj_pos_scaled  3  (X,Y negados)
    43-44  gripper         2
    45-47  obj_rel_ee      3
    """
    obj_pos_scaled = obj_pos_world * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    obj_rel_ee     = obj_pos_world - ee_pos

    obs = np.concatenate([
        joint_pos,       # 9
        joint_vel,       # 9
        prev_action,     # 8
        target_pos,      # 3
        target_quat,     # 4
        ee_pos,          # 3
        ee_quat,         # 4
        obj_pos_scaled,  # 3
        gripper,         # 2
        obj_rel_ee,      # 3
    ]).astype(np.float32)

    assert obs.shape == (48,), f"[ERROR] Obs shape: {obs.shape}"
    return obs.reshape(1, -1)


# =============================================================================
#  VISION LOOP
# =============================================================================

async def main_vision():
    print("-" * 50 + " INITIALIZING RENDERER " + "-" * 50)
    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except Exception as e:
        print(f"Cleanup skipped: {e}")

    rp          = rep.create.render_product("/World/Camera_01", resolution=RESOLUTION)
    rgb_annot   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage       = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_01")

    for _ in range(10):
        await rep.orchestrator.step_async()

    loop = asyncio.get_event_loop()
    consecutive_detections = 0
    stability_count        = 3
    last_stable_xyz        = np.array([0.0, 0.0, 0.0])

    print("STARTING COORDINATE SEARCH...")
    while True:
        await rep.orchestrator.step_async()
        rgb_data   = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        y_start, y_end = 200, 600
        x_start, x_end = 400, 1000
        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            try:
                result = await loop.run_in_executor(
                    None, get_prediction, INSTRUCTION, cropped_img
                )
                if result and result.get("target") and result["target"].get("found"):
                    raw_bbox = result["target"]["bbox_xyxy"]
                    crop_h, crop_w = (y_end - y_start), (x_end - x_start)
                    v_crop  = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                    u_crop  = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                    u_final = int(np.clip(u_crop + x_start, 0, 1279))
                    v_final = int(np.clip(v_crop + y_start, 0, 719))

                    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                    cam_matrix      = np.array(world_transform).reshape(4, 4).T
                    current_xyz     = get_3d_target_calibrated(u_final, v_final, depth_data, cam_matrix)

                    if current_xyz[2] < -0.1 or current_xyz[0] > 2.0:
                        continue

                    distance = np.linalg.norm(current_xyz - last_stable_xyz)
                    if distance < 0.05:
                        consecutive_detections += 1
                        print(f"Stable detections: {consecutive_detections}/{stability_count}")
                    else:
                        consecutive_detections = 1
                        last_stable_xyz        = current_xyz

                    if consecutive_detections >= stability_count:
                        print(f"--- TARGET LOCKED: {last_stable_xyz} ---")
                        VisualSphere(
                            prim_path="/World/green_target",
                            name="green_target",
                            position=last_stable_xyz,
                            radius=0.02,
                            color=np.array([0, 1, 0]),
                        )
                        return {"world_xyz": last_stable_xyz, "raw_json": result}
            except Exception as e:
                print(f"VLM Connection error: {e}")
        await asyncio.sleep(0.1)


# =============================================================================
#  MAIN RUN
# =============================================================================

async def run():

    # ── 1. VLM: localizar cubo ────────────────────────────────────────────
    target_data = await main_vision()
    target_pos  = target_data["world_xyz"]

    # ── 2. RMPFlow pre-grasp ──────────────────────────────────────────────
    grasp_height = 0.045
    final_coords = [target_pos[0], target_pos[1], grasp_height]
    print(f"Target locked: {target_pos} -> Pre-grasp: {final_coords}")
    await execute_movement(final_coords)
    print("Pre-grasp position reached.")

    # ── 3. PPO grasp via RMPFlow ──────────────────────────────────────────
    print("Loading PPO model...")
    franka_art = Articulation("/World/Franka_Robot")
    franka_art.initialize()
    franka_rmp = Franka(prim_path="/World/Franka_Robot", name="franka_ppo")
    franka_rmp.initialize()
    rmp = RMPFlowController(name="ppo_rmp", robot_articulation=franka_rmp)

    model = PPO.load(MODEL_PATH, device="cpu")
    print(f"PPO loaded | obs={model.observation_space.shape} | act={model.action_space.shape}")

    MAX_STEPS       = 300
    grasp_success   = False
    prev_action     = np.zeros(8, dtype=np.float32)
    obj_pos_world   = np.array(target_pos, dtype=np.float32)
    target_quat_obs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    print("Starting PPO loop...")
    for step in range(MAX_STEPS):
        j_pos = np.array(franka_art.get_joint_positions()).flatten().astype(np.float32)
        j_vel = np.array(franka_art.get_joint_velocities()).flatten().astype(np.float32)
        ee_p, ee_q = get_world_pose("/World/Franka_Robot/panda_hand")

        obs = build_obs(
            j_pos, j_vel,
            np.array(ee_p, dtype=np.float32),
            np.array(ee_q, dtype=np.float32),
            obj_pos_world,
            j_pos[7:9],
            obj_pos_world,
            target_quat_obs,
            prev_action,
        )

        action, _   = model.predict(obs, deterministic=True)
        action      = np.array(action).flatten().astype(np.float32)
        prev_action = action

        # Safety: no bajar de 0.02m para evitar colisión con la mesa
        ee_target    = action[:3].copy()
        ee_target[2] = max(ee_target[2], 0.02)

        rmp_actions = rmp.forward(
            target_end_effector_position=ee_target,
            target_end_effector_orientation=None
        )
        franka_rmp.apply_action(rmp_actions)

        if action[7] > 0:
            franka_rmp.gripper.close()
        else:
            franka_rmp.gripper.open()

        await rep.orchestrator.step_async()

        dist = np.linalg.norm(np.array(ee_p) - obj_pos_world)
        if dist < 0.04 and action[7] > 0:
            print(f"✓ GRASP SUCCESS at step {step} | dist={dist:.4f}m")
            grasp_success = True
            for _ in range(30):
                franka_rmp.gripper.close()
                await rep.orchestrator.step_async()
            break

        if step % 50 == 0:
            gripper_state = "CLOSE" if action[7] > 0 else "OPEN"
            print(f"[{step:3d}/{MAX_STEPS}] dist={dist:.4f}m | gripper={gripper_state} | ee_target={ee_target.round(3)}")

    if not grasp_success:
        print("✗ PPO Timeout - Forcing gripper close")
        for _ in range(40):
            franka_rmp.gripper.close()
            await rep.orchestrator.step_async()

    print("Pipeline complete!")


# =============================================================================
#  ENTRY POINT
# =============================================================================
asyncio.ensure_future(run())
