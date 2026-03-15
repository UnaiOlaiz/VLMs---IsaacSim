# =============================================================================
#  VLM + PPO FULL PIPELINE — Isaac Sim (PFG COORDINATE-FIXED VERSION)
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import omni.replicator.core as rep
import asyncio
import sys
import os
import json

# Isaac Sim / USD Imports
from pxr import UsdPhysics, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils import stage as stage_utils
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import VisualSphere
from omni.physx.scripts import utils
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

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
URL         = "http://127.0.0.1:8000/ground"
RESOLUTION  = (1280, 720)
INSTRUCTION = "red cube"
MODEL_PATH  = "/home/unaiolaizolaosa/Documents/PFG/Models/Lift_03-14/model.zip"

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

async def main_vision():
    print("-" * 30 + " VLM SEARCHING " + "-" * 30)
    rp = rep.create.render_product("/World/Camera_01", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = stage_utils.get_current_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_01")
    
    consecutive_detections = 0
    stability_count = 3
    last_stable_xyz = np.array([0.0, 0.0, 0.0])

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()
        
        if rgb_data is None or len(rgb_data.shape) < 2:
            continue

        y_start, y_end, x_start, x_end = 200, 600, 400, 1000
        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            result = get_prediction(INSTRUCTION, cropped_img)
            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]
                crop_h, crop_w = (y_end - y_start), (x_end - x_start)
                v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                u_final, v_final = int(u_crop + x_start), int(v_crop + y_start)

                world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_transform).reshape(4, 4).T
                current_xyz = get_3d_target_calibrated(u_final, v_final, depth_data, cam_matrix)

                if np.linalg.norm(current_xyz - last_stable_xyz) < 0.05:
                    consecutive_detections += 1
                else:
                    consecutive_detections = 1
                    last_stable_xyz = current_xyz

                if consecutive_detections >= stability_count:
                    print(f"TARGET LOCKED: {last_stable_xyz}")
                    return {"world_xyz": last_stable_xyz}
        await asyncio.sleep(0.1)

# =============================================================================
#  PPO HELPERS
# =============================================================================

def build_obs(joint_pos, joint_vel, ee_pos, ee_quat, obj_pos, gripper, default_joint_pos, prev_action):
    # relative to default pose
    joint_pos_rel = (joint_pos[:7] - default_joint_pos[:7])
    joint_vel_arm = joint_vel[:7]
    
    # RELATIVE POSITION: Hand to Object
    obj_rel_ee = obj_pos - ee_pos
    
    obs = np.concatenate([
        joint_pos_rel, joint_vel_arm, ee_pos, ee_quat, 
        obj_pos, obj_rel_ee, [gripper[0]], prev_action
    ]).astype(np.float32)
    return obs.reshape(1, -1)

# =============================================================================
#  MAIN EXECUTION
# =============================================================================

async def run():
    # 1. Setup
    world = World.instance() or World()
    stage = stage_utils.get_current_stage()
    if not any(prim.GetTypeName() == "PhysicsScene" for prim in stage.Traverse()):
        import omni.kit.commands
        omni.kit.commands.execute("AddPhysicsSceneCommand", stage=stage, path="/World/physicsScene")
    
    print("Resetting World...")
    await world.reset_async()

    # 2. VLM Phase
    target_data = await main_vision()
    cube_world_xyz = target_data["world_xyz"]

    # 3. RMPFlow Phase
    # Move to exactly 15cm above the detected cube
    pre_grasp_coords = [cube_world_xyz[0], cube_world_xyz[1], 0.15]
    print(f"RMPFlow: Moving to {pre_grasp_coords}")
    await execute_movement(pre_grasp_coords)
    for _ in range(80): await rep.orchestrator.step_async()

    # 4. PPO Phase
    print("Starting PPO Inference...")
    franka_art = Articulation("/World/Franka_Robot")
    franka_art.initialize()
    
    # Get robot base to calculate relative coordinates
    robot_base_pos, _ = get_world_pose("/World/Franka_Robot")
    # PPO EXPECTS CUBE POSITION RELATIVE TO ROBOT BASE
    cube_rel_base = cube_world_xyz - robot_base_pos

    default_j_pos = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.04, 0.04], dtype=np.float32)
    model = PPO.load(MODEL_PATH, device="cpu")
    
    prev_action = np.zeros(8, dtype=np.float32)
    for step in range(800):
        j_pos = np.array(franka_art.get_joint_positions()).flatten().astype(np.float32)
        j_vel = np.array(franka_art.get_joint_velocities()).flatten().astype(np.float32)
        ee_p_world, ee_q_world = get_world_pose("/World/Franka_Robot/panda_hand")
        
        # Convert End-Effector to Base Frame
        ee_p_base = np.array(ee_p_world) - np.array(robot_base_pos)

        # Build Observation with base-frame coordinates
        obs = build_obs(j_pos, j_vel, ee_p_base, np.array(ee_q_world), 
                        cube_rel_base, j_pos[7:9], default_j_pos, prev_action)
        
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action).flatten().astype(np.float32)
        prev_action = action

        # Apply Action
        arm_targets = default_j_pos[:7] + (action[:7] * 0.5)
        gripper_val = 0.0 if action[7] > 0 else 0.04
        full_targets = np.concatenate([arm_targets, [gripper_val, gripper_val]])
        franka_art.apply_action(ArticulationAction(joint_positions=full_targets))
        
        await rep.orchestrator.step_async()

        # Success Check (Relative Height)
        # Check world Z height for lifting
        if ee_p_world[2] > (cube_world_xyz[2] + 0.18) and action[7] > 0.5 and step > 100:
            print(f"✓ REAL LIFT DETECTED! Hand is at {ee_p_world[2]:.3f}m")
            break

    print("Pipeline Complete.")

asyncio.ensure_future(run())