# =============================================================================
#  VLM + PPO + CONTROLLER FULL PIPELINE — Franka_1 Pick & Place on Jetbot
# =============================================================================

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import omni.replicator.core as rep
import omni.usd
import asyncio
import sys
import os
import json

from pxr import UsdGeom
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
    from Scripts.Control.aaaa.franka_controller import execute_movement, FrankaControl

    print(f"Movement scripts correctly loaded from: '{scripts_path}'")
except ImportError as e:
    print(f"Error loading scripts from: {scripts_path}")
    raise e

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
URL = "http://127.0.0.1:8000/ground"
URL_JETBOT = "http://127.0.0.1:8000/find_jetbot"
RESOLUTION = (1280, 720)
INSTRUCTION = "red cube"
MODEL_PATH = "/home/unaiolaizolaosa/Documents/PFG/Models/Lift_03-14/model.zip"
ROBOT_PATH = "/World/Franka_Robot"
CAMERA_PATH = "/World/Camera_01"

# Crop — same as working script
Y_START, Y_END = 200, 600
X_START, X_END = 400, 1000

# Grasp detection thresholds
LIFT_HEIGHT_THRESHOLD = 0.18
MIN_GRIPPER_ACTION = 0.5
TRANSIT_HEIGHT = 0.25
MAX_RL_STEPS = 800


# =============================================================================
#  CAMERA HELPERS
# =============================================================================


def setup_camera():
    """Initialize replicator annotators for Camera_01."""
    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except:
        pass
    rp = rep.create.render_product(CAMERA_PATH, resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])
    return rgb_annot, depth_annot


def get_3d_target_calibrated(u, v, depth_map, cam_matrix):
    """Same unprojection as working script — no offset."""
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return None
    f_pixel = (18.14 * 1280) / 20.955
    cx, cy = 640, 360
    x_cam = (u - cx) * z_depth / f_pixel
    y_cam = (v - cy) * z_depth / f_pixel
    z_cam = -z_depth
    local = np.array([x_cam, y_cam, z_cam, 1.0])
    world = np.dot(cam_matrix, local)
    return world[:3]


# =============================================================================
#  VLM HELPERS
# =============================================================================


def call_vlm(url, instruction, rgb_image, timeout=30):
    """Send image to BentoML VLM endpoint."""
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"instruction": instruction, "image_b64": img_str}
    response = requests.post(url, json=payload, timeout=timeout)
    return response.json()


async def vlm_detect(
    rgb_annot,
    depth_annot,
    camera_prim,
    url,
    instruction,
    stability_count=3,
    label="target",
    use_crop=True,
):
    """
    Run VLM detection loop until stable detection found.
    Returns world XYZ of detected object.
    use_crop=True: use the working script crop for cube detection
    use_crop=False: use full image for Jetbot detection
    """
    print(f"\n{'=' * 20} VLM: Looking for '{label}' {'=' * 20}")
    loop = asyncio.get_event_loop()
    consecutive = 0
    last_xyz = np.zeros(3)

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is None or rgb_data.size == 0:
            continue

        # Crop — same as working script for cubes, full image for Jetbot
        if use_crop:
            send_img = rgb_data[Y_START:Y_END, X_START:X_END]
        else:
            send_img = rgb_data

        if send_img is None or send_img.size == 0:
            continue

        try:
            result = await loop.run_in_executor(
                None, call_vlm, url, instruction, send_img
            )

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]

                if use_crop:
                    # Same bbox mapping as working script
                    crop_h = Y_END - Y_START
                    crop_w = X_END - X_START
                    v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                    u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                    u_final = int(u_crop + X_START)
                    v_final = int(v_crop + Y_START)
                else:
                    # Full image mapping for Jetbot
                    v_norm = (raw_bbox[0] + raw_bbox[2]) / 2 / 1000
                    u_norm = (raw_bbox[1] + raw_bbox[3]) / 2 / 1000
                    u_final = int(np.clip(u_norm * RESOLUTION[0], 0, RESOLUTION[0] - 1))
                    v_final = int(np.clip(v_norm * RESOLUTION[1], 0, RESOLUTION[1] - 1))

                world_transform = UsdGeom.Xformable(
                    camera_prim
                ).ComputeLocalToWorldTransform(0)
                cam_matrix = np.array(world_transform).reshape(4, 4).T
                xyz = get_3d_target_calibrated(u_final, v_final, depth_data, cam_matrix)

                if xyz is None:
                    print("Invalid depth, skipping")
                    continue

                # Same filter as working script
                if xyz[2] < -0.1 or xyz[0] > 2.0:
                    print(f"Skipping hallucination: {xyz}")
                    continue

                if np.linalg.norm(xyz - last_xyz) < 0.05:
                    consecutive += 1
                    print(f"Stable: {consecutive}/{stability_count} at {xyz}")
                else:
                    consecutive = 1
                    last_xyz = xyz
                    print(f"Jitter reset at: {xyz}")

                if consecutive >= stability_count:
                    print(f"--- {label.upper()} LOCKED: {last_xyz} ---")
                    VisualSphere(
                        prim_path=f"/World/vlm_{label.replace(' ', '_')}",
                        name=f"vlm_{label.replace(' ', '_')}",
                        position=last_xyz,
                        radius=0.03,
                        color=np.array([0, 1, 0]),
                    )
                    return last_xyz
            else:
                print(f"VLM found nothing for '{label}'")

        except Exception as e:
            print(f"VLM error: {e}")

        await asyncio.sleep(0.1)


# =============================================================================
#  PPO HELPERS
# =============================================================================

# Replace the build_obs function and run_rl_grasp in orchestration_franka1.py
# with these corrected versions


def build_obs(joint_pos, joint_vel, obj_pos_robot_frame, target_pose, last_action):
    """
    Build observation matching Isaac-Lift-Cube-Franka-v0 training env.

    Terms (in order):
    1. joint_pos_rel:              9 values  (7 arm + 2 fingers, relative to default)
    2. joint_vel_rel:              9 values  (7 arm + 2 fingers, relative to default=0)
    3. object_position_in_robot_root_frame: 3 values
    4. target_object_position (generated_commands): 7 values (3 pos + 4 quat)
    5. last_action:                8 values
    Total: 36 values
    """
    default_j_pos = np.array(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741, 0.04, 0.04], dtype=np.float32
    )
    default_j_vel = np.zeros(9, dtype=np.float32)

    joint_pos_rel = (joint_pos - default_j_pos).astype(np.float32)
    joint_vel_rel = (joint_vel - default_j_vel).astype(np.float32)

    obs = np.concatenate(
        [
            joint_pos_rel,  # 9
            joint_vel_rel,  # 9
            obj_pos_robot_frame,  # 3
            target_pose,  # 7 (3 pos + 4 quat)
            last_action,  # 8
        ]
    ).astype(np.float32)

    assert obs.shape[0] == 36, f"Obs shape mismatch: {obs.shape[0]} != 36"
    return obs.reshape(1, -1)


async def run_rl_grasp(franka_art, robot_base_pos, cube_world_xyz):
    print(f"\n{'=' * 20} PPO: Grasping cube {'=' * 20}")

    # Load VLM default joint positions — same as training reset pose
    vlm_path = os.path.expanduser(
        "~/Documents/PFG/Scripts/Control/rl_start_near_cube_v2.json"
    )
    with open(vlm_path) as f:
        vlm_data = json.load(f)
    default_j_pos = np.array(vlm_data["joint_positions"], dtype=np.float32)

    # Cube in robot frame — NEGATED to match training convention
    cube_robot_frame = cube_world_xyz - robot_base_pos
    cube_train = np.array(
        [-cube_robot_frame[0], -cube_robot_frame[1], cube_robot_frame[2]],
        dtype=np.float32,
    )

    # Target pose — same negation, height above cube
    target_pose = np.array(
        [cube_train[0], cube_train[1], 0.35, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    print(f"cube_train (negated): {cube_train}")
    print(f"target_pose: {target_pose}")

    model = PPO.load(MODEL_PATH, device="cpu")
    last_action = np.zeros(8, dtype=np.float32)

    for step in range(MAX_RL_STEPS):
        j_pos = np.array(franka_art.get_joint_positions()).flatten().astype(np.float32)
        j_vel = np.array(franka_art.get_joint_velocities()).flatten().astype(np.float32)

        # joint_pos_rel relative to VLM default — matches training
        joint_pos_rel = (j_pos - default_j_pos).astype(np.float32)
        joint_vel_rel = j_vel.astype(np.float32)

        obs = (
            np.concatenate(
                [
                    joint_pos_rel,  # 9
                    joint_vel_rel,  # 9
                    cube_train,  # 3
                    target_pose,  # 7
                    last_action,  # 8
                ]
            )
            .astype(np.float32)
            .reshape(1, -1)
        )

        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action).flatten().astype(np.float32)
        last_action = action

        arm_targets = default_j_pos[:7] + (action[:7] * 0.5)
        gripper_val = 0.0 if action[7] > 0 else 0.04
        full_targets = np.concatenate([arm_targets, [gripper_val, gripper_val]])
        franka_art.apply_action(ArticulationAction(joint_positions=full_targets))

        await rep.orchestrator.step_async()

        ee_p_world, _ = get_world_pose(f"{ROBOT_PATH}/panda_hand")
        if (
            ee_p_world[2] > (cube_world_xyz[2] + LIFT_HEIGHT_THRESHOLD)
            and action[7] > MIN_GRIPPER_ACTION
            and step > 100
        ):
            print(f"✓ LIFTED at step {step}! Hand: {ee_p_world[2]:.3f}m")
            return True

    print("✗ Max steps reached")
    return False


# =============================================================================
#  MAIN ORCHESTRATION
# =============================================================================


async def run():
    print("\n" + "=" * 60)
    print("  FRANKA_1 FULL PIPELINE: VLM → RL GRASP → PLACE ON JETBOT")
    print("=" * 60)

    # ── Setup — no world reset ─────────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()

    rgb_annot, depth_annot = setup_camera()
    # =============================================================================
    #  MAIN ORCHESTRATION
    # =============================================================================
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid():
        print("ERROR: Camera_01 not found!")
        return

    robot_base_pos, _ = get_world_pose(ROBOT_PATH)
    print(f"Robot base: {robot_base_pos}")

    # ── PHASE 1: VLM — Find cube ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: VLM cube detection")
    print("=" * 60)
    cube_world_xyz = await vlm_detect(
        rgb_annot,
        depth_annot,
        camera_prim,
        URL,
        INSTRUCTION,
        label=INSTRUCTION,
        use_crop=True,  # use working script crop
    )

    # ── PHASE 2: Controller — Pre-grasp position ───────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: Moving to pre-grasp position")
    print("=" * 60)
    pre_grasp = [cube_world_xyz[0], cube_world_xyz[1], 0.15]
    print(f"Pre-grasp target: {pre_grasp}")
    await execute_movement(pre_grasp)
    for _ in range(80):
        await rep.orchestrator.step_async()

    # ── PHASE 3: PPO — Lift and grasp ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: RL grasp and lift")
    print("=" * 60)
    franka_art = Articulation(ROBOT_PATH)
    franka_art.initialize()

    grasped = await run_rl_grasp(franka_art, robot_base_pos, cube_world_xyz)

    if not grasped:
        print("Pipeline failed at grasp phase — aborting")
        return

    # ── PHASE 4: VLM — Find Jetbot ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4: VLM Jetbot detection")
    print("=" * 60)
    jetbot_xyz = await vlm_detect(
        rgb_annot,
        depth_annot,
        camera_prim,
        URL_JETBOT,
        "small green wheeled robot vehicle",
        label="jetbot",
        use_crop=False,  # full image for Jetbot
    )

    # ── PHASE 5: Controller — Move to Jetbot with gripper closed ──────────
    print("\n" + "=" * 60)
    print("  PHASE 5: Moving to Jetbot (gripper closed)")
    print("=" * 60)

    # Move up to safe transit height first
    transit_pos = [cube_world_xyz[0], cube_world_xyz[1], TRANSIT_HEIGHT]
    print(f"Lifting to transit height: {transit_pos}")
    await execute_movement(transit_pos, keep_gripper_closed=True)

    # Move above Jetbot
    place_pos = [jetbot_xyz[0], jetbot_xyz[1], TRANSIT_HEIGHT]
    print(f"Moving above Jetbot: {place_pos}")
    await execute_movement(place_pos, keep_gripper_closed=True)

    # Lower onto Jetbot platform
    lower_pos = [jetbot_xyz[0], jetbot_xyz[1], 0.15]
    print(f"Lowering onto Jetbot: {lower_pos}")
    await execute_movement(lower_pos, keep_gripper_closed=True)

    # ── PHASE 6: Open gripper — place cube on Jetbot ──────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 6: Releasing cube onto Jetbot")
    print("=" * 60)
    manager = FrankaControl(ROBOT_PATH)
    manager.open_gripper()
    for _ in range(30):
        await rep.orchestrator.step_async()
    print("✓ Cube released onto Jetbot!")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — Cube placed on Jetbot!")
    print("=" * 60)


asyncio.ensure_future(run())
