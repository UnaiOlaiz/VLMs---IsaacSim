# Script 2: Run VLM standalone (outside Isaac Sim, in terminal)
# Run this AFTER save_camera_data.py and WITH BentoML running on GPU
# Usage: python run_vlm_standalone.py
# Optionally: python run_vlm_standalone.py --instruction "green cube"

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import json
import os
import argparse
import re

# --- Config ---
SAVE_DIR = os.path.expanduser("~/Documents/PFG/Scripts/Control/camera_data")
JSON_OUT = os.path.expanduser("~/Documents/PFG/Scripts/Control/rl_first_franka.json")
URL = "http://127.0.0.1:8000/ground"

# Camera constants
F_PIXEL = (18.14 * 1280) / 20.955
CX, CY = 640, 360
RESOLUTION = (1280, 720)


def get_prediction(instruction, image_path, timeout=30):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {"instruction": instruction, "image_b64": img_str}
    print(f"Sending image to VLM: {image_path}")
    response = requests.post(URL, json=payload, timeout=timeout)
    return response.json()


def get_3d_target_calibrated(u, v, depth_map, cam_matrix):
    z_depth = depth_map[v, u]
    if z_depth == 0 or np.isnan(z_depth) or np.isinf(z_depth):
        return None

    x_cam = (u - CX) * z_depth / F_PIXEL
    y_cam = (v - CY) * z_depth / F_PIXEL
    z_cam = -z_depth

    target_pos_local = np.array([x_cam, y_cam, z_cam, 1.0])
    target_pos_world = np.dot(cam_matrix, target_pos_local)
    return target_pos_world[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default="red cube")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Number of VLM calls to make for stability")
    args = parser.parse_args()

    instruction = args.instruction
    print(f"Looking for: '{instruction}'")

    # Load saved data
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")
    depth_path = os.path.join(SAVE_DIR, "depth_data.npy")
    crop_path = os.path.join(SAVE_DIR, "camera01_crop.png")

    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.json not found at {SAVE_DIR}")
        print("Run save_camera_data.py inside Isaac Sim first!")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    cam_matrix = np.array(metadata["cam_matrix"])
    robot_pos = np.array(metadata["robot_pos"])
    crop = metadata["crop"]
    y_start = crop["y_start"]
    x_start = crop["x_start"]
    crop_h = crop["y_end"] - crop["y_start"]
    crop_w = crop["x_end"] - crop["x_start"]

    depth_data = np.load(depth_path)

    # Run VLM multiple times for stability
    detections = []
    for i in range(args.attempts):
        print(f"\nAttempt {i+1}/{args.attempts}...")
        try:
            result = get_prediction(instruction, crop_path, timeout=300)
            print(f"VLM result: {result}")

            if result and result.get("target") and result["target"].get("found"):
                raw_bbox = result["target"]["bbox_xyxy"]

                # Map bbox from crop space → full image pixel coords
                v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                u_final = int(np.clip(u_crop + x_start, 0, RESOLUTION[0] - 1))
                v_final = int(np.clip(v_crop + y_start, 0, RESOLUTION[1] - 1))

                xyz = get_3d_target_calibrated(u_final, v_final, depth_data, cam_matrix)
                if xyz is None:
                    print("Invalid depth at detection point, skipping")
                    continue

                print(f"World XYZ: {xyz}")

                # Simple filter
                if xyz[2] < -0.1 or xyz[0] > 2.0:
                    print(f"Rejected by filter: {xyz}")
                    continue

                detections.append(xyz)
                print(f"Valid detection: {xyz}")
            else:
                print("VLM found nothing")

        except Exception as e:
            print(f"Error: {e}")

    if not detections:
        print("\nERROR: No valid detections found. Check camera view and BentoML.")
        return

    # Use median for robustness
    detections = np.array(detections)
    final_pos = np.median(detections, axis=0)
    print(f"\n--- FINAL TARGET: {final_pos} ---")
    print(f"(from {len(detections)} valid detections)")

    # Save to JSON
    relative_cube_pos = final_pos - robot_pos
    data_save = {
        "target_pos": final_pos.tolist(),
        "robot_pos": robot_pos.tolist(),
        "cube_world_pos": relative_cube_pos.tolist(),
    }

    with open(JSON_OUT, "w") as f:
        json.dump(data_save, f, indent=4)

    print(f"\n--- Coordinates saved to {JSON_OUT} ---")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("  1. Kill BentoML: lsof -i :8000  →  kill -9 <PID>")
    print("  2. Restart Isaac Sim")
    print("  3. Run movement_franka1.py in the Script Editor")
    print("=" * 60)


if __name__ == "__main__":
    main()