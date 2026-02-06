# Dependencies needed
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

# URL where the BentoML service is running
URL = "http://127.0.0.1:8000/ground"

# FIXED RESOLUTION OF THE ENVIRONMENT INSIDE ISAACSIM
RESOLUTION = (1280, 720)

# the instruction will also be declared as variable if needed to be adapted
INSTRUCTION = "red cube" # in this case, what to be found

def get_prediction(instruction, rgb_image):
    '''
    Function that will perform the prediction given the instruction+environment image pair.
    '''
    rgb_clean = np.ascontiguousarray(rgb_image[..., :3], dtype=np.uint8)
    img = Image.fromarray(rgb_clean)
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "instruction": instruction,
        "image_b64": img_str  # Changed from image_64
    }

    response = requests.post(URL, json=payload, timeout=10)
    return response.json()

def get_3d_target(bbox, depth_map, cam_view_matrix, cam_proj_matrix):
    '''
    Docstring for get_3d_target
    
    :param bbox: Description
    :param depth_map: Description
    :param cam_view_matrix: Description
    :param cam_proj_matrix: Description
    '''
    # bbox is [ymin, xmin, ymax, xmax]
    ymin, xmin, ymax, xmax = bbox

    # Pixel normalization for the fixed resolution (1280x720)
    u = int(((xmin+xmax) / 2) * 1280 / 1000)
    v = int(((ymin+ymax) / 2) * 720 / 1000)
    # We fix the values into the given ranges to fall into correct values
    u = np.clip(u, 0, 1279)
    v = np.clip(v, 0, 719)

    z_depth = depth_map[v, u]

    f_len = 900.0 # focal length of the fixed camera
    cx, cy = 640, 360

    x_cam = (u - cx) * z_depth / f_len
    y_cam = (v - cy) * z_depth / f_len
    z_cam = -z_depth

    target_pos_local = np.array([x_cam, y_cam, z_cam, 1.0])
    target_pos_world = np.dot(cam_view_matrix, target_pos_local)

    return target_pos_world[:3]

def get_3d_target_direct(u, v, depth_map, cam_view_matrix):
    '''
    Calculates 3D World coordinates from direct pixel coordinates and depth.
    '''
    # 1. Clip coordinates to image boundaries
    u = np.clip(u, 0, 1279)
    v = np.clip(v, 0, 719)

    # 2. Get depth (Distance to Camera in meters)
    z_depth = depth_map[v, u]

    # 3. Calculate Focal Length in pixels
    # Based on your 18.14mm focal length and assuming a standard 36mm sensor width
    f_pixel = (18.14 * 1280) / 36.0 
    cx, cy = 640, 360 # Center of 1280x720 frame

    # 4. Unproject from 2D Pixels to 3D Camera Space
    # Isaac Sim cameras usually look down the -Z axis
    x_cam = (u - cx) * z_depth / f_pixel
    y_cam = (v - cy) * z_depth / f_pixel
    z_cam = -z_depth 

    # 5. Transform from Camera Space to World Space
    target_pos_local = np.array([x_cam, y_cam, z_cam, 1.0])
    target_pos_world = np.dot(cam_view_matrix, target_pos_local)

    return target_pos_world[:3]



async def main_vision():
    '''
    This function will serve as a bridge, it will capture what the camera sees in the environment and send it to the service
    '''
    print("-" * 50 + "INITIALIZING RENDERER" + "-" * 50)

    try:
        rep.orchestrator.stop()
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        if rgb_annot:
            rgb_annot.detach()
    except Exception as e:
        print(f"Cleanup skipped (normal for first run): {e}")

    rp = rep.create.render_product("/World/Camera_01", resolution=RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath("/World/Camera_01")
    if not camera_prim.IsValid():
        print("-" * 50 + "CAMERA NOT FOUND" + "-" * 50)
        return
    
    loop = asyncio.get_event_loop()

    # parameters for coordinate findings
    consecutive_detections = 0
    stability_count = 3 # we will stop the loop when finding 3 same coordinates in a row
    last_stable_xyz = np.array([0.0, 0.0, 0.0])

    print("STARTING COORDINATE SEARCHING!")

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        y_start, y_end = 200, 600
        x_start, x_end = 400, 1000
        cropped_img = rgb_data[y_start:y_end, x_start:x_end]

        if cropped_img is not None and cropped_img.size > 0:
            try:
                result = await loop.run_in_executor(None, get_prediction, INSTRUCTION, cropped_img)
            
                if result and result.get("target") and result["target"].get("found"):
                    raw_bbox = result["target"]["bbox_xyxy"]
                
                    crop_h, crop_w = (y_end - y_start), (x_end - x_start)
                    v_crop = (raw_bbox[0] + raw_bbox[2]) / 2 * crop_h / 1000
                    u_crop = (raw_bbox[1] + raw_bbox[3]) / 2 * crop_w / 1000
                    u_final = int(u_crop + x_start)
                    v_final = int(v_crop + y_start)

                    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                    cam_matrix = np.array(world_transform).reshape(4, 4).T
                    current_xyz = get_3d_target_direct(u_final, v_final, depth_data, cam_matrix)

                    if current_xyz[2] < -0.1 or current_xyz[0] > 2.0:
                        print(f"Skipping hallucination: {current_xyz}")
                        continue

                    distance = np.linalg.norm(current_xyz - last_stable_xyz)
                    if distance < 0.05: 
                        consecutive_detections += 1
                        print(f"Detections stable: {consecutive_detections}/{stability_count}")
                    else:
                        consecutive_detections = 1
                        last_stable_xyz = current_xyz
                        print(f"VLM jitter detected. Resetting stability at: {current_xyz}")

                    if consecutive_detections >= stability_count:
                        print(f"--- FINAL TARGET LOCKED: {last_stable_xyz} ---")

                        VisualSphere(
                            prim_path="/World/green_target",
                            name="green_target",
                            position=last_stable_xyz,
                            radius=.02,
                            color=np.array([0,1,0])
                        )

                        return {
                            "world_xyz": last_stable_xyz,
                            "raw_json": result
                        }

            except Exception as e:
                print(f"Connection error: {e}")
        await asyncio.sleep(.1)

async def run():
    target_data = await main_vision()
    target_pos = target_data["world_xyz"]

    # the movement script will be called here

asyncio.ensure_future(run())
