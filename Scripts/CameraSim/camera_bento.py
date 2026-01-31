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
    y1, x1, y2, x2 = bbox
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    
    # Get depth (Z) in meters
    z_depth = depth_map[v, u]
    
    # Simple Unprojection logic (using center of bbox)
    # This assumes a 1280x720 frame
    focal_length = 900.0 
    cx, cy = 640, 360
    
    x_cam = (u - cx) * z_depth / focal_length
    y_cam = (v - cy) * z_depth / focal_length
    z_cam = z_depth
    
    # Multiply by camera's world matrix to get XYZ in the simulation
    # (Isaac Sim uses a 4x4 transform matrix for this)
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
    final_target_xyz = None

    print("STARTING COORDINATE SEARCHING!")

    while True:
        await rep.orchestrator.step_async()
        rgb_data = rgb_annot.get_data()
        depth_data = depth_annot.get_data()

        if rgb_data is not None and rgb_data.size > 0:
            try:
                result = await loop.run_in_executor(None, get_prediction, INSTRUCTION, rgb_data)
                if result and result.get("target"):
                    consecutive_detections += 1
                    world_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
                    cam_matrix = np.array(world_transform).reshape(4, 4).T
                    final_target_xyz = get_3d_target(result["target"]["bbox_xyxy"], depth_data, cam_matrix, None)

                    if consecutive_detections >= stability_count:
                        print(f"FINAL TARGET FOUND: {final_target_xyz}")
                        return {
                            "world_xyz": final_target_xyz,
                            "raw_json": result
                        }

                    target_3d = get_3d_target(result["target"]["bbox_xyxy"], depth_data, cam_matrix, None)
                    print(f"VLM found {INSTRUCTION} at World XYZ: {target_3d}")
            except Exception as e:
                print(f"Connection error: {e}")
        await asyncio.sleep(.1)

async def run():
    target_pos = await main_vision()

asyncio.ensure_future(run())
