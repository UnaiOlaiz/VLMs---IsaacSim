import omni.replicator.core as rep
import numpy as np
import requests
import base64
import io
from PIL import Image

# 1. SET YOUR CONSTANTS (Match your viewport!)
URL = "http://127.0.0.1:8000/ground"
CAMERA_PATH = "/World/Camera_01"
RES = (1280, 720) 

def rgb_to_png_b64(rgb_array):
    # This fixes the 'SystemError: tile' by making memory contiguous
    # and stripping the alpha channel
    rgb_clean = np.ascontiguousarray(rgb_array[..., :3], dtype=np.uint8)
    pil_img = Image.fromarray(rgb_clean)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

async def run_vlm_task():
    print("🚀 Capturing frame for VLM...")
    
    # 2. Setup Render Product
    rp = rep.create.render_product(CAMERA_PATH, resolution=RES)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    
    # 3. Step the simulation to fill the buffer
    await rep.orchestrator.step_async()
    
    img_data = rgb.get_data()
    
    # 4. Check for the IndexError/Null data before proceeding
    if img_data is None or img_data.size == 0:
        print("❌ Error: Received empty data from camera.")
        return

    # 5. Send to your live server
    payload = {
        "instruction": "Locate the red cube on the floor. Ignore the white robot arm. Provide the bounding box for only the red cube.",
        "image_b64": rgb_to_png_b64(img_data)
    }
    
    print("📡 Sending to Server at port 8000...")
    response = requests.post(URL, json=payload)
    print(f"✅ Server Response: {response.json()}")

# Run the task
import asyncio
asyncio.ensure_future(run_vlm_task())