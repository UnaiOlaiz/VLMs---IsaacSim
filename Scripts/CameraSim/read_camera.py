import omni.replicator.core as rep
import omni.kit.app
import numpy as np
from omni.kit.async_engine import run_coroutine

camera_path = "/World/Camera_01"

async def main():
    print("▶ Starting camera capture...") # Using print for the Script Editor console

    # Ensure the simulation is playing, otherwise annotators won't fill
    if not omni.timeline.get_timeline_interface().is_playing():
        print("Warning: Simulation is not playing. Starting timeline...")
        omni.timeline.get_timeline_interface().play()

    # Create render product
    rp = rep.create.render_product(camera_path, resolution=(640, 480))

    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth.attach([rp])

    # Wait for the renderer to process frames
    app = omni.kit.app.get_app()
    for i in range(20): # Increased wait time for buffer safety
        await app.next_update_async()

    img = rgb.get_data()
    d = depth.get_data()

    # Output to Script Editor Console
    print(f"--- CAPTURE RESULTS ---")
    print(f"RGB shape: {getattr(img, 'shape', None)}")
    print(f"Depth shape: {getattr(d, 'shape', None)}")

    if hasattr(d, "size") and d.size > 0:
        # Filter out infinities for a clean min/max
        finite_depth = d[np.isfinite(d)]
        if finite_depth.size > 0:
            print(f"Depth min/max: {finite_depth.min():.2f}m / {finite_depth.max():.2f}m")
    else:
        print("❌ ERROR: Depth data is empty. Check if Camera_01 exists in the Stage.")

# Schedule the coroutine
run_coroutine(main())
print("✅ Task successfully scheduled in Script Editor")
