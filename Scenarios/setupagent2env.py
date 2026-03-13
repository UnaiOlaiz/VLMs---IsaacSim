import asyncio
from isaacsim.core.api.world import World
import omni.client
import omni.usd

async def spawn_a2d_local():
    if World.instance():
        World.instance().clear_instance()

    world = World(physics_prim_path="/World/PhysicsScene", stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Scan common localhost roots for your file
    candidates = [
        "omniverse://localhost/NVIDIA/Assets/Robots/Agibot/A2D/configuration/A2D_physics.usd",
        "omniverse://localhost/Library/Robots/Agibot/A2D/configuration/A2D_physics.usd",
        "omniverse://localhost/Projects/Robots/Agibot/A2D/configuration/A2D_physics.usd",
        "omniverse://localhost/Robots/Agibot/A2D/configuration/A2D_physics.usd",
        "omniverse://localhost/Users/admin/Robots/Agibot/A2D/configuration/A2D_physics.usd",
    ]

    usd_path = None
    for path in candidates:
        result, _ = omni.client.stat(path)
        print(f"  {result} -> {path}")
        if str(result) == "Result.OK":
            usd_path = path
            break

    if usd_path is None:
        # List what's actually at the localhost root so we can find it
        print("\nScanning omniverse://localhost/ ...")
        result, entries = omni.client.list("omniverse://localhost/")
        for e in entries:
            print(f"  {e.relative_path}")
        return

    print(f"\nFound: {usd_path}")
    from omni.isaac.core.utils.stage import add_reference_to_stage
    prim = add_reference_to_stage(usd_path=usd_path, prim_path="/World/A2D")

    await world.initialize_async()
    await world.reset_async()
    await world.step_async()
    print("Done — check viewport.")

asyncio.ensure_future(spawn_a2d_local())
