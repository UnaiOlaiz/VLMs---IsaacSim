import numpy as np
import asyncio
from isaacsim.core.api.world import World
from isaacsim.robot.manipulators.examples.franka import Franka

async def spawn_and_move():
    if World.instance():
        World.instance().clear_instance()

    world = World(physics_prim_path="/World/PhysicsScene", stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    franka = world.scene.add(
        Franka(prim_path="/World/Franka", name="franka", position=np.array([0, 0, 0]))
    )

    await world.reset_async()
    
    # Intentamos cargar el controlador de la extension directamente
    from isaacsim.robot.manipulators.controllers import PickAndPlaceController
    
    target_pos = np.array([0.4, 0.2, 0.4])
    target_orient = np.array([0, 1, 0, 0])

    # RMPFlow suele estar bajo omni.isaac.motion_generation
    from omni.isaac.motion_generation import RmpFlow
    from omni.isaac.motion_generation.interface_config_loader import load_config
    
    # Metodo simplificado usando la API de Franka para obtener su propio controlador
    # Si el import anterior fallo, este es el mas robusto:
    for i in range(100):
        if franka.gripper is not None:
            # Aplicamos una accion simple de articulacion para testear comunicacion
            franka.get_articulation_controller().apply_action(
                franka.motion_policy.get_next_articulation_action(target_pos, target_orient)
            )
        await world.step_async()
    
    print("OK: Ejecutado.")

asyncio.ensure_future(spawn_and_move())
