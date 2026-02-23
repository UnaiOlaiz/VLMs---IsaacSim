# Dependencies
import numpy as np
import asyncio
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import get_current_stage
import omni.usd
from pxr import UsdPhysics
from omni.isaac.core.utils.viewports import set_camera_view
import omni.replicator.core as rep
import omni.timeline
from omni.isaac.core.world import World

class FrankaControl:
    def __init__(self, prim_path="/World/Franka_Robot", name="franka_pfg"):
        if not is_prim_path_valid(prim_path):
            prim_path="/Franka_Robot"
            
        print("PFG: Inicializando FrankaControl...")
        self.robot = Franka(prim_path=prim_path, name=name)
        # Forzamos la inicialización interna del robot
        self.robot.initialize() 
        
        # Ahora dof_names ya no será None
        dof_names = self.robot.dof_names
        if dof_names is None:
            print("ERROR: El robot no ha cargado sus articulaciones. ¿Está el Play activado?")
            return

        self.robot.post_reset()

        # En la v5.1, para el PickPlaceController estándar, 
        # a veces es mejor dejar que él gestione los índices si pasas el robot
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )

    def move_to_target(self, target_pos):
        # Aseguramos que el robot esté en el mundo antes de pedir posición
        current_joint_positions = self.robot.get_joint_positions()
        
        actions = self.controller.forward(
            picking_position=np.array(target_pos),
            placing_position=np.array([0.5, 0.0, 0.05]),
            current_joint_positions=current_joint_positions
        )

        self.robot.apply_action(actions)
        return self.controller.is_done()


async def execute_movement(final_coords):
    from omni.isaac.core.world import World
    from omni.isaac.core.utils.stage import get_current_stage
    import omni.timeline
    import asyncio

    stage = get_current_stage()
    timeline = omni.timeline.get_timeline_interface()

    scene_path = "/PhysicsScene"
    if not stage.GetPrimAtPath(scene_path).IsValid():
        from pxr import UsdPhysics
        UsdPhysics.Scene.Define(stage, scene_path)
        print("PFG: PhysicsScene creada.")

    if not timeline.is_playing():
        timeline.play()
        # Aumentamos a 2 segundos: Isaac Sim 5.1 es pesado cargando articulaciones
        await asyncio.sleep(2.0)

    world = World()

    # --- CAMBIO CRÍTICO: Reintentos para evitar el error 'link_names' ---
    manager = None
    for i in range(5):
        try:
            manager = FrankaControl()
            print(f"PFG: Franka vinculado con éxito en el intento {i+1}")
            break
        except Exception as e:
            print(f"PFG: Esperando al robot (intento {i+1}/5)...")
            await asyncio.sleep(1.5)

    if manager is None:
        print("PFG ERROR: No se pudo inicializar el Franka. Revisa el Prim Path.")
        return

    print(f"Moving Franka to: {final_coords}")

    done = False
    while not done:
        await asyncio.sleep(0.01)
        try:
            done = manager.move_to_target(final_coords)
        except Exception as e:
            # Si el backend da error momentáneo, seguimos intentando
            continue

    print("Movement Completed!")
