# Dependencies
import numpy as np
import asyncio
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
import omni.replicator.core as rep

class FrankaControl:
    def __init__(self, prim_path="/World/Franka_Robot", name="franka_pfg"):
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
    # 1. Creamos el manager
    manager = FrankaControl()
    
    print(f"Moving Franka to coordinates: {final_coords}")
    
    # 2. Damos 10 frames de margen para que la física se asiente
    for _ in range(10):
        await rep.orchestrator.step_async()
    
    done = False
    while not done:
        await rep.orchestrator.step_async()
        done = manager.move_to_target(final_coords)
    
    print("Movement Completed!")
