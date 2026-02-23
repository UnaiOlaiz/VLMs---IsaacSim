import numpy as np
import asyncio
# Usamos las rutas preferidas de la nueva API
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.world import World
from isaacsim.core.utils.prims import is_prim_path_valid
# ELIMINADA la importación de get_physics_context de omni.isaac.core.utils.physics
from isaacsim.core.api.simulation_context import SimulationContext

class FrankaControl:
    def __init__(self, prim_path="/World/Franka_Robot", name="franka_pfg"):
        if not is_prim_path_valid(prim_path):
            print(f"ERROR: No se encuentra el robot en {prim_path}.")
            prim_path = "/Franka_Robot"
            
        self.robot = Franka(prim_path=prim_path, name=name)
        self.robot.initialize() 
        self.robot.gripper.open() 

        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )

    def move_to_target(self, target_pos):
        current_joint_positions = self.robot.get_joint_positions()
        
        # Ajuste de seguridad para la Z detectada
        actual_target = np.array(target_pos)
        if actual_target[2] < 0.01:
            actual_target[2] = 0.015 

        actions = self.controller.forward(
            picking_position=actual_target,
            placing_position=np.array([0.4, 0.4, 0.2]),
            current_joint_positions=current_joint_positions,
            end_effector_offset=np.array([0, 0, 0.02])
        )
        self.robot.apply_action(actions)
        return self.controller.is_done()

import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.api.simulation_context import SimulationContext

async def execute_movement(final_coords):
    # 1. Forzamos la limpieza de cualquier instancia previa "fantasma"
    if World.instance():
        World.instance().clear_instance()

    # 2. Inicializamos el SimulationContext ANTES que el World
    # Esto vincula el motor a la ruta exacta que vemos en tus capturas
    sim_context = SimulationContext(
        physics_prim_path="/World/PhysicsScene",
        stage_units_in_meters=1.0
    )

    # 3. Ahora instanciamos el World
    world = World()

    # Verificación de seguridad
    if world.get_physics_context() is None:
        print("PFG ERROR: El motor de física sigue sin responder.")
        return

    print("PFG: Motor de física vinculado correctamente.")
    await world.reset_async()

    # 4. Instanciamos el controlador del robot
    manager = FrankaControl()

    # 5. Corrección de seguridad para el objetivo (Z positiva siempre)
    # Tus logs mostraban Z negativas que causarían colisión con el GroundPlane
    safe_target = np.array(final_coords)
    if safe_target[2] < 0.01:
        print(f"PFG: Ajustando altura de seguridad de {safe_target[2]} a 0.015")
        safe_target[2] = 0.015

    # 6. Bucle de ejecución con renderizado activo
    for _ in range(60):
        world.step(render=True)

    done = False
    while not done:
        world.step(render=True)
        try:
            done = manager.move_to_target(safe_target)
        except Exception as e:
            continue

    print("PFG: ¡Objetivo alcanzado con éxito!")
