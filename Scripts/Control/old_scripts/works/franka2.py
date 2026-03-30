import numpy as np
import asyncio
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka
from omni.isaac.core.world import World
import omni.timeline
from omni.isaac.core.utils.prims import is_prim_path_valid

class FrankaControl:
    # Ajustamos la ruta a la que se ve en tu Stage dentro de World
    def __init__(self, prim_path="/World/Franka_Robot", name="franka_pfg"):
        if not is_prim_path_valid(prim_path):
            print(f"ERROR: No hay nada en {prim_path}. Verifica el nombre en el Stage.")
            # Si falla, intentamos la ruta raíz por si acaso
            prim_path = "/Franka_Robot"
            
        print(f"PFG: Vinculando robot en {prim_path}...")
        self.robot = Franka(prim_path=prim_path, name=name)
        self.robot.initialize() 

        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )

    def move_to_target(self, target_pos):
        current_joint_positions = self.robot.get_joint_positions()
        actions = self.controller.forward(
            picking_position=np.array(target_pos),
            placing_position=np.array([0.4, 0.4, 0.05]), 
            current_joint_positions=current_joint_positions
        )
        self.robot.apply_action(actions)
        return self.controller.is_done()

async def execute_movement(final_coords):
    timeline = omni.timeline.get_timeline_interface()
    
    # IMPORTANTE: Usamos la physicsScene de la raíz que se ve en tu imagen
    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(2.0) 

    world = World()
    
    try:
        manager = FrankaControl()
        print(f"PFG: Franka listo. Moviendo a {final_coords}")
        
        done = False
        while not done:
            await asyncio.sleep(0.01)
            try:
                done = manager.move_to_target(final_coords)
            except:
                continue
        print("PFG: ¡Movimiento completado!")
        
    except Exception as e:
        print(f"Error en la ejecución: {e}")
