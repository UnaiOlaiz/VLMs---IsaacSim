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

        self.offset_index = 0
        self.search_offsets = [
            [0, 0], [0.015, 0], [-0.015, 0], [0, 0.015], [0, -0.015]
        ]

    def move_to_target(self, target_pos):
        # Apply the current offset to the VLM prediction
        off = self.search_offsets[self.offset_index]
        current_target = np.array(target_pos) + np.array([off[0], off[1], 0])
        
        self.controller.reset() # so it starts from scratch and opens its arms

        actions = self.controller.forward(
            picking_position=current_target,
            placing_position=np.array([0.4, 0.4, 0.05]),
            current_joint_positions=self.robot.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.01])
        )
        
        self.robot.apply_action(actions)
        
        # Check if the controller finished its sequence
        if self.controller.is_done():
            # SUCCESS CHECK: Is there something in the gripper?
            # If the fingers are completely closed (distance ~ 0), we missed.
            gripper_pos = self.robot.gripper.get_joint_positions()
            # Franka gripper fingers are at indices 0 and 1, usually 0.04m each when open
            if np.sum(gripper_pos) < 0.05: 
                print(f"Gripper empty (pos: {np.sum(gripper_pos)}). Missed!")
                self.offset_index += 1
                self.controller.restet()
                return False
            else:
                print(f"Cube captured! At position: {np.sum(gripper_pos)}")
                return True
async def execute_movement(final_coords):
    timeline = omni.timeline.get_timeline_interface()

    # 1. Corrección de coordenadas (Z segura para el descenso)
    target_pos = np.array(final_coords)
    target_pos[2] = 0.025 

    print(f"PFG: Target corregido para descenso: {target_pos}")

    if not timeline.is_playing():
        timeline.play()
        await asyncio.sleep(1.0) 

    # Inicializamos el mundo para poder llamar a world.step()
    from omni.isaac.core import World
    world = World.instance() # Usamos la instancia existente del mundo
    
    try:
        manager = FrankaControl()
        print(f"PFG: Franka vinculado en /World/Franka_Robot")
        
        # 2. Forzar apertura de pinza antes de mover el brazo
        print("PFG: Abriendo pinza para aproximación...")
        manager.robot.gripper.open()
        
        # Esperamos físicamente a que la pinza termine de abrirse
        for _ in range(30):
            world.step(render=True)
            await asyncio.sleep(0.01)

        print(f"PFG: Iniciando trayectoria hacia {target_pos}...")
        
        done = False
        timeout_counter = 0
        while not done and timeout_counter < 1000:
            # 3. CRÍTICO: Avanzamos la física del simulador
            world.step(render=True)
            
            try:
                # Usamos target_pos, NO final_coords
                done = manager.move_to_target(target_pos)
            except Exception as e:
                print(f"Error en paso de trayectoria: {e}")
                break
            
            # Pequeña pausa para no saturar el hilo de asyncio
            await asyncio.sleep(0.001)
            timeout_counter += 1

        if done:
            print("PFG: ¡Movimiento completado con éxito!")
        else:
            print("PFG: El movimiento se detuvo por tiempo límite o error de IK.")

    except Exception as e:
        print(f"Error crítico en la ejecución: {e}")
