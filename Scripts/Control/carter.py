import numpy as np
import asyncio
from isaacsim.core.api.world import World
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationAction


async def navegar_carter_autonomo():
    world = World.instance()
    if world is None:
        world = World()

    # Cambiamos el path al que Isaac Sim realmente espera si el robot fue arrastrado
    # A veces es /World/carter_v2_chassis/chassis_link
    robot_path = "/World/carter_v2_chassis"

    # Intentamos obtener el objeto. Si falla, lo añadimos de nuevo con limpieza
    carter = world.scene.get_object("carter_robot")
    if carter:
        world.scene.remove_object("carter_robot")

    carter = world.scene.add(Robot(prim_path=robot_path, name="carter_robot"))

    # IMPORTANTE: En lugar de reset_async (que da error de is_homogeneous),
    # simplemente inicializamos y damos al Play si no lo está.
    if not world.is_playing():
        await world.initialize_simulation_context_async()
        world.play()

    # Esperamos un segundo a que PhysX registre el Articulation Root
    await asyncio.sleep(1.0)

    try:
        carter.initialize()
    except Exception as e:
        print(
            f"Error inicializando: {e}. ¿Has añadido el Articulation Root manualmente?"
        )
        return

    # Comprobamos si detecta los joints
    print(f"Joints detectados: {carter.dof_names}")
    if len(carter.dof_names) == 0:
        print(
            "ERROR: El robot no tiene articulaciones físicas. Revisa el Articulation Root."
        )
        return

    l_idx = carter.get_dof_index("left_wheel_joint")
    r_idx = carter.get_dof_index("right_wheel_joint")

    target_pos = np.array([2.0, 0.0])

    for i in range(1000):
        current_pos, _ = carter.get_world_pose()
        diff = target_pos - current_pos[:2]
        dist = np.linalg.norm(diff)

        velocities = np.zeros(carter.num_dof)
        if dist > 0.2:
            velocities[l_idx] = 5.0
            velocities[r_idx] = 5.0
        else:
            print("Llegamos.")
            break

        carter.get_articulation_controller().apply_action(
            ArticulationAction(joint_velocities=velocities)
        )
        await world.step_async()


asyncio.ensure_future(navegar_carter_autonomo())
