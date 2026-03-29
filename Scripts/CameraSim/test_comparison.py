from omni.isaac.core.utils.xforms import get_world_pose
for name in ["/World/Cubes/Red_Cube", "/World/Cubes/Green_Cube", "/World/Cubes/Blue_Cube", 
             "/World/Palettes/Red_Palette", "/World/Palettes/Blue_Palette", "/World/Palettes/Black_Palette",
             "/World/Franka_Robot", "/World/Franka_Robot_01"]:
    try:
        pos, _ = get_world_pose(name)
        print(f"{name}: {pos}")
    except:
        print(f"{name}: not found")

