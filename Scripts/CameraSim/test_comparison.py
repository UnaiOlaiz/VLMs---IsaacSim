from omni.isaac.core.utils.xforms import get_world_pose
for name in ["/World/Cubes/Red_Cube", "/World/Cubes/Green_Cube", "/World/Cubes/Blue_Cube"]:
    try:
        pos, _ = get_world_pose(name)
        print(f"{name}: {pos}")
    except:
        print(f"{name}: not found")

for name in ["/World/Palettes/Red_Palette", "/World/Palettes/Blue_Palette", "/World/Palettes/Black_Palette"]:
    try: 
        pos, _ = get_world_pose(name)
        print(f"{name}: {pos}")
    except:
        print(f"{name}: not found...")