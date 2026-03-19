# Final environment creation and setup file

# Dependencies
import omni.usd
import omni.kit.commands
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
from omni.isaac.core.utils.stage import add_reference_to_stage


# Paths used to load the assets for the environment
franka_usd = "/home/unaiolaizolaosa/isaac-sim-5.1.0/assets/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
jetbot_usd = "/home/unaiolaizolaosa/isaac-sim-5.1.0/assets/Assets/Isaac/5.1/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"


# ----------------------------
# Helper functions
# ----------------------------
def ensure_xform(stage, path):
    """Create an Xform prim if it does not already exist."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        omni.kit.commands.execute("CreatePrim", prim_path=path, prim_type="Xform")
        prim = stage.GetPrimAtPath(path)
    return prim


def add_prim(stage, path, usd_path, position, rotation_z_deg=0.0, scale=(1.0, 1.0, 1.0)):
    """Add a referenced USD asset and apply transform."""
    ensure_xform(stage, path)
    add_reference_to_stage(usd_path=usd_path, prim_path=path)

    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(rotation_z_deg)
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    return prim


def create_box(stage, path, position, size, color_rgb):
    """Create a colored cube."""
    omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Cube",
        prim_path=path,
    )

    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    )

    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color_rgb)])
    return prim


def create_nest(stage, path, position, color_rgb, size=(0.25, 0.25, 0.03)):
    """Create a nest from 5 cubes: floor + 4 walls."""
    ensure_xform(stage, path)

    cx, cy, cz = position
    w, d, h = size
    wall_t = 0.02
    wall_h = 0.06

    parts = {
        "floor":  ((cx, cy, cz),                        (w,      d,      h)),
        "wall_n": ((cx, cy + d / 2.0, cz + wall_h / 2.0), (w,      wall_t, wall_h)),
        "wall_s": ((cx, cy - d / 2.0, cz + wall_h / 2.0), (w,      wall_t, wall_h)),
        "wall_e": ((cx + w / 2.0, cy, cz + wall_h / 2.0), (wall_t, d,      wall_h)),
        "wall_w": ((cx - w / 2.0, cy, cz + wall_h / 2.0), (wall_t, d,      wall_h)),
    }

    for name, (pos, sz) in parts.items():
        p = f"{path}/{name}"
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Cube",
            prim_path=p,
        )
        prim = stage.GetPrimAtPath(p)

        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(sz[0] / 2.0, sz[1] / 2.0, sz[2] / 2.0)
        )

        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color_rgb)])
        UsdPhysics.CollisionAPI.Apply(prim)

    return stage.GetPrimAtPath(path)


def add_physics_scene(stage):
    """Create the physics scene with gravity."""
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        omni.kit.commands.execute(
            "AddPhysicsSceneCommand",
            stage=stage,
            path="/World/PhysicsScene",
        )

    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    scene = UsdPhysics.Scene(scene_prim)

    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    PhysxSchema.PhysxSceneAPI.Apply(scene_prim)


def add_floor(stage, path, position, scale=(3.0, 3.0, 0.05)):
    """Create a solid floor cube."""
    omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Cube",
        prim_path=path,
    )

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(scale[0] / 2.0, scale[1] / 2.0, scale[2] / 2.0)
    )

    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.01, 0.01, 0.01)])
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def add_wall(stage, path, position, scale):
    """Create a solid wall cube."""
    omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Cube",
        prim_path=path,
    )

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(scale[0] / 2.0, scale[1] / 2.0, scale[2] / 2.0)
    )

    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.65)])
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def set_rigid_body(prim, mass=0.1):
    """Apply rigid body, collision and mass to a prim."""
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    prim.GetAttribute("physics:mass").Set(mass)


# ----------------------------
# Scene parameters
# ----------------------------
colors = {
    "red":   (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue":  (0.0, 0.0, 1.0),
}

cube_sizes = (0.05, 0.05, 0.05)

# Slightly above the floor so physics can settle naturally.
# Floor top is at z = 0.0
cube_positions = {
    "red":   (-2.0, 1.2, 0.03),
    "green": (-2.0, 1.4, 0.03),
    "blue":  (-2.0, 1.0, 0.03),
}

# Nest floor thickness = 0.03 -> center at 0.015 sits on top of floor
nest_positions = {
    "red":   (2.0, 1.2, 0.015),
    "green": (2.0, 1.4, 0.015),
    "blue":  (2.0, 1.0, 0.015),
}


# ----------------------------
# Main builder
# ----------------------------
def build_scene():
    print("Starting environment build...")

    stage = omni.usd.get_context().get_stage()

    # Create base hierarchy
    ensure_xform(stage, "/World")
    ensure_xform(stage, "/World/Cubes")
    ensure_xform(stage, "/World/Nests")
    ensure_xform(stage, "/World/Walls")
    ensure_xform(stage, "/World/Lights")

    # Physics scene
    add_physics_scene(stage)

    # Floor
    # Thickness = 0.1, centered at z = -0.05 => top face at z = 0.0
    floor_prim = add_floor(stage, "/World/Floor", (0.0, 0.7, -0.05), scale=(10.0, 10.0, 0.1))

    # Optional surrounding walls
    # add_wall(stage, "/World/Walls/Back",  (0.0,  4.0, 0.5), (6.0, 0.2, 1.0))
    # add_wall(stage, "/World/Walls/Front", (0.0, -2.0, 0.5), (6.0, 0.2, 1.0))
    # add_wall(stage, "/World/Walls/Left",  (-3.0, 1.0, 0.5), (0.2, 6.0, 1.0))
    # add_wall(stage, "/World/Walls/Right", (3.0,  1.0, 0.5), (0.2, 6.0, 1.0))

    # Robots
    # Note: keep native scale for physics stability
    add_prim(
        stage,
        "/World/Franka_1",
        franka_usd,
        position=(-2.0, 0.5, 0.0),
        rotation_z_deg=90.0,
        scale=(1.0, 1.0, 1.0),
    )

    add_prim(
        stage,
        "/World/Franka_2",
        franka_usd,
        position=(2.0, 0.5, 0.0),
        rotation_z_deg=-90.0,
        scale=(1.0, 1.0, 1.0),
    )

    add_prim(
        stage,
        "/World/Jetbot",
        jetbot_usd,
        position=(0.0, 0.5, 0.0),
        rotation_z_deg=0.0,
        scale=(1.0, 1.0, 1.0),
    )

    # Colored cubes
    for color, pos in cube_positions.items():
        p = f"/World/Cubes/Cube_{color.capitalize()}"
        prim = create_box(stage, p, pos, cube_sizes, colors[color])
        set_rigid_body(prim, mass=0.1)

    # Nests
    for color, pos in nest_positions.items():
        create_nest(
            stage,
            f"/World/Nests/Nest_{color.capitalize()}",
            pos,
            colors[color],
            size=(0.20, 0.20, 0.03),
        )

    # Lighting
    omni.kit.commands.execute(
        "CreatePrim",
        prim_path="/World/Lights/DistantLight",
        prim_type="DistantLight",
    )

    print("Environment successfully created!")


build_scene()