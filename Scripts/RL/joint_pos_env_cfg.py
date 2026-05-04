import os
import json
import torch
import random

import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, EventTermCfg

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
import isaaclab_tasks.manager_based.manipulation.lift.mdp as lift_mdp

# --- 1. DATASET ---
json_folder = os.path.expanduser("~/Documents/PFG/Scripts/Control/cube_jsons/")
cube_colors = ["red", "green", "blue"]

def load_vlm_dataset(folder, colors):
    dataset = []
    for color in colors:
        path = os.path.join(folder, f"rl_start_{color}_cube.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list): dataset.extend(data)
                else: dataset.append(data)
    print(f"Fase 1: Cargados {len(dataset)} estados de precisión exacta.")
    return dataset

VLM_DATASET = load_vlm_dataset(json_folder, cube_colors)

# --- 2. LÓGICA DE RECOMPENSA Y RESET ---

def object_vel_norm(env, asset_cfg: SceneEntityCfg):
    """Evita errores de dimensión en el buffer de rewards."""
    return torch.norm(mdp.root_lin_vel_w(env, asset_cfg), dim=-1)

def reset_to_exact_vlm_pose(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Fase 1: Reset sin aleatoriedad (Jitter = 0)."""
    if not VLM_DATASET: return

    robot = env.scene[asset_cfg.name]
    obj = env.scene["object"]
    env_origins = env.scene.env_origins[env_ids]
    
    all_joint_pos = []
    all_cube_poses = []

    for i, _ in enumerate(env_ids):
        # Seleccionamos un par coherente Robot-Cubo
        data = random.choice(VLM_DATASET)
        
        # Posición de articulaciones
        all_joint_pos.append(torch.tensor(data["joint_positions"], device=env.device, dtype=torch.float32))
        
        # Posición del Cubo RELATIVA al origen de este entorno
        rel_pos = torch.tensor(data["world_cube_xyz"], device=env.device, dtype=torch.float32)
        rel_pos[2] = 0.042 # Altura fija para contacto perfecto con mesa
        
        # POSICIÓN EXACTA (Jitter = 0 para Fase 1)
        final_pos = env_origins[i] + rel_pos
        
        # Orientación identidad [1, 0, 0, 0]
        all_cube_poses.append(torch.cat([final_pos, torch.tensor([1, 0, 0, 0], device=env.device)]))

    # Escribir en simulador (Posiciones y Velocidades iniciales cero)
    joint_pos_tensor = torch.stack(all_joint_pos)
    robot.write_joint_state_to_sim(joint_pos_tensor, torch.zeros_like(joint_pos_tensor), env_ids=env_ids)
    obj.write_root_pose_to_sim(torch.stack(all_cube_poses), env_ids=env_ids)

# --- 3. CONFIGURACIÓN DEL ENTORNO ---

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Control del brazo y pinza
        self.actions.arm_action = lift_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = lift_mdp.BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0},
        )

        # End Effector
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            )],
        )
        self.commands.object_pose.body_name = "panda_hand"

        # Escena
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(solver_position_iteration_count=16),
            ),
        )

        # Recompensas enfocadas en levantar (Stay Up)
        self.rewards.object_is_lifted_stay_up = RewTerm(
            func=lift_mdp.object_is_lifted, 
            params={"minimal_height": 0.08, "object_cfg": SceneEntityCfg("object")}, 
            weight=15.0
        )
        self.rewards.object_velocity_penalty = RewTerm(
            func=object_vel_norm, 
            params={"asset_cfg": SceneEntityCfg("object")}, 
            weight=-0.01
        )

        # Evento de Reset de Fase 1
        if hasattr(self.events, "reset_object_position"):
            self.events.reset_object_position = None
            
        self.events.vlm_start_reset = EventTermCfg(
            func=reset_to_exact_vlm_pose, 
            mode="reset", 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
