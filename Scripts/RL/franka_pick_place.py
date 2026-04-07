# RL script for pick place
# dependencies
import os
import json 
import torch 
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from .lift_env_cfg import LiftEnvCfg

# paths for data loading
RL_START_DIR = os.path.expanduser("~/Documents/PFG/Scripts/Control/RL_START") # where end joint positions are saved
SCENARIO_FILES = ["rl_ready_red_cube_franka.json", "rl_ready_green_cube_franka.json", "rl_ready_blue_cube_franka.json"] # each scenario to be randomized at each start

# function to load the scenarios
def load_scenarios():
    data = [] # where the scenarios data will be added (all at once)
    for scenario in SCENARIO_FILES:
        p = os.path.join(RL_START_DIR, scenario)
        with open(p, "r") as f:
            data.append(json.load(f))

scenarios_data = load_scenarios()
# print(scenarios_data)

@configclass
class FrankaPickPlaceEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if scenarios_data:
            joints = scenarios_data[0]["joint_positions"]
            # 7 usual joints + 2 fingers
            self.scene.robot.init_state.joint_positions = {f"panda_joint{i+1}": joints[i] for i in range(7)}
            self.scene.robot.init_state.joint_positions["panda_finger_joint1"] = joints[7]
            self.scene.robot.init_state.joint_positions["panda_finger_joint2"] = joints[8]
            # arbitrary coordinates for the place (will change for jetbot coordinates)
            self.commands.object_pose.ranges.pos_x = (0.45, 0.45) 
            self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
            self.commands.object_pose.ranges.pos_z = (0.25, 0.25)

            # reward system
            self.rewards.object_goal_tracking.weight = 20.0  
            self.rewards.lifting_object.weight = 10.0        
        
            # a little bit of cube position randomization noise 
            self.events.reset_object_position.params["pose_range"] = {
                "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)
            }