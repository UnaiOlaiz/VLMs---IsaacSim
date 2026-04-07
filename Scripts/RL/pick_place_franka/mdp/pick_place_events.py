# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


import json
import os


# Path to your JSON files
JSON_PATH = os.path.abspath("../Control/RL_START")


def reset_to_predefined_pose(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Resets robot and object to positions defined in the color JSON files."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    colors = ["red", "green", "blue"]

    for i in env_ids:
        # Randomly pick one of the three scenarios
        color = colors[torch.randint(0, len(colors), (1,)).item()]
        file_path = os.path.join(JSON_PATH, f"{color}.json")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Apply Joint Positions (7 arm + 2 gripper)
        joint_pos = torch.tensor(data["joint_positions"], device=env.device)
        robot.write_joint_state_to_sim(joint_pos, env_ids=i.unsqueeze(0))

        # Apply Cube World Position
        cube_pos = torch.tensor(data["cube_world_pos"], device=env.device)
        root_state = obj.data.default_root_state[i].clone()
        root_state[0:3] = cube_pos + env.scene.env_origins[i]
        obj.write_root_state_to_sim(root_state.unsqueeze(0), env_ids=i.unsqueeze(0))
