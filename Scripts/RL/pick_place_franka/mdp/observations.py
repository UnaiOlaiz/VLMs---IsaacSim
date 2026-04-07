# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_obs(env: ManagerBasedRLEnv, eef_link_name: str) -> torch.Tensor:
    """Object observations: object pos, object quat, and eef-to-object vector."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(eef_link_name)
    # Fixed: added .env_origins to match Isaac Lab API
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w
    eef_to_object = object_pos - eef_pos

    return torch.cat((object_pos, object_quat, eef_to_object), dim=1)

def get_robot_joint_state(env: ManagerBasedRLEnv, joint_names: list[str]) -> torch.Tensor:
    """Returns the current joint positions for the specified joints."""
    indices, _ = env.scene["robot"].find_joints(joint_names)
    return env.scene["robot"].data.joint_pos[:, indices]
