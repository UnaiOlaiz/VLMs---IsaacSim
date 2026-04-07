# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "panda_hand",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = 0.35,
    max_x: float = 0.65,
    min_y: float = -0.2,
    max_y: float = 0.2,
    max_height: float = 0.5,
    min_vel: float = 0.1,
) -> torch.Tensor:
    """Termination condition: object is within target bounds and stationary."""
    object: RigidObject = env.scene[object_cfg.name]
    # Fixed: Corrected attribute access
    pos = object.data.root_pos_w - env.scene.env_origins
    vel = torch.norm(object.data.root_vel_w, dim=1)

    within_x = (pos[:, 0] > min_x) & (pos[:, 0] < max_x)
    within_y = (pos[:, 1] > min_y) & (pos[:, 1] < max_y)
    at_height = pos[:, 2] < max_height
    is_static = vel < min_vel

    return within_x & within_y & at_height & is_static
