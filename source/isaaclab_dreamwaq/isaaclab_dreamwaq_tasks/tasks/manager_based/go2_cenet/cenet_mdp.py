"""Task-specific rewards and curricula for Go2 CENet locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def action_soft_bound_l2(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize only action magnitudes outside the normal locomotion range."""

    excess = torch.clamp(torch.abs(env.action_manager.action) - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def _standing_gate(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """Return one only for near-zero linear and yaw velocity commands."""

    command = env.command_manager.get_command(command_name)
    return (torch.linalg.vector_norm(command, dim=1) < command_threshold).to(torch.float32)


def stand_four_feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Reward a four-foot support polygon only while commanded to stand."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1
    )
    contacts = (forces.amax(dim=1) > contact_threshold).to(torch.float32)
    contact_fraction = torch.mean(contacts, dim=1)
    all_contact = torch.prod(contacts, dim=1)
    score = 0.5 * contact_fraction + 0.5 * all_contact
    return score * _standing_gate(env, command_name, command_threshold)


def stand_foot_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    velocity_normalizer: float,
) -> torch.Tensor:
    """Penalize horizontal and vertical foot motion while standing."""

    if velocity_normalizer <= 0.0:
        raise ValueError("velocity_normalizer must be positive")
    asset = env.scene[asset_cfg.name]
    velocity_sq = torch.sum(
        torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]), dim=-1
    )
    normalized = torch.clamp(
        torch.mean(velocity_sq, dim=1) / velocity_normalizer**2, max=4.0
    )
    return normalized * _standing_gate(env, command_name, command_threshold)


def stand_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    velocity_normalizer: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual joint motion while standing."""

    if velocity_normalizer <= 0.0:
        raise ValueError("velocity_normalizer must be positive")
    asset = env.scene[asset_cfg.name]
    velocity_sq = torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids])
    normalized = torch.clamp(
        torch.mean(velocity_sq, dim=1) / velocity_normalizer**2, max=4.0
    )
    return normalized * _standing_gate(env, command_name, command_threshold)


def stand_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize policy-output oscillation only at a zero command."""

    action_delta = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(torch.square(action_delta), dim=1)
    return penalty * _standing_gate(env, command_name, command_threshold)


def _foot_positions_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=env.device)
    relative_w = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, len(body_ids), -1)
    return quat_apply_inverse(yaw, relative_w)


def no_foot_crosses_inward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_lateral_distance: float,
) -> torch.Tensor:
    """Penalize feet that cross toward the body centerline."""

    foot_y = _foot_positions_yaw(env, asset_cfg)[..., 1]
    side_sign = torch.tensor((1.0, -1.0, 1.0, -1.0), device=env.device)
    signed_distance = foot_y * side_sign
    violation = torch.clamp(minimum_lateral_distance - signed_distance, min=0.0)
    return torch.mean(torch.square(violation / minimum_lateral_distance), dim=1)


def velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    stages: tuple[dict, ...],
) -> dict[str, float]:
    """Expand velocity commands only after easier locomotion has been rehearsed."""

    del env_ids
    stage_index = len(stages) - 1
    for index, stage in enumerate(stages):
        if env.common_step_counter < stage["until_step"]:
            stage_index = index
            break

    stage = stages[stage_index]
    command = env.command_manager.get_term(command_name)
    command.cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    command.cfg.ranges.lin_vel_y = tuple(stage["lin_vel_y"])
    command.cfg.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
    command.cfg.rel_standing_envs = float(stage["standing_probability"])
    command.cfg.rel_heading_envs = 0.0
    command.cfg.heading_command = False

    return {
        "stage": float(stage_index),
        "standing_probability": float(stage["standing_probability"]),
        "max_lin_vel_x": float(stage["lin_vel_x"][1]),
        "max_lin_vel_y": float(stage["lin_vel_y"][1]),
        "max_ang_vel_z": float(stage["ang_vel_z"][1]),
    }
