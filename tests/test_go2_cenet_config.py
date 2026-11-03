"""Contract tests executed when the Isaac Sim Python modules are available."""

import pytest

pytest.importorskip("pxr", reason="launch through Isaac Sim for environment config tests")

from isaaclab_dreamwaq_tasks.tasks.manager_based.go2_cenet.go2_cenet_env_cfg import (
    POLICY_JOINT_NAMES,
    Go2CENetRoughEnvCfg,
    Go2CENetStandRefineEnvCfg,
)
from isaaclab_dreamwaq_tasks.tasks.manager_based.go2_cenet.agents.rsl_rl_ppo_cfg import Go2CENetPPORunnerCfg


def test_environment_contract() -> None:
    cfg = Go2CENetRoughEnvCfg()

    assert cfg.decimation == 4
    assert cfg.sim.dt == 0.005
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.actions.joint_pos.joint_names == POLICY_JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order is True
    assert cfg.observations.policy.history_length == 6
    assert cfg.observations.policy.flatten_history_dim is False
    assert cfg.observations.policy.enable_corruption is True
    assert cfg.observations.policy.joint_vel.scale == 0.05
    assert cfg.observations.policy.base_ang_vel.scale == 0.2
    assert cfg.observations.policy.joint_pos.clip == (-5.0, 5.0)
    assert cfg.observations.policy.actions.clip == (-2.5, 2.5)
    assert cfg.rewards.base_height.weight == -1.0
    assert cfg.rewards.base_height.params["target_height"] == 0.4
    assert cfg.rewards.undesired_contacts.weight == -1.0
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.stand_four_feet_contact.weight == 1.0
    assert cfg.rewards.stand_foot_velocity_l2.weight == -0.25
    assert cfg.terminations.bad_orientation.params["limit_angle"] == 0.8
    assert cfg.commands.base_velocity.rel_standing_envs == 0.10
    assert cfg.scene.terrain.max_init_terrain_level == 1

    refine_cfg = Go2CENetStandRefineEnvCfg()
    assert refine_cfg.commands.base_velocity.rel_standing_envs == 0.20
    assert refine_cfg.curriculum.velocity_commands is None
    assert refine_cfg.rewards.stand_four_feet_contact.weight == 1.0
    assert refine_cfg.rewards.stand_still.weight == -0.75


def test_agent_contract() -> None:
    cfg = Go2CENetPPORunnerCfg()

    assert cfg.max_iterations == 2000
    assert cfg.clip_actions == 2.5
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert cfg.actor.history_length == 6
    assert cfg.actor.observation_dim == 45
    assert cfg.actor.latent_dim == 16
    assert cfg.actor.hidden_dims == [128, 128, 128]
    assert cfg.actor.detach_context_from_actor is True
    assert cfg.algorithm.max_action_std == 0.8
