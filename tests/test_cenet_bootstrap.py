"""CPU tests for transferring a proven 45-observation policy into CENet."""

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel

from isaaclab_dreamwaq_rl.models import CENetActor
from isaaclab_dreamwaq_rl.bootstrap import _bootstrap_cenet_from_legacy_state


def test_legacy_actor_transfer_is_exact_and_preserves_cenet() -> None:
    torch.manual_seed(11)
    batch_size = 8
    current_obs = torch.randn(batch_size, 45)
    history = torch.randn(batch_size, 6, 45)
    history[:, -1] = current_obs
    legacy_obs = TensorDict({"policy": current_obs, "critic": torch.randn(batch_size, 235)}, batch_size=[batch_size])
    cenet_obs = TensorDict(
        {"policy": history, "critic": legacy_obs["critic"]},
        batch_size=[batch_size],
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}

    legacy_actor = MLPModel(
        legacy_obs,
        obs_groups,
        "actor",
        12,
        hidden_dims=[128, 128, 128],
        distribution_cfg=distribution_cfg.copy(),
    )
    legacy_critic = MLPModel(legacy_obs, obs_groups, "critic", 1, hidden_dims=[128, 128, 128])
    cenet_actor = CENetActor(
        cenet_obs,
        obs_groups,
        "actor",
        12,
        hidden_dims=[128, 128, 128],
        distribution_cfg=distribution_cfg.copy(),
    )
    cenet_critic = MLPModel(cenet_obs, obs_groups, "critic", 1, hidden_dims=[128, 128, 128])
    encoder_before = cenet_actor.encoder[0].weight.detach().clone()

    checkpoint = {
        "actor_state_dict": legacy_actor.state_dict(),
        "critic_state_dict": legacy_critic.state_dict(),
    }
    _bootstrap_cenet_from_legacy_state(cenet_actor, cenet_critic, checkpoint)

    legacy_actor.eval()
    cenet_actor.eval()
    legacy_actions = legacy_actor(legacy_obs, stochastic_output=False)
    cenet_actions = cenet_actor(cenet_obs, stochastic_output=False)

    torch.testing.assert_close(cenet_actions, legacy_actions, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(cenet_critic(cenet_obs), legacy_critic(legacy_obs), rtol=0.0, atol=0.0)
    assert torch.equal(cenet_actor.encoder[0].weight, encoder_before)
    assert torch.count_nonzero(cenet_actor.mlp[0].weight[:, 45:]).item() == 0
