"""CPU tests for the CENet actor and its deployment export wrapper."""

import torch
from tensordict import TensorDict

from isaaclab_dreamwaq_rl.models import CENetActor


def _observations(batch_size: int = 8) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 6, 45),
            "cenet_velocity": torch.randn(batch_size, 3),
            "cenet_reconstruction": torch.randn(batch_size, 45),
        },
        batch_size=[batch_size],
    )


def _actor(obs: TensorDict) -> CENetActor:
    return CENetActor(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=12,
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
        history_length=6,
        observation_dim=45,
        encoder_hidden_dims=[128, 64],
        latent_dim=16,
        velocity_dim=3,
        decoder_hidden_dims=[64, 128],
    )


def test_actor_shapes_losses_and_gradients() -> None:
    obs = _observations()
    actor = _actor(obs)
    actor.train()

    actions = actor(obs, stochastic_output=False)
    actions.square().mean().backward(retain_graph=True)

    # DreamWaQ separation: PPO's actor loss must not reshape the estimator.
    assert actor.encoder[0].weight.grad is None
    actor.zero_grad(set_to_none=True)

    velocity_loss, reconstruction_loss, kl_loss = actor.get_cenet_losses(
        obs["cenet_velocity"],
        obs["cenet_reconstruction"],
    )
    total_loss = velocity_loss + reconstruction_loss + 0.01 * kl_loss
    total_loss.backward()

    assert actions.shape == (8, 12)
    assert actor._last_velocity.shape == (8, 3)
    assert actor._last_latent_mean.shape == (8, 16)
    assert actor._last_reconstruction.shape == (8, 45)
    assert torch.isfinite(total_loss)
    assert actor.encoder[0].weight.grad is not None
    assert torch.count_nonzero(actor.encoder[0].weight.grad).item() > 0


def test_eval_and_export_are_deterministic_two_input_models() -> None:
    obs = _observations(batch_size=2)
    actor = _actor(obs).eval()

    first = actor(obs)
    second = actor(obs)
    assert torch.equal(first, second)

    exported = actor.as_jit().eval()
    current = obs["policy"][:, -1, :]
    direct = exported(current, obs["policy"])
    assert direct.shape == (2, 12)
    assert torch.allclose(first, direct)

    scripted = torch.jit.script(exported)
    scripted_actions = scripted(current, obs["policy"])
    assert torch.allclose(direct, scripted_actions)
