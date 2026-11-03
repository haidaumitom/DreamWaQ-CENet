"""Small synthetic rollout test for the RSL-RL 5 CENet PPO integration."""

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from isaaclab_dreamwaq_rl.algorithms import CENetPPO
from isaaclab_dreamwaq_rl.models import CENetActor


def _obs(batch_size: int) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 6, 45),
            "critic": torch.randn(batch_size, 51),
            "cenet_velocity": torch.randn(batch_size, 3),
            "cenet_reconstruction": torch.randn(batch_size, 45),
        },
        batch_size=[batch_size],
    )


def test_cenet_ppo_rollout_target_and_update() -> None:
    torch.manual_seed(7)
    num_envs = 4
    num_steps = 2
    initial_obs = _obs(num_envs)
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = CENetActor(
        initial_obs,
        obs_groups,
        "actor",
        12,
        hidden_dims=[64, 32],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
    )
    critic = MLPModel(initial_obs, obs_groups, "critic", 1, hidden_dims=[64, 32])
    storage = RolloutStorage("rl", num_envs, num_steps, initial_obs, [12], "cpu")
    algorithm = CENetPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=5.0e-4,
        rnd_cfg=None,
        symmetry_cfg=None,
    )
    with torch.no_grad():
        actor.distribution.std_param.fill_(1.2)

    obs = initial_obs
    encoder_before = actor.encoder[0].weight.detach().clone()
    for step in range(num_steps):
        algorithm.act(obs)
        next_obs = _obs(num_envs)
        dones = torch.tensor([1, 0, 0, 0]) if step == 0 else torch.zeros(num_envs, dtype=torch.long)
        current_target = obs["cenet_reconstruction"].clone()
        next_target = next_obs["cenet_reconstruction"].clone()
        algorithm.process_env_step(next_obs, torch.randn(num_envs), dones, {})

        stored_target = storage.observations[step]["cenet_reconstruction"]
        expected_first = current_target[0] if dones[0] else next_target[0]
        assert torch.equal(stored_target[0], expected_first)
        assert torch.equal(stored_target[1:], next_target[1:])
        obs = next_obs

    algorithm.compute_returns(obs)
    losses = algorithm.update()

    assert set(losses) == {
        "value",
        "surrogate",
        "entropy",
        "cenet_velocity",
        "cenet_reconstruction",
        "cenet_kl",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(encoder_before, actor.encoder[0].weight.detach())
    assert actor.distribution.std_param.max().item() <= 0.8 + 1.0e-6
    assert storage.step == 0
