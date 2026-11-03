"""PPO with auxiliary CENet objectives for RSL-RL 5."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution

from ..models import CENetActor


class CENetPPO(PPO):
    """Add velocity estimation, next-frame reconstruction, and latent KL losses."""

    actor: CENetActor

    def __init__(
        self,
        *args,
        velocity_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 1.0,
        latent_kl_loss_coef: float = 0.01,
        min_action_std: float = 0.05,
        max_action_std: float = 0.8,
        velocity_target_group: str = "cenet_velocity",
        reconstruction_target_group: str = "cenet_reconstruction",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, CENetActor):
            raise TypeError(f"CENetPPO requires CENetActor, received {type(self.actor).__name__}")
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("CENetPPO does not yet combine auxiliary losses with RND or symmetry augmentation")

        self.velocity_loss_coef = velocity_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.latent_kl_loss_coef = latent_kl_loss_coef
        if not 0.0 < min_action_std <= max_action_std:
            raise ValueError("Expected 0 < min_action_std <= max_action_std")
        self.min_action_std = min_action_std
        self.max_action_std = max_action_std
        self.velocity_target_group = velocity_target_group
        self.reconstruction_target_group = reconstruction_target_group

    def _clamp_action_std(self) -> None:
        """Keep learned exploration noise inside a deployment-safe range."""

        distribution = self.actor.distribution
        if not isinstance(distribution, GaussianDistribution):
            raise TypeError("CENetPPO action-std limits require GaussianDistribution")
        with torch.no_grad():
            if distribution.std_type == "scalar":
                distribution.std_param.clamp_(self.min_action_std, self.max_action_std)
            elif distribution.std_type == "log":
                distribution.log_std_param.clamp_(
                    torch.log(torch.tensor(self.min_action_std, device=distribution.log_std_param.device)),
                    torch.log(torch.tensor(self.max_action_std, device=distribution.log_std_param.device)),
                )
            else:
                raise ValueError(f"Unsupported action std type: {distribution.std_type}")

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Replace the stored clean current frame with the clean next-frame target.

        IsaacLab returns the reset observation on terminal transitions. Those
        transitions retain their current frame as target so CENet never learns
        an artificial cross-episode jump.
        """

        transition_obs = self.transition.observations
        if transition_obs is None:
            raise RuntimeError("process_env_step called without a pending transition")

        current_target = transition_obs[self.reconstruction_target_group]
        next_target = obs[self.reconstruction_target_group]
        terminal_mask = dones.to(dtype=torch.bool).view(-1, 1)
        transition_obs[self.reconstruction_target_group] = torch.where(
            terminal_mask,
            current_target,
            next_target,
        )
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Optimize PPO and CENet parameters over the same rollout mini-batches."""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_velocity_loss = 0.0
        mean_reconstruction_loss = 0.0
        mean_latent_kl_loss = 0.0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            velocity_target = batch.observations[self.velocity_target_group].detach()
            reconstruction_target = batch.observations[self.reconstruction_target_group].detach()
            velocity_loss, reconstruction_loss, latent_kl_loss = self.actor.get_cenet_losses(
                velocity_target,
                reconstruction_target,
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.velocity_loss_coef * velocity_loss
                + self.reconstruction_loss_coef * reconstruction_loss
                + self.latent_kl_loss_coef * latent_kl_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_action_std()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_velocity_loss += velocity_loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_latent_kl_loss += latent_kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "cenet_velocity": mean_velocity_loss / num_updates,
            "cenet_reconstruction": mean_reconstruction_loss / num_updates,
            "cenet_kl": mean_latent_kl_loss / num_updates,
        }
