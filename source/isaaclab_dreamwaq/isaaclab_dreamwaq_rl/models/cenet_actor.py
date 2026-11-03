"""CENet-conditioned actor compatible with RSL-RL 5."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Create an MLP without an activation on its output."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(copy.deepcopy(resolve_nn_activation(activation)))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class CENetActor(MLPModel):
    """Actor that infers velocity and latent context from six observation frames.

    RSL-RL receives one actor observation group with shape ``[B, H, D]``. The
    base :class:`MLPModel` supplies the stochastic action distribution and actor
    MLP, while this class replaces its latent construction with CENet.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_length: int = 6,
        observation_dim: int = 45,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        latent_dim: int = 16,
        velocity_dim: int = 3,
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        detach_context_from_actor: bool = True,
    ) -> None:
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.context_dim = velocity_dim + latent_dim
        self.actor_input_dim = observation_dim + self.context_dim
        self.detach_context_from_actor = detach_context_from_actor
        self._history_obs_groups: list[str] = []

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        encoder_hidden_dims = list(encoder_hidden_dims)
        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must contain at least one layer")
        encoder_output_dim = encoder_hidden_dims[-1]
        encoder_prefix = encoder_hidden_dims[:-1]
        self.encoder = _build_mlp(
            history_length * observation_dim,
            encoder_prefix,
            encoder_output_dim,
            activation,
        )
        self.velocity_head = nn.Linear(encoder_output_dim, velocity_dim)
        self.latent_mean_head = nn.Linear(encoder_output_dim, latent_dim)
        self.latent_logvar_head = nn.Linear(encoder_output_dim, latent_dim)
        self.decoder = _build_mlp(self.context_dim, list(decoder_hidden_dims), observation_dim, activation)

        self._last_velocity: torch.Tensor | None = None
        self._last_latent_mean: torch.Tensor | None = None
        self._last_latent_logvar: torch.Tensor | None = None
        self._last_reconstruction: torch.Tensor | None = None

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        if len(active_obs_groups) != 1:
            raise ValueError(f"CENetActor expects one history group, received {active_obs_groups}")

        group_name = active_obs_groups[0]
        group_shape = tuple(obs[group_name].shape[1:])
        expected_shape = (self.history_length, self.observation_dim)
        if group_shape != expected_shape:
            raise ValueError(
                f"CENetActor expected '{group_name}' shape [B,{self.history_length},{self.observation_dim}], "
                f"received {tuple(obs[group_name].shape)}"
            )
        self._history_obs_groups = active_obs_groups
        return active_obs_groups, math.prod(group_shape)

    def _get_latent_dim(self) -> int:
        return self.actor_input_dim

    def _history_tensor(self, obs: TensorDict) -> torch.Tensor:
        history = obs[self._history_obs_groups[0]]
        if history.shape[-2:] != (self.history_length, self.observation_dim):
            raise ValueError(
                f"Invalid runtime history shape {tuple(history.shape)}; expected "
                f"[...,{self.history_length},{self.observation_dim}]"
            )
        return history

    def cenet_forward(
        self, history: torch.Tensor, deterministic_latent: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return velocity, latent statistics, reconstruction, and sampled context."""

        flat_history = history.reshape(*history.shape[:-2], -1)
        flat_history = self.obs_normalizer(flat_history)
        encoded = self.encoder(flat_history)
        velocity = self.velocity_head(encoded)
        latent_mean = self.latent_mean_head(encoded)
        latent_logvar = self.latent_logvar_head(encoded).clamp(-10.0, 10.0)

        if deterministic_latent is None:
            deterministic_latent = not self.training
        if deterministic_latent:
            latent = latent_mean
        else:
            latent_std = torch.exp(0.5 * latent_logvar)
            latent = latent_mean + latent_std * torch.randn_like(latent_std)

        context = torch.cat((velocity, latent), dim=-1)
        reconstruction = self.decoder(context)
        return velocity, latent_mean, latent_logvar, reconstruction, context

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        history = self._history_tensor(obs)
        velocity, latent_mean, latent_logvar, reconstruction, context = self.cenet_forward(history)

        self._last_velocity = velocity
        self._last_latent_mean = latent_mean
        self._last_latent_logvar = latent_logvar
        self._last_reconstruction = reconstruction

        current_observation = history[..., -1, :]
        actor_context = context.detach() if self.detach_context_from_actor else context
        return torch.cat((current_observation, actor_context), dim=-1)

    def get_cenet_losses(
        self, velocity_target: torch.Tensor, reconstruction_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute auxiliary losses from the most recent actor forward pass."""

        if any(
            output is None
            for output in (
                self._last_velocity,
                self._last_latent_mean,
                self._last_latent_logvar,
                self._last_reconstruction,
            )
        ):
            raise RuntimeError("Actor forward must run before CENet losses are requested")

        velocity_loss = nn.functional.mse_loss(self._last_velocity, velocity_target)
        reconstruction_loss = nn.functional.mse_loss(self._last_reconstruction, reconstruction_target)
        latent_kl_loss = -0.5 * torch.mean(
            1.0
            + self._last_latent_logvar
            - self._last_latent_mean.pow(2)
            - self._last_latent_logvar.exp()
        )
        return velocity_loss, reconstruction_loss, latent_kl_loss

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            history = self._history_tensor(obs)
            self.obs_normalizer.update(history.reshape(history.shape[0], -1))

    def as_jit(self) -> nn.Module:
        return _CENetInferencePolicy(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _CENetInferencePolicy(self, verbose=verbose)


class _CENetInferencePolicy(nn.Module):
    """Deterministic, two-input CENet policy used by JIT and ONNX export."""

    def __init__(self, model: CENetActor, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.history_length = model.history_length
        self.observation_dim = model.observation_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.velocity_head = copy.deepcopy(model.velocity_head)
        self.latent_mean_head = copy.deepcopy(model.latent_mean_head)
        self.actor = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, current_obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        flat_history = obs_history.reshape(obs_history.shape[0], -1)
        flat_history = self.obs_normalizer(flat_history)
        encoded = self.encoder(flat_history)
        velocity = self.velocity_head(encoded)
        latent_mean = self.latent_mean_head(encoded)
        context = torch.cat((velocity, latent_mean), dim=-1)
        actor_output = self.actor(torch.cat((current_obs, context), dim=-1))
        return self.deterministic_output(actor_output)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.observation_dim),
            torch.zeros(1, self.history_length, self.observation_dim),
        )

    @property
    def input_names(self) -> list[str]:
        return ["current_obs", "obs_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @torch.jit.export
    def reset(self) -> None:
        pass
