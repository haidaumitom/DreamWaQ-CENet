"""Checkpoint transfer helpers for CENet training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .models import CENetActor


def bootstrap_cenet_from_legacy_checkpoint(
    actor: CENetActor,
    critic: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Transfer a 45-observation locomotion policy into a CENet actor.

    The legacy actor and the CENet actor must use the same hidden layers. The
    legacy first-layer weights are copied into the current-observation columns;
    all new context columns are initialized to zero. Consequently the CENet
    policy initially produces exactly the legacy actor output while its
    estimator learns from the six-frame auxiliary targets.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _bootstrap_cenet_from_legacy_state(actor, critic, checkpoint)
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_iteration": int(checkpoint.get("iter", -1)),
        "observation_dim": actor.observation_dim,
        "context_dim": actor.context_dim,
    }


def _bootstrap_cenet_from_legacy_state(
    actor: CENetActor,
    critic: nn.Module,
    checkpoint: dict[str, Any],
) -> None:
    """Apply an already-loaded legacy checkpoint (kept separate for tests)."""

    source_actor = checkpoint["actor_state_dict"]
    source_critic = checkpoint["critic_state_dict"]
    target_actor = actor.state_dict()

    first_layer_key = "mlp.0.weight"
    source_first = source_actor[first_layer_key]
    target_first = target_actor[first_layer_key]
    expected_source_shape = (target_first.shape[0], actor.observation_dim)
    if tuple(source_first.shape) != expected_source_shape:
        raise ValueError(
            f"Expected legacy actor first layer {expected_source_shape}, received {tuple(source_first.shape)}"
        )
    if target_first.shape[1] != actor.observation_dim + actor.context_dim:
        raise ValueError("CENet actor first-layer input does not match observation + context dimensions")

    with torch.no_grad():
        target_first.zero_()
        target_first[:, : actor.observation_dim].copy_(source_first)

        for name, source_value in source_actor.items():
            if name == first_layer_key:
                continue
            if not (name.startswith("mlp.") or name.startswith("distribution.")):
                continue
            if name not in target_actor:
                raise ValueError(f"Legacy actor parameter is absent from CENet actor: {name}")
            if target_actor[name].shape != source_value.shape:
                raise ValueError(
                    f"Actor parameter shape mismatch for {name}: "
                    f"expected {tuple(target_actor[name].shape)}, received {tuple(source_value.shape)}"
                )
            target_actor[name].copy_(source_value)

    actor.load_state_dict(target_actor, strict=True)
    critic.load_state_dict(source_critic, strict=True)
