"""IsaacLab configuration objects for the CENet actor and PPO losses."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class CENetActorCfg(RslRlMLPModelCfg):
    """Configuration for :class:`isaaclab_dreamwaq_rl.models.CENetActor`."""

    class_name: str = "isaaclab_dreamwaq_rl.models:CENetActor"
    history_length: int = 6
    observation_dim: int = 45
    encoder_hidden_dims: list[int] = [128, 64]
    latent_dim: int = 16
    velocity_dim: int = 3
    decoder_hidden_dims: list[int] = [64, 128]
    detach_context_from_actor: bool = True


@configclass
class CENetPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with DreamWaQ auxiliary-loss coefficients."""

    class_name: str = "isaaclab_dreamwaq_rl.algorithms:CENetPPO"
    velocity_loss_coef: float = 1.0
    reconstruction_loss_coef: float = 1.0
    latent_kl_loss_coef: float = 0.01
    min_action_std: float = 0.05
    max_action_std: float = 0.8
    velocity_target_group: str = "cenet_velocity"
    reconstruction_target_group: str = "cenet_reconstruction"
