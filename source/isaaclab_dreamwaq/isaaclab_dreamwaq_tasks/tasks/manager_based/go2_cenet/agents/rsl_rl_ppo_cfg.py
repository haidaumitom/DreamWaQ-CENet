"""RSL-RL configuration for Go2 DreamWaQ/CENet training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from isaaclab_dreamwaq_rl.rsl_rl import CENetActorCfg, CENetPPOCfg


@configclass
class Go2CENetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Healthy CENet locomotion phase with progressive commands and terrain."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "go2_dreamwaq_cenet_v2"
    run_name = "healthy_curriculum"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 2.5

    actor = CENetActorCfg(
        # Match the proven rinderudon healthy/stand-refine backbone so its
        # 45-observation actor can be transferred exactly at initialization.
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=CENetActorCfg.GaussianDistributionCfg(init_std=0.5, std_type="scalar"),
        history_length=6,
        observation_dim=45,
        encoder_hidden_dims=[128, 64],
        latent_dim=16,
        velocity_dim=3,
        decoder_hidden_dims=[64, 128],
        detach_context_from_actor=True,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = CENetPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        velocity_loss_coef=1.0,
        reconstruction_loss_coef=1.0,
        latent_kl_loss_coef=0.01,
        velocity_target_group="cenet_velocity",
        reconstruction_target_group="cenet_reconstruction",
        min_action_std=0.05,
        max_action_std=0.8,
    )


@configclass
class Go2CENetStandRefinePPORunnerCfg(Go2CENetPPORunnerCfg):
    """Low-entropy continuation that refines zero-command standing."""

    max_iterations = 500
    run_name = "stand_refine"
    algorithm = CENetPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        velocity_loss_coef=1.0,
        reconstruction_loss_coef=1.0,
        latent_kl_loss_coef=0.01,
        velocity_target_group="cenet_velocity",
        reconstruction_target_group="cenet_reconstruction",
        min_action_std=0.05,
        max_action_std=0.6,
    )
