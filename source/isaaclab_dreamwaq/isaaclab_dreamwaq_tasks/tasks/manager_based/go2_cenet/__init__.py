"""Gym registration for Go2 DreamWaQ/CENet."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-CENet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_cenet_env_cfg:Go2CENetRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2CENetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-CENet-StandRefine-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_cenet_env_cfg:Go2CENetStandRefineEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2CENetStandRefinePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-CENet-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_cenet_env_cfg:Go2CENetRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2CENetPPORunnerCfg",
    },
)
