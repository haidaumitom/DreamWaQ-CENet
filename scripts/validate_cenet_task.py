"""Launch Isaac Sim and validate the Go2 CENet environment contract."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_dreamwaq_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK_ID = "Isaac-Velocity-Rough-Unitree-Go2-CENet-v0"


def main() -> None:
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=8)
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        assert observations["policy"].shape == (env.unwrapped.num_envs, 6, 45)
        assert observations["critic"].ndim == 2
        assert observations["cenet_velocity"].shape == (env.unwrapped.num_envs, 3)
        assert observations["cenet_reconstruction"].shape == (env.unwrapped.num_envs, 45)
        assert torch.equal(observations["policy"][:, 0], observations["policy"][:, -1])

        zero_actions = torch.zeros(
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
            device=env.unwrapped.device,
        )
        next_observations, rewards, terminated, truncated, _ = env.step(zero_actions)
        assert next_observations["policy"].shape == (env.unwrapped.num_envs, 6, 45)
        assert rewards.shape == (env.unwrapped.num_envs,)
        assert terminated.shape == truncated.shape == (env.unwrapped.num_envs,)
        print("CENet task validation passed")
        print(f"policy={tuple(next_observations['policy'].shape)}")
        print(f"critic={tuple(next_observations['critic'].shape)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
