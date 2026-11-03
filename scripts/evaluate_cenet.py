"""Headless quantitative evaluation and ablation suite for a Go2 CENet checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a Go2 CENet checkpoint without opening the simulator GUI.")
parser.add_argument("--checkpoint", type=str, required=True, help="CENet checkpoint to evaluate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-CENet-Play-v0",
    help="Registered deterministic evaluation task.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel evaluation environments.")
parser.add_argument("--steps", type=int, default=1000, help="Policy steps per command and history mode.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Initial steps excluded from metrics.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument(
    "--modes",
    nargs="+",
    choices=("normal", "zero_past", "reversed_past", "zero_context"),
    default=("normal", "zero_past", "reversed_past", "zero_context"),
    help="CENet input/context variants to compare.",
)
parser.add_argument(
    "--commands",
    nargs="+",
    choices=("stand", "forward", "lateral", "yaw"),
    default=("stand", "forward", "lateral", "yaw"),
    help="Fixed commands to evaluate.",
)
parser.add_argument(
    "--profile",
    choices=("nominal", "low_friction", "heavy_payload", "sensor_noise", "pushes", "rough_terrain", "combined"),
    default="nominal",
    help="Controlled dynamics/sensor profile. Non-nominal profiles require the training task config.",
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON result path.")
parser.add_argument("--export", action="store_true", help="Export deterministic JIT and ONNX policies.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_dreamwaq_tasks  # noqa: F401
from isaaclab_dreamwaq_rl.models import CENetActor


COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "lateral": (0.0, 0.3, 0.0),
    "yaw": (0.0, 0.0, 0.4),
}


def _configure_profile(env_cfg, profile: str) -> None:
    """Apply one controlled robustness condition before the simulator is created."""

    events = env_cfg.events
    terrain_generator = env_cfg.scene.terrain.terrain_generator
    env_cfg.scene.terrain.max_init_terrain_level = None

    # Start every profile from the same deterministic dynamics. Individual
    # profiles then enable exactly one stressor (or all stressors for combined).
    env_cfg.observations.policy.enable_corruption = False
    events.base_external_force_torque = None
    events.base_com = None
    events.randomize_actuator_gains = None
    events.randomize_joint_parameters = None
    events.physics_material.params.update(
        {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
        }
    )

    use_mass = profile in ("heavy_payload", "combined")
    use_pushes = profile in ("pushes", "combined")
    if use_mass:
        if events.add_base_mass is None:
            raise ValueError(f"Profile '{profile}' requires the training task, whose base-mass event is enabled")
        # +3 kg is deliberately beyond the +2 kg training range.
        events.add_base_mass.params["mass_distribution_params"] = (3.0, 3.0)
    else:
        events.add_base_mass = None

    if use_pushes:
        if events.push_robot is None:
            raise ValueError(f"Profile '{profile}' requires the training task, whose push event is enabled")
        events.push_robot.interval_range_s = (2.0, 2.0)
        events.push_robot.params["velocity_range"] = {
            "x": (-0.6, 0.6),
            "y": (-0.5, 0.5),
        }
    else:
        events.push_robot = None

    if profile == "low_friction":
        events.physics_material.params.update(
            {"static_friction_range": (0.4, 0.4), "dynamic_friction_range": (0.35, 0.35)}
        )
    elif profile == "combined":
        events.physics_material.params.update(
            {"static_friction_range": (0.45, 0.45), "dynamic_friction_range": (0.4, 0.4)}
        )

    if profile in ("sensor_noise", "combined"):
        env_cfg.observations.policy.enable_corruption = True
        for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity"):
            noise = getattr(env_cfg.observations.policy, term_name).noise
            noise.n_min *= 2.0
            noise.n_max *= 2.0

    if terrain_generator is not None:
        terrain_generator.curriculum = False
        if profile == "rough_terrain":
            if "boxes" in terrain_generator.sub_terrains:
                terrain_generator.sub_terrains["boxes"].grid_height_range = (0.10, 0.16)
            if "random_rough" in terrain_generator.sub_terrains:
                terrain_generator.sub_terrains["random_rough"].noise_range = (0.06, 0.10)
                terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


def _prepare_actor_observation(obs: TensorDict, mode: str) -> TensorDict:
    """Preserve the current frame while perturbing only CENet's temporal evidence."""

    if mode in ("normal", "zero_context"):
        return obs

    modified = obs.clone()
    history = modified["policy"]
    if mode == "zero_past":
        history[..., :-1, :] = 0.0
    elif mode == "reversed_past":
        history[..., :-1, :] = torch.flip(history[..., :-1, :], dims=(-2,))
    else:
        raise ValueError(f"Unsupported history mode: {mode}")
    return modified


def _set_fixed_command(command_term, command: tuple[float, float, float]) -> None:
    vx, vy, yaw = command
    command_term.cfg.ranges.lin_vel_x = (vx, vx)
    command_term.cfg.ranges.lin_vel_y = (vy, vy)
    command_term.cfg.ranges.ang_vel_z = (yaw, yaw)
    command_term.cfg.ranges.heading = (0.0, 0.0)
    command_term.cfg.rel_standing_envs = 1.0 if command == (0.0, 0.0, 0.0) else 0.0


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.steps <= args_cli.warmup_steps:
        raise ValueError("--steps must be greater than --warmup_steps")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.curriculum.velocity_commands = None
    env_cfg.curriculum.terrain_levels = None

    # Make every phase start from the same nominal state. Terrain remains varied
    # across environments but is identical across ablation modes.
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    _configure_profile(env_cfg, args_cli.profile)

    checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    env_cfg.log_dir = str(checkpoint_path.parent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint_path))
        actor = runner.alg.actor
        if not isinstance(actor, CENetActor):
            raise TypeError(f"Expected CENetActor, received {type(actor).__name__}")
        actor.eval()
        deterministic_output = actor.distribution.as_deterministic_output_module()

        if args_cli.export:
            export_dir = checkpoint_path.parent / "exported"
            runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
            runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")

        base_env = env.unwrapped
        command_term = base_env.command_manager.get_term("base_velocity")
        results: list[dict[str, float | int | str | list[float]]] = []

        for command_name in args_cli.commands:
            command = COMMANDS[command_name]
            _set_fixed_command(command_term, command)
            for mode in args_cli.modes:
                obs, _ = env.reset()
                sums = {
                    "reward": 0.0,
                    "velocity_abs_error": 0.0,
                    "velocity_sq_error": 0.0,
                    "tracking_xy_error": 0.0,
                    "tracking_yaw_error": 0.0,
                    "action_sq": 0.0,
                    "context_sq": 0.0,
                }
                terminated_count = 0
                timeout_count = 0
                sample_count = 0

                for step in range(args_cli.steps):
                    with torch.inference_mode():
                        actor_obs = _prepare_actor_observation(obs, mode)
                        history = actor_obs["policy"]
                        velocity, _, _, _, context = actor.cenet_forward(history, deterministic_latent=True)

                        if mode == "zero_context":
                            current = history[..., -1, :]
                            zero_context = torch.zeros_like(context)
                            actor_output = actor.mlp(torch.cat((current, zero_context), dim=-1))
                            actions = deterministic_output(actor_output)
                        else:
                            actions = actor(actor_obs, stochastic_output=False)

                        true_velocity = obs["cenet_velocity"]
                        current_command = obs["policy"][..., -1, 42:45]
                        yaw_velocity = base_env.scene["robot"].data.root_ang_vel_b[:, 2]
                        velocity_error = velocity - true_velocity

                    next_obs, rewards, dones, _ = env.step(actions)

                    if step >= args_cli.warmup_steps:
                        sums["reward"] += rewards.sum().item()
                        sums["velocity_abs_error"] += velocity_error.abs().sum().item()
                        sums["velocity_sq_error"] += velocity_error.square().sum().item()
                        sums["tracking_xy_error"] += torch.linalg.vector_norm(
                            true_velocity[:, :2] - current_command[:, :2], dim=1
                        ).sum().item()
                        sums["tracking_yaw_error"] += torch.abs(
                            yaw_velocity - current_command[:, 2]
                        ).sum().item()
                        sums["action_sq"] += actions.square().sum().item()
                        sums["context_sq"] += context.square().sum().item()
                        terminated_count += int(base_env.reset_terminated.sum().item())
                        timeout_count += int(base_env.reset_time_outs.sum().item())
                        sample_count += args_cli.num_envs

                    actor.reset(dones)
                    obs = next_obs

                episodes = terminated_count + timeout_count
                results.append(
                    {
                        "command": command_name,
                        "command_values": list(command),
                        "mode": mode,
                        "samples": sample_count,
                        "mean_reward_per_step": sums["reward"] / sample_count,
                        "velocity_mae": sums["velocity_abs_error"] / (sample_count * 3),
                        "velocity_rmse": (sums["velocity_sq_error"] / (sample_count * 3)) ** 0.5,
                        "tracking_xy_error": sums["tracking_xy_error"] / sample_count,
                        "tracking_yaw_error": sums["tracking_yaw_error"] / sample_count,
                        "action_rms": (sums["action_sq"] / (sample_count * 12)) ** 0.5,
                        "context_rms": (sums["context_sq"] / (sample_count * actor.context_dim)) ** 0.5,
                        "terminated_episodes": terminated_count,
                        "timed_out_episodes": timeout_count,
                        "termination_fraction": terminated_count / episodes if episodes else 0.0,
                    }
                )
                print(json.dumps(results[-1], sort_keys=True))

        report = {
            "checkpoint": str(checkpoint_path),
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "warmup_steps": args_cli.warmup_steps,
            "profile": args_cli.profile,
            "results": results,
        }
        report_text = json.dumps(report, indent=2, sort_keys=True)
        if args_cli.output is not None:
            output_path = Path(args_cli.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text + "\n", encoding="utf-8")
            print(f"Saved evaluation report to: {output_path}")
        else:
            print(report_text)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
