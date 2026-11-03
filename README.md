# DreamWAQ CENet

Task-focused implementation of a DreamWaQ-style Context-Aided Estimator Network (CENet) for Unitree Go2 locomotion in NVIDIA Isaac Lab.

This repository contains only the CENet workstream: the estimator-aware actor, PPO auxiliary losses, the Go2 training environment, training/evaluation scripts, and automated tests. It is not a copy of the complete UPB-RAT DreamWaQ project, and it does not contain the downstream `go2_onboard_deployment` project.

## Project status

- CENet is integrated with the Actor and PPO training loop.
- The policy uses six observation frames: five past frames plus the current frame.
- A policy was trained for 2,000 iterations (last checkpoint index: `1999`).
- The exported ONNX policy was tested in Gazebo and can track forward, backward, lateral, and yaw-rate commands.
- Automated result: `4 passed, 1 skipped`; the skipped configuration test requires a running Isaac Sim process.

The final checkpoint and ONNX model are intentionally not stored in this source-only sample repository.

## What CENet does

At every control step, the Actor must choose 12 joint commands. A single observation shows the robot now, while CENet uses recent history to estimate how the robot is moving and to produce a compact latent description of the current dynamics.

```text
six-frame observation history [B, 6, 45]
                    |
                    v
              CENet encoder
                    |
            +-------+--------+
            |                |
            v                v
estimated velocity [B, 3]  latent context [B, 16]
            |                |
            +-------+--------+
                    |
             context [B, 19]
                    |
current observation [B, 45] + context [B, 19]
                    |
                    v
             Actor input [B, 64]
                    |
                    v
              actions [B, 12]
```

CENet learns through three auxiliary objectives:

1. `velocity loss`: estimate the simulator's true body linear velocity.
2. `reconstruction loss`: predict the next clean 45-dimensional observation.
3. `KL loss`: keep the 16-dimensional latent representation structured.

PPO still learns the locomotion behavior from rewards and penalties. The auxiliary losses teach CENet to provide useful motion and dynamics information to the Actor.

## Observation contract

One observation frame has 45 values in this exact order:

| Slice | Size | Signal | Processing |
| --- | ---: | --- | --- |
| `0:12` | 12 | relative joint positions | scale `1.0` |
| `12:24` | 12 | joint velocities | raw clip `[-100, 100]`, scale `0.05` |
| `24:36` | 12 | previous actions | clip `[-2.5, 2.5]` |
| `36:39` | 3 | body angular velocity | raw clip `[-20, 20]`, scale `0.2` |
| `39:42` | 3 | projected gravity | clip `[-1, 1]` |
| `42:45` | 3 | commanded `vx`, `vy`, yaw rate | clip `[-5, 5]` |

Training and deployment must both apply the final observation clamp `[-5.0, 5.0]`.

History is ordered from oldest to newest:

```text
[o(t-5), o(t-4), o(t-3), o(t-2), o(t-1), o(t)]
```

With a simulation time step of `0.005 s` and control decimation `4`, the policy runs every `0.02 s` (50 Hz). Six frames contain five time intervals, so the history covers:

```text
(6 - 1) * 0.02 s = 0.10 s
```

## Network contract

| Tensor | Shape |
| --- | --- |
| current observation | `[B, 45]` |
| observation history | `[B, 6, 45]` |
| flattened CENet input | `[B, 270]` |
| estimated velocity | `[B, 3]` |
| latent mean / log variance | `[B, 16]` each |
| combined context | `[B, 19]` |
| Actor input | `[B, 64]` |
| actions | `[B, 12]` |

The deterministic exported model has this interface:

```text
forward(
    current_obs: [B, 45],
    obs_history: [B, 6, 45]
) -> actions: [B, 12]
```

## Action contract

Policy joint order:

```text
FL/FR/RL/RR hips,
FL/FR/RL/RR thighs,
FL/FR/RL/RR calves
```

Actions are applied as joint-position targets:

```text
clipped_action = clamp(action, -2.5, 2.5)
desired_joint_position = default_joint_position + 0.25 * clipped_action
```

Default joint positions in policy order:

```text
[ 0.10, -0.10,  0.10, -0.10,
  0.80,  0.80,  1.00,  1.00,
 -1.50, -1.50, -1.50, -1.50]
```

## Repository structure

```text
DreamWaQ-CENet/
|-- README.md
|-- pyproject.toml
|-- scripts/
|   |-- isaaclab_dreamwaq_rsl_rl/
|   |   |-- train.py
|   |   |-- play.py
|   |   `-- cli_args.py
|   |-- evaluate_cenet.py
|   `-- validate_cenet_task.py
|-- source/isaaclab_dreamwaq/
|   |-- setup.py
|   |-- config/extension.toml
|   |-- isaaclab_dreamwaq_rl/
|   |   |-- models/cenet_actor.py
|   |   |-- algorithms/cenet_ppo.py
|   |   |-- bootstrap.py
|   |   `-- rsl_rl/cenet_cfg.py
|   `-- isaaclab_dreamwaq_tasks/tasks/manager_based/go2_cenet/
|       |-- go2_cenet_env_cfg.py
|       |-- cenet_mdp.py
|       `-- agents/rsl_rl_ppo_cfg.py
`-- tests/
    |-- test_cenet_actor.py
    |-- test_cenet_bootstrap.py
    |-- test_cenet_ppo.py
    `-- test_go2_cenet_config.py
```

## Tested environment

The current implementation was validated with:

- Ubuntu Linux
- Python `3.11.15`
- NVIDIA Isaac Sim `5.1`
- Isaac Lab `0.54.3`
- RSL-RL `5.0.1`
- PyTorch `2.7.0+cu128`

Install Isaac Sim and Isaac Lab first. They are external dependencies and are not vendored in this repository.

## Installation

From an Isaac Lab installation, install this project in editable mode:

```bash
./isaaclab.sh -p -m pip install -e /path/to/DreamWaQ-CENet/source/isaaclab_dreamwaq
```

Replace `/path/to/DreamWaQ-CENet` and run the remaining commands from the repository root.

## Validate the task

The validation script checks task registration, observation shapes, history order, action shape, and a short environment rollout:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/validate_cenet_task.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-CENet-v0 \
  --num_envs 16 \
  --headless
```

## Run automated tests

```bash
PYTHONPATH=source/isaaclab_dreamwaq \
  /path/to/IsaacLab/isaaclab.sh -p -m pytest tests -q
```

The environment configuration test is skipped outside an Isaac Sim process because it requires the `pxr` runtime.

## Train a policy

Training from a teacher checkpoint is recommended because the legacy 45-observation Actor already knows a stable locomotion behavior. CENet is initialized so that its context initially has no effect, and it learns progressively during PPO updates.

```bash
/path/to/IsaacLab/isaaclab.sh -p \
  scripts/isaaclab_dreamwaq_rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-CENet-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  --bootstrap_checkpoint /path/to/teacher_checkpoint.pt
```

The task collects `24` transitions per environment per iteration. With `1024` environments, one rollout contains:

```text
1024 * 24 = 24,576 transitions
```

Fallen environments reset independently; their terminal transitions remain in the rollout, and the neural network is not reset.

## Resume training

```bash
/path/to/IsaacLab/isaaclab.sh -p \
  scripts/isaaclab_dreamwaq_rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-CENet-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  --resume \
  --load_run RUN_DIRECTORY_NAME \
  --checkpoint model_1650.pt
```

`--resume` and `--bootstrap_checkpoint` are intentionally mutually exclusive.

## Play and export ONNX

Visual testing is intentionally performed by the project owner:

```bash
/path/to/IsaacLab/isaaclab.sh -p \
  scripts/isaaclab_dreamwaq_rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-CENet-Play-v0 \
  --num_envs 16 \
  --checkpoint /path/to/model_1999.pt
```

The play script loads the checkpoint and exports deterministic JIT and ONNX models into an `exported/` directory beside the checkpoint.

## Quantitative CENet evaluation

The evaluation script compares normal history with deliberately damaged CENet inputs:

- `zero_past`: erase the five past frames.
- `reversed_past`: reverse the past-frame order.
- `zero_context`: prevent the Actor from receiving CENet context.

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/evaluate_cenet.py \
  --checkpoint /path/to/model_1999.pt \
  --num_envs 64 \
  --steps 1000 \
  --headless \
  --export
```

The trained V2 policy showed lower forward and lateral command-tracking error with normal CENet history/context than with the three damaged-input comparisons. This supports that the Actor uses CENet temporal information rather than merely carrying unused CENet code.

## Source-only scope

This repository intentionally excludes:

- training logs and TensorBoard events;
- intermediate and failed checkpoints;
- exported model binaries;
- videos and simulator outputs;
- Isaac Sim, Isaac Lab, and Conda installations;
- the `go2_onboard_deployment` repository;
- unrelated UPB-RAT template tasks and UI examples.

## References

This task was developed using the following projects as technical references and dependencies:

- [UPB-RAT/isaaclab_dreamwaq](https://github.com/UPB-RAT/isaaclab_dreamwaq)
- [Manaro-Alpha/DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ)
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

Existing source-file copyright and SPDX headers are preserved.
