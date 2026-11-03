"""Go2 rough-terrain task with a six-frame proprioceptive history."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from . import cenet_mdp


POLICY_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

_FEET = SceneEntityCfg(
    "robot",
    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    preserve_order=True,
)
_CONTACT_FEET = SceneEntityCfg(
    "contact_forces",
    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    preserve_order=True,
)
_STAND_COMMAND = {"command_name": "base_velocity", "command_threshold": 0.1}


def _joint_entity() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True)


@configclass
class Go2CENetObservationsCfg:
    """Actor history, asymmetric critic state, and clean CENet targets."""

    @configclass
    class PolicyCfg(ObsGroup):
        # This declaration order is the deployment observation contract.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _joint_entity()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-5.0, 5.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _joint_entity()},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-2.5, 2.5))
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-20.0, 20.0),
            scale=0.2,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 6
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _joint_entity()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _joint_entity()})
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CENetVelocityCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CENetReconstructionCfg(ObsGroup):
        # Same 45D order/scales as PolicyCfg, but without sensor corruption/history.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _joint_entity()},
            clip=(-5.0, 5.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _joint_entity()},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-2.5, 2.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-20.0, 20.0), scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-1.0, 1.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    cenet_velocity: CENetVelocityCfg = CENetVelocityCfg()
    cenet_reconstruction: CENetReconstructionCfg = CENetReconstructionCfg()


@configclass
class Go2CENetRewardsCfg(RewardsCfg):
    """Locomotion rewards that prevent the low-crouch survival shortcut."""

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)
    action_soft_bound_l2 = RewTerm(
        func=cenet_mdp.action_soft_bound_l2,
        weight=-0.20,
        params={"threshold": 1.5},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": _CONTACT_FEET, "asset_cfg": _FEET},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,
        },
    )
    no_foot_crosses_inward = RewTerm(
        func=cenet_mdp.no_foot_crosses_inward,
        weight=-1.0,
        params={"asset_cfg": _FEET, "minimum_lateral_distance": 0.06},
    )
    # Gated to zero-command samples, so these proven stand-refine terms do not
    # constrain commanded locomotion.
    stand_four_feet_contact = RewTerm(
        func=cenet_mdp.stand_four_feet_contact,
        weight=1.0,
        params={
            **_STAND_COMMAND,
            "sensor_cfg": _CONTACT_FEET,
            "contact_threshold": 1.0,
        },
    )
    stand_foot_velocity_l2 = RewTerm(
        func=cenet_mdp.stand_foot_velocity_l2,
        weight=-0.25,
        params={**_STAND_COMMAND, "asset_cfg": _FEET, "velocity_normalizer": 0.25},
    )
    stand_joint_velocity_l2 = RewTerm(
        func=cenet_mdp.stand_joint_velocity_l2,
        weight=-0.05,
        params={**_STAND_COMMAND, "velocity_normalizer": 2.0},
    )
    stand_action_rate_l2 = RewTerm(
        func=cenet_mdp.stand_action_rate_l2,
        weight=-0.04,
        params={**_STAND_COMMAND},
    )


@configclass
class Go2CENetStandRefineRewardsCfg(Go2CENetRewardsCfg):
    """Zero-command-only terms for a quiet four-foot stance."""

    pass


@configclass
class Go2CENetTerminationsCfg(TerminationsCfg):
    """Reject fallen orientations before the robot can exploit ground support."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class Go2CENetCurriculumCfg(CurriculumCfg):
    """Teach small commands before exposing the actor to the final range."""

    velocity_commands = CurrTerm(
        func=cenet_mdp.velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "stages": (
                {
                    "until_step": 12_000,
                    "lin_vel_x": (-0.4, 0.6),
                    "lin_vel_y": (-0.2, 0.2),
                    "ang_vel_z": (-0.35, 0.35),
                    "standing_probability": 0.10,
                },
                {
                    "until_step": 36_000,
                    "lin_vel_x": (-0.6, 0.7),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.50, 0.50),
                    "standing_probability": 0.08,
                },
                {
                    "until_step": 1_000_000_000,
                    "lin_vel_x": (-0.8, 0.8),
                    "lin_vel_y": (-0.5, 0.5),
                    "ang_vel_z": (-0.70, 0.70),
                    "standing_probability": 0.05,
                },
            ),
        },
    )


@configclass
class Go2CENetRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Training environment for the first Go2 CENet policy."""

    observations: Go2CENetObservationsCfg = Go2CENetObservationsCfg()
    rewards: Go2CENetRewardsCfg = Go2CENetRewardsCfg()
    terminations: Go2CENetTerminationsCfg = Go2CENetTerminationsCfg()
    curriculum: Go2CENetCurriculumCfg = Go2CENetCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Explicit ordering prevents a simulator/hardware joint-contract drift.
        self.actions.joint_pos.joint_names = POLICY_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25

        # Direct vx/vy/yaw-rate commands, matching the three deployment inputs.
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.10
        command.resampling_time_range = (5.0, 5.0)
        command.ranges.heading = (0.0, 0.0)
        command.ranges.lin_vel_x = (-0.4, 0.6)
        command.ranges.lin_vel_y = (-0.2, 0.2)
        command.ranges.ang_vel_z = (-0.35, 0.35)

        # Go2-specific terrain and reward details.
        terrain_generator = self.scene.terrain.terrain_generator
        self.scene.terrain.max_init_terrain_level = 1
        if terrain_generator is not None:
            terrain_generator.curriculum = True
            if "boxes" in terrain_generator.sub_terrains:
                terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.10)
            if "random_rough" in terrain_generator.sub_terrains:
                terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
                terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -2.0
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # Dynamics randomization counters sensor spikes and improves sim-to-real robustness.
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.5, 1.2),
                "restitution_range": (0.0, 0.1),
            }
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_com.params["asset_cfg"].body_names = "base"
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.02, 0.02),
        }
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.4, 0.4),
            "y": (-0.3, 0.3),
        }
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.randomize_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.9, 1.1),
                "armature_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.1, 0.1)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.05, 0.05),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
        }


@configclass
class Go2CENetStandRefineEnvCfg(Go2CENetRoughEnvCfg):
    """Conservative continuation task for four-foot zero-command standing."""

    rewards: Go2CENetStandRefineRewardsCfg = Go2CENetStandRefineRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.curriculum.velocity_commands = None
        self.scene.terrain.max_init_terrain_level = 2
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.20
        command.ranges.lin_vel_x = (-0.8, 0.8)
        command.ranges.lin_vel_y = (-0.5, 0.5)
        command.ranges.ang_vel_z = (-0.70, 0.70)
        self.rewards.stand_still.weight = -0.75


@configclass
class Go2CENetRoughEnvCfg_PLAY(Go2CENetRoughEnvCfg):
    """Small deterministic scene for visual policy checks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None
        self.curriculum.terrain_levels = None
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.5, -3.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.35)
