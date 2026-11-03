"""MJLab velocity environment for Open Duck Mini V2."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .open_duck import (
  BASE_BODY,
  FOOT_GEOMS,
  FOOT_SITES,
  OPEN_DUCK_ACTION_SCALE,
  get_open_duck_robot_cfg,
)


def open_duck_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage-0 flat-terrain locomotion task for Open Duck Mini V2."""
  cfg = make_velocity_env_cfg()

  # Robot.
  cfg.scene.entities = {"robot": get_open_duck_robot_cfg()}

  # Contact and foot-height sensing required by gait rewards.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  foot_height_scan_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
    pattern=RingPatternCfg.single_ring(radius=0.02, num_samples=2),
    ray_alignment="yaw",
    max_distance=0.5,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  cfg.scene.sensors = (feet_ground_cfg, foot_height_scan_cfg)

  # Stage 0 uses only a flat plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  # No terrain-height map is needed on a plane.
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Stage 0: learn the nominal gait first, without observation corruption.
  cfg.observations["actor"].enable_corruption = False

  # Residual joint-position action around the home pose.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = OPEN_DUCK_ACTION_SCALE
  joint_pos_action.use_default_offset = True

  # Keep resets close to the nominal crouched stance.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.01),
    "yaw": (-0.05, 0.05),
  }

  # Stage 0 deliberately removes robustness randomization and external pushes.
  # Re-enable these only after a stable forward gait has emerged.
  for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)

  # Posture regularization: keep a weak home-pose prior so it does not dominate
  # the locomotion objective.
  cfg.rewards["pose"].params["std_standing"] = {
    ".*hip_yaw": 0.08,
    ".*hip_roll": 0.06,
    ".*hip_pitch": 0.12,
    ".*knee": 0.15,
    ".*ankle": 0.12,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    ".*hip_yaw": 0.20,
    ".*hip_roll": 0.15,
    ".*hip_pitch": 0.35,
    ".*knee": 0.45,
    ".*ankle": 0.30,
  }
  cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params[
    "std_walking"
  ]
  cfg.rewards["pose"].params["walking_threshold"] = 0.03
  cfg.rewards["pose"].weight = 0.25

  # Body stabilization. These weights are intentionally weaker than the forward
  # velocity objective so that standing/crouching is not the best local optimum.
  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["upright"].weight = 1.0
  cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = -0.01

  # Forward velocity tracking is the dominant Stage-0 objective.
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.10)

  # Yaw-rate command is zero in Stage 0. Keep only a small stabilization reward
  # so this easy term cannot dominate the return.
  cfg.rewards["track_angular_velocity"].weight = 0.5
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.50)

  # Mild gait shaping. These terms should encourage foot exchange without making
  # motion more expensive than simply standing still.
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["threshold_min"] = 0.10
  cfg.rewards["air_time"].params["threshold_max"] = 0.30
  cfg.rewards["air_time"].params["command_threshold"] = 0.03

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

  cfg.rewards["foot_clearance"].weight = -0.10
  cfg.rewards["foot_clearance"].params["target_height"] = 0.02
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.03

  cfg.rewards["foot_swing_height"].weight = -0.10
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.02
  cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.03

  cfg.rewards["foot_slip"].weight = -0.05
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.03

  # Keep actions feasible, but do not over-penalize motion during gait discovery.
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["dof_pos_limits"].weight = -1.0
  cfg.rewards.pop("soft_landing", None)

  # Stage-0 command distribution:
  #   10% standing,
  #   90% straight forward walking at 0.10-0.20 m/s,
  #   no lateral motion and no yaw command.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_standing_envs = 0.10
  command.rel_heading_envs = 0.0
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 0.0
  command.heading_command = False
  command.ranges.lin_vel_x = (0.10, 0.20)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.ranges.heading = None
  cfg.curriculum.pop("command_vel", None)

  # 500 Hz physics and 50 Hz policy/control updates.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  cfg.episode_length_s = 20.0

  cfg.viewer.body_name = BASE_BODY
  cfg.viewer.distance = 1.0
  cfg.viewer.elevation = -10.0

  if play:
    # Deterministic forward command for visual evaluation of learned walking.
    cfg.episode_length_s = int(1e9)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.15, 0.15)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

  return cfg
