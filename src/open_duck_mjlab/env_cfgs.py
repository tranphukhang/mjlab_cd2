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
  """Create the initial flat-terrain locomotion task for Open Duck Mini V2."""
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

  # Flat terrain only for the first training milestone.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)

  # No terrain-height map is needed on a plane.
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Residual joint-position action around the home pose.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = OPEN_DUCK_ACTION_SCALE
  joint_pos_action.use_default_offset = True

  # Robot-specific event targets.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  # Keep the initial reset close to the nominal crouched stance.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.01),
    "yaw": (-0.05, 0.05),
  }
  cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.15, 0.15),
    "y": (-0.10, 0.10),
  }

  # Posture regularization: tight while standing, looser while walking.
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
  cfg.rewards["pose"].weight = 1.0

  # Body stabilization.
  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # Velocity tracking is the main objective.
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.10)
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.50)

  # Gait shaping. Start weaker than MicroDuck and tune only after basic walking.
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.10
  cfg.rewards["air_time"].params["threshold_max"] = 0.30
  cfg.rewards["air_time"].params["command_threshold"] = 0.03

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITES

  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.02
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.03
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.02
  cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.03
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.03

  # Smooth, feasible commands for a real position-servo robot.
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["dof_pos_limits"].weight = -1.0
  cfg.rewards.pop("soft_landing", None)

  # Stage 0 command distribution: forward/backward only.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_standing_envs = 0.20
  command.rel_heading_envs = 0.0
  command.rel_forward_envs = 0.80
  command.heading_command = False
  command.ranges.lin_vel_x = (-0.20, 0.25)
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
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
