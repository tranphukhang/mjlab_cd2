"""MJLab velocity environment for Open Duck Mini V2."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import gait_first_mdp
from .open_duck import (
  BASE_BODY,
  FOOT_BODIES,
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


# Env steps per PPO iteration (must match ``num_steps_per_env`` in rl_cfg.py).
_STEPS_PER_ITER = 24
# Curriculum phase boundaries, in PPO iterations.
_GAIT_PHASE_END = 400
_BLEND_PHASE_END = 1000


def open_duck_gait_first_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat task that learns stepping first and velocity tracking second.

  Phase A (iter 0-400): air time and single-foot contact dominate, velocity
  tracking is loose and commands are slow forward walks.
  Phase B (iter 400-1000): gait terms fade, velocity tracking tightens.
  Phase C (iter 1000+): velocity tracking dominates, backward commands and
  standing envs are enabled.
  """
  cfg = open_duck_flat_env_cfg(play=play)

  # Rewards an alternating gait; air_time alone also rewards two-foot hops.
  cfg.rewards["single_foot_contact"] = RewardTermCfg(
    func=gait_first_mdp.single_foot_contact,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.03,
    },
  )

  # Phase A values; the curriculum below takes over from step 0.
  cfg.rewards["air_time"].weight = 2.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.08
  cfg.rewards["air_time"].params["command_threshold"] = 0.0
  cfg.rewards["foot_swing_height"].weight = -0.2
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["action_rate_l2"].weight = -0.01

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_standing_envs = 0.0
  command.ranges.lin_vel_x = (0.05, 0.15)

  if play:
    # Play mode keeps the fixed 0.15 m/s command from the base config.
    command.ranges.lin_vel_x = (0.15, 0.15)
    return cfg

  b = _GAIT_PHASE_END * _STEPS_PER_ITER
  c = _BLEND_PHASE_END * _STEPS_PER_ITER

  def reward_stages(name: str, stages: list[dict]) -> CurriculumTermCfg:
    return CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={"reward_name": name, "stages": stages},
    )

  cfg.curriculum["gait_air_time"] = reward_stages(
    "air_time",
    [
      {"step": 0, "weight": 2.0, "params": {"command_threshold": 0.0}},
      {"step": b, "weight": 1.0, "params": {"command_threshold": 0.03}},
      {"step": c, "weight": 0.5},
    ],
  )
  cfg.curriculum["gait_single_foot"] = reward_stages(
    "single_foot_contact",
    [
      {"step": 0, "weight": 1.0},
      {"step": b, "weight": 0.5},
      {"step": c, "weight": 0.25},
    ],
  )
  cfg.curriculum["gait_swing_height"] = reward_stages(
    "foot_swing_height",
    [
      {"step": 0, "weight": -0.2},
      {"step": b, "weight": -0.1},
    ],
  )
  cfg.curriculum["track_lin_vel"] = reward_stages(
    "track_linear_velocity",
    [
      {"step": 0, "weight": 1.0, "params": {"std": math.sqrt(0.25)}},
      {"step": b, "weight": 3.0, "params": {"std": math.sqrt(0.10)}},
      {"step": c, "weight": 4.0, "params": {"std": math.sqrt(0.05)}},
    ],
  )
  cfg.curriculum["action_rate"] = reward_stages(
    "action_rate_l2",
    [
      {"step": 0, "weight": -0.01},
      {"step": b, "weight": -0.02},
      {"step": c, "weight": -0.05},
    ],
  )
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (0.05, 0.15)},
        {"step": b, "lin_vel_x": (0.0, 0.25)},
        {"step": c, "lin_vel_x": (-0.20, 0.30)},
      ],
    },
  )
  cfg.curriculum["standing_envs"] = CurriculumTermCfg(
    func=gait_first_mdp.standing_envs_curriculum,
    params={
      "command_name": "twist",
      "stages": [
        {"step": 0, "rel_standing_envs": 0.0},
        {"step": c, "rel_standing_envs": 0.10},
      ],
    },
  )

  return cfg


def open_duck_gait_first_zmp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gait-first task with ZMP-margin and friction-cone penalties.

  Meant for fine-tuning a finished GaitFirst checkpoint. The curricula count
  env steps, so a fresh env would restart at phase A; they are replaced here by
  their final (phase C) values.
  """
  cfg = open_duck_gait_first_env_cfg(play=play)
  for name in (
    "gait_air_time",
    "gait_single_foot",
    "gait_swing_height",
    "track_lin_vel",
    "action_rate",
    "command_vel",
    "standing_envs",
  ):
    cfg.curriculum.pop(name, None)

  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["command_threshold"] = 0.03
  cfg.rewards["single_foot_contact"].weight = 0.5  # Keep stepping under the foot penalties.
  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.05)
  cfg.rewards["action_rate_l2"].weight = -0.05

  if not play:
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.rel_standing_envs = 0.10
    command.ranges.lin_vel_x = (-0.20, 0.30)

  # Up to 4 contacts per foot with positions and contact-frame forces, for the
  # center of pressure (ZMP on flat ground) and the friction cone.
  feet_contact_points_cfg = ContactSensorCfg(
    name="feet_contact_points",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos"),
    reduce="maxforce",
    num_slots=4,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, feet_contact_points_cfg)

  feet_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)
  cfg.rewards["zmp_margin"] = RewardTermCfg(
    func=gait_first_mdp.zmp_margin_penalty,
    weight=-1.0,
    params={"sensor_name": "feet_contact_points", "std": 0.01, "asset_cfg": feet_cfg},
  )
  # Walking on the sole edge keeps the ZMP margin near zero; penalize the
  # sideways roll of loaded feet directly for a smoother learning signal.
  cfg.rewards["foot_tilt"] = RewardTermCfg(
    func=gait_first_mdp.stance_foot_tilt_penalty,
    weight=-2.0,
    params={"sensor_name": "feet_contact_points", "asset_cfg": feet_cfg},
  )
  # Sim floor friction is ~0.8; keep the gait inside mu = 0.5 for slippery floors.
  cfg.rewards["friction_cone"] = RewardTermCfg(
    func=gait_first_mdp.friction_cone_penalty,
    weight=-1.0,
    params={"sensor_name": "feet_contact_points", "mu": 0.5},
  )

  return cfg


# Walking gait clock, derived from the MJCF (see README "Gait clock timing"):
# CoM height 0.152 m (LIPM omega 8.0 rad/s), leg 0.158 m, feet 0.18 m apart,
# STS3215 servos. Single support of 0.16 s gives omega * T_ss ~ 1.3, the human
# walking ratio. Cycle: double support 0.04 s -> right swing 0.16 s -> double
# support 0.04 s -> left swing 0.16 s. A 0.6 s cycle (0.24 s swings) was only
# reached by pivoting on the stance foot; with heading held the policy fell back
# to ~0.1 s steps (wide stance, no ankle roll joint).
GAIT_PERIOD_S = 0.40
GAIT_DOUBLE_SUPPORT = 0.10  # Fraction of the period per double-support phase.


def open_duck_gait_phase_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """ZMP task plus a gait clock with a walking contact schedule.

  The actor and critic observe ``[sin, cos]`` of the clock phase (appended
  last, so an existing ZMP checkpoint can be widened with zero weights). The
  contact-schedule reward asks for the double support -> right swing ->
  double support -> left swing cycle of humanoid walking.
  """
  cfg = open_duck_gait_first_zmp_env_cfg(play=play)

  clock = ObservationTermCfg(
    func=gait_first_mdp.gait_clock, params={"period": GAIT_PERIOD_S}
  )
  cfg.observations["actor"].terms["gait_clock"] = clock
  cfg.observations["critic"].terms["gait_clock"] = clock

  cfg.rewards["gait_schedule"] = RewardTermCfg(
    func=gait_first_mdp.gait_contact_schedule,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "period": GAIT_PERIOD_S,
      "double_support": GAIT_DOUBLE_SUPPORT,
      "command_threshold": 0.03,
    },
  )
  # Rewarding exactly one foot down fights the double-support phases.
  cfg.rewards.pop("single_foot_contact", None)
  # feet_air_time pays every step a foot has been airborne 0.1-0.3 s, so a
  # 0.24 s + 0.16 s limp earned more than two scheduled 0.16 s swings. The
  # schedule sets swing timing; penalize deviations from it at touchdown.
  # At -10 the term averaged ~-0.09 per step against +1.2 for gait_schedule
  # and left the limp untouched; -100 puts it on par with zmp_margin.
  cfg.rewards.pop("air_time", None)
  cfg.rewards["swing_duration"] = RewardTermCfg(
    func=gait_first_mdp.swing_duration_penalty,
    weight=-100.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "target": (0.5 - GAIT_DOUBLE_SUPPORT) * GAIT_PERIOD_S,
      "command_threshold": 0.03,
    },
  )
  # Commands are in the body frame, so spinning while stepping forward still
  # tracks vx. With the base 0.5 / sqrt(0.5) yaw term the policy spun at
  # ~1.75 rad/s, pivoting on the stance foot to hit the schedule.
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
  # Even then it kept a steady clockwise drift (~0.24 rad/s). Hold a world
  # heading instead: the yaw-rate command becomes 1.0 * heading error, so any
  # drift is commanded back, and random targets teach turning both ways.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.heading_command = True
  command.heading_control_stiffness = 1.0
  command.rel_heading_envs = 1.0
  command.ranges.ang_vel_z = (-0.5, 0.5)
  # Play walks straight along world +x.
  command.ranges.heading = (0.0, 0.0) if play else (-math.pi, math.pi)
  return cfg
