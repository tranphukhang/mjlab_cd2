"""Extra MDP terms for the gait-first Open Duck curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor

# Sole outline of each ankle body in its own frame, measured from the
# L5/R5_ankle collision meshes at the home pose. The sole normal is the body
# -x axis; the sole spans body y (heel to toe) and body z (width).
# Rows are (left, right); columns are (y_min, y_max, z_min, z_max) in meters.
_SOLE_BOUNDS = (
  (-0.0620, 0.0327, -0.0124, 0.0144),
  (-0.0322, 0.0626, -0.0123, 0.0145),
)
# A foot carrying less normal force than this (N) is treated as unloaded.
_MIN_FOOT_FORCE = 0.5


class StandingStage(TypedDict):
  step: int
  rel_standing_envs: float


def standing_envs_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[StandingStage],
) -> dict[str, torch.Tensor]:
  """Change the fraction of standing (zero-command) envs over training.

  The velocity command reads ``rel_standing_envs`` on every resample, so
  updating the config is enough for the change to take effect.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      cfg.rel_standing_envs = stage["rel_standing_envs"]
  return {"rel_standing_envs": torch.tensor(cfg.rel_standing_envs)}


def single_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.03,
) -> torch.Tensor:
  """Reward 1 when exactly one foot is on the ground.

  ``feet_air_time`` also rewards hopping with both feet; this term favors an
  alternating left/right stepping pattern. It is gated by the command so
  standing envs are not pushed to step in place.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  contact_time = sensor.data.current_contact_time
  assert contact_time is not None, "Contact sensor needs track_air_time=True."
  num_in_contact = torch.sum((contact_time > 0.0).float(), dim=1)
  reward = (num_in_contact == 1).float()
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward *= (total_command > command_threshold).float()
  return reward


def _foot_contact_forces(
  env: ManagerBasedRlEnv, sensor_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-contact normal force, tangential force and position, per foot.

  Returns ``(f_n, f_t, pos)`` with shapes ``[B, P, S]``, ``[B, P, S, 2]`` and
  ``[B, P, S, 3]`` for P feet and S contact slots. The sensor must use a
  non-netforce reduce so forces stay in the contact frame (normal first). On
  flat ground every contact shares the plane normal, hence the same tangent
  axes, so tangential forces can be summed across contacts.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None and data.pos is not None and data.found is not None
  num_feet = len(sensor.primary_names)
  b = data.force.shape[0]
  force = data.force.view(b, num_feet, -1, 3)
  found = (data.found.view(b, num_feet, -1) > 0).float()
  f_n = torch.abs(force[..., 0]) * found
  f_t = force[..., 1:] * found.unsqueeze(-1)
  pos = data.pos.view(b, num_feet, -1, 3)
  return f_n, f_t, pos


def zmp_margin_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize each loaded foot's center of pressure nearing its sole edge.

  On flat ground the ZMP equals the center of pressure (CoP); in single
  support it is the stance foot's CoP. Each loaded foot's CoP is expressed in
  its ankle frame and its distance to the nearest sole edge (``margin``) gives
  ``exp(-max(margin, 0) / std)``: ~0 when centered, 1 at or past the edge,
  where the foot starts to roll. The result is averaged over loaded feet so
  single and double support are penalized alike; penalizing single support
  only taught the policy to shuffle with both feet down instead of stepping.
  """
  asset: Entity = env.scene[asset_cfg.name]
  f_n, _, pos = _foot_contact_forces(env, sensor_name)
  foot_force = f_n.sum(dim=-1)  # [B, P]
  loaded = (foot_force > _MIN_FOOT_FORCE).float()
  single = (loaded.sum(dim=1) == 1).float()

  cop_w = (f_n.unsqueeze(-1) * pos).sum(dim=2) / foot_force.clamp(min=1e-6).unsqueeze(-1)
  foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # [B, P, 3]
  foot_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, P, 4]
  b, p = foot_force.shape
  cop_b = quat_apply_inverse(
    foot_quat.reshape(-1, 4), (cop_w - foot_pos).reshape(-1, 3)
  ).view(b, p, 3)

  bounds = torch.tensor(_SOLE_BOUNDS, device=cop_b.device)  # [P, 4]
  margin = torch.stack(
    (
      cop_b[..., 1] - bounds[:, 0],
      bounds[:, 1] - cop_b[..., 1],
      cop_b[..., 2] - bounds[:, 2],
      bounds[:, 3] - cop_b[..., 2],
    ),
    dim=-1,
  ).amin(dim=-1)  # [B, P]

  # Masked means instead of boolean indexing avoid a GPU sync every step.
  # zmp_margin_mm: stance foot during single support (the true ZMP margin).
  stance_mask = loaded * single.unsqueeze(1)
  env.extras["log"]["Metrics/zmp_margin_mm"] = (
    1000.0 * (margin * stance_mask).sum() / stance_mask.sum().clamp(min=1.0)
  )
  env.extras["log"]["Metrics/foot_cop_margin_mm"] = (
    1000.0 * (margin * loaded).sum() / loaded.sum().clamp(min=1.0)
  )
  env.extras["log"]["Metrics/single_support_frac"] = single.mean()
  penalty = loaded * torch.exp(-margin.clamp(min=0.0) / std)
  return penalty.sum(dim=1) / loaded.sum(dim=1).clamp(min=1.0)


def friction_cone_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  mu: float,
) -> torch.Tensor:
  """Penalize foot forces outside a friction cone of coefficient ``mu``.

  The simulator enforces its own (higher) friction, so this keeps the gait
  within a safety margin for floors with less grip. The cone is checked on
  each foot's net contact force, which is what decides whether the foot slips:
  ``sum_feet(relu(|F_t| - mu * F_n)) / sum_feet(F_n)``.
  """
  f_n, f_t, _ = _foot_contact_forces(env, sensor_name)
  foot_n = f_n.sum(dim=2)  # [B, P]
  foot_t = torch.norm(f_t.sum(dim=2), dim=-1)  # [B, P]
  excess = torch.relu(foot_t - mu * foot_n).sum(dim=1)

  loaded = (foot_n > _MIN_FOOT_FORCE).float()
  num_loaded = loaded.sum().clamp(min=1.0)
  ratio = foot_t / foot_n.clamp(min=1e-6)
  env.extras["log"]["Metrics/friction_ratio_mean"] = (ratio * loaded).sum() / num_loaded
  env.extras["log"]["Metrics/friction_cone_violation_frac"] = (
    (ratio > mu).float() * loaded
  ).sum() / num_loaded
  return excess / foot_n.sum(dim=1).clamp(min=1.0)


def stance_foot_tilt_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize sideways roll of loaded feet, i.e. standing on the sole edge.

  The sole width runs along the ankle body z axis, so the world-z component of
  that axis is ``sin(roll)``. Heel-to-toe pitch (push-off) is left free. The
  term averages ``|roll|`` (rad) over feet carrying weight.
  """
  asset: Entity = env.scene[asset_cfg.name]
  f_n, _, _ = _foot_contact_forces(env, sensor_name)
  foot_force = f_n.sum(dim=-1)  # [B, P]
  loaded = (foot_force > _MIN_FOOT_FORCE).float()
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, P, 4]
  b, p = quat.shape[:2]
  quat = quat.reshape(-1, 4)
  axes = torch.zeros(2, b * p, 3, device=quat.device)
  axes[0, :, 2] = 1.0  # sole width (body z)
  axes[1, :, 1] = 1.0  # heel to toe (body y)
  width_z = quat_apply(quat, axes[0])[:, 2].view(b, p)
  length_z = quat_apply(quat, axes[1])[:, 2].view(b, p)
  roll = torch.asin(width_z.clamp(-1.0, 1.0))
  pitch = torch.asin(length_z.clamp(-1.0, 1.0))

  num_loaded = loaded.sum().clamp(min=1.0)
  rad2deg = 57.29578
  env.extras["log"]["Metrics/stance_roll_deg"] = (
    rad2deg * (roll.abs() * loaded).sum() / num_loaded
  )
  env.extras["log"]["Metrics/stance_pitch_deg"] = (
    rad2deg * (pitch.abs() * loaded).sum() / num_loaded
  )
  env.extras["log"]["Metrics/stance_contacts"] = (
    ((f_n > 0.05).float().sum(dim=-1) * loaded).sum() / num_loaded
  )
  return (roll.abs() * loaded).sum(dim=1) / loaded.sum(dim=1).clamp(min=1.0)
