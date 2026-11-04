"""Extra MDP terms for the gait-first Open Duck curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


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
