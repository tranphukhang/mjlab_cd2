"""Left/right mirror symmetry for rsl_rl symmetry augmentation.

The mirror plane is the robot's sagittal plane (body y = 0). Forward kinematics
of the Open Duck model shows every left/right joint pair mirrors as
``q_R = -q_L`` (hip yaw, hip roll, hip pitch, knee and ankle alike), so joint
vectors mirror by swapping the left/right halves and negating. Base vectors
mirror as polar (``x, -y, z``) or axial (``-x, y, -z``) vectors, and the gait
clock shifts by half a cycle: the right-swing window of the contact schedule
maps onto the left-swing window.

Every mirror here is a signed permutation, built per observation group from the
observation manager's term names, so a changed term layout fails loudly instead
of being mirrored wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_NUM_JOINTS = 10  # L_hip_yaw..L_ankle, then R_hip_yaw..R_ankle.
_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)


def _joint_mirror() -> tuple[list[int], list[float]]:
  half = _NUM_JOINTS // 2
  perm = list(range(half, _NUM_JOINTS)) + list(range(half))
  return perm, [-1.0] * _NUM_JOINTS


def _term_mirror(name: str, dim: int) -> tuple[list[int], list[float]]:
  """Signed permutation (indices, signs) mirroring one observation term."""
  if name in ("base_lin_vel", "projected_gravity"):
    return [0, 1, 2], list(_POLAR)
  if name == "base_ang_vel":
    return [0, 1, 2], list(_AXIAL)
  if name in ("joint_pos", "joint_vel", "actions"):
    assert dim == _NUM_JOINTS, f"{name} has {dim} dims"
    return _joint_mirror()
  if name == "command":  # (vx, vy, wz)
    return [0, 1, 2], [1.0, -1.0, -1.0]
  if name == "gait_clock":  # (sin, cos) of phase -> phase + 0.5
    return [0, 1], [-1.0, -1.0]
  if name in ("foot_height", "foot_air_time", "foot_contact"):  # (left, right)
    return [1, 0], [1.0, 1.0]
  if name == "foot_contact_forces":  # (left xyz, right xyz), world frame
    return [3, 4, 5, 0, 1, 2], list(_POLAR) * 2
  raise KeyError(f"No mirror rule for observation term '{name}'.")


_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _group_mirror(
  env: ManagerBasedRlEnv, group: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  key = (id(env), group)
  if key not in _CACHE:
    om = env.observation_manager
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for name, shape in zip(
      om.active_terms[group], om.group_obs_term_dim[group], strict=True
    ):
      dim = shape[0]
      p, s = _term_mirror(name, dim)
      assert len(p) == dim, f"mirror of {name} covers {len(p)} of {dim} dims"
      perm += [offset + i for i in p]
      sign += s
      offset += dim
    _CACHE[key] = (
      torch.tensor(perm, device=device),
      torch.tensor(sign, device=device),
    )
  return _CACHE[key]


def mirror_obs(env: ManagerBasedRlEnv, group: str, obs: torch.Tensor) -> torch.Tensor:
  perm, sign = _group_mirror(env, group, obs.device)
  return obs[..., perm] * sign


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  perm, sign = _joint_mirror()
  return actions[..., perm] * torch.tensor(sign, device=actions.device)


def mirror_augmentation(
  env, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """rsl_rl ``data_augmentation_func``: append mirrored samples to the batch."""
  base = getattr(env, "unwrapped", env)
  obs_aug = None
  if obs is not None:
    mirrored = TensorDict(
      {k: mirror_obs(base, k, v) for k, v in obs.items()},
      batch_size=obs.batch_size,
      device=obs.device,
    )
    obs_aug = torch.cat((obs, mirrored), dim=0)
  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
  return obs_aug, actions_aug
