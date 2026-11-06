"""Sensor-side pieces of the model-based walker, usable on the real robot.

Everything here needs only what the Open Duck hardware measures (IMU
orientation and gyro, servo joint angles) plus the MuJoCo model:

- ``ComEstimator``: leg odometry. The stance foot is assumed not to slip, so
  the base position follows from the IMU orientation and the joint angles;
  the CoM and its (filtered) velocity follow from the model.
- ``BalanceFeedback``: offsets added to the IK joint targets before they go
  to the position servos:
  - gyro damping: stance legs are commanded to tilt the base against its
    measured angular velocity (like the gyro feedback of Darwin-OP-style walk
    engines, but mapped through the leg Jacobians so no per-joint signs are
    needed);
  - load compensation: a position servo only makes torque from tracking
    error, ``tau = kp (q_cmd - q)``, so the joint torque the planned motion
    needs (gravity and the ground force on the stance legs) is commanded as
    a position offset ``tau / kp``. Without it the stance knee sags under
    load.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

GRAVITY = 9.81


class ComEstimator:
  def __init__(
    self,
    model: mujoco.MjModel,
    qadr: np.ndarray,
    site_ids: list[int],
    base_id: int,
    dt: float,
    vel_filter: float = 0.5,
  ):
    self.m = model
    self.d = mujoco.MjData(model)
    self.qadr = qadr
    self.site_ids = site_ids
    self.base_id = base_id
    self.dt = dt
    self.alpha = vel_filter  # weight of the new finite-difference sample
    self.foot_world: np.ndarray | None = None
    self.contact = np.array([True, True])
    self.com: np.ndarray | None = None
    self.vcom = np.zeros(3)

  def reset(self, feet_world: np.ndarray) -> None:
    self.foot_world = np.array(feet_world, dtype=float)
    self.foot_z0 = self.foot_world[:, 2].copy()
    self.contact = np.array([True, True])
    self.com = None
    self.vcom = np.zeros(3)

  def update(
    self, quat: np.ndarray, q_joints: np.ndarray, contact: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (CoM, CoM velocity) in the world frame."""
    d = self.d
    d.qpos[:] = 0.0
    d.qpos[3:7] = quat
    d.qpos[self.qadr] = q_joints
    mujoco.mj_kinematics(self.m, d)
    mujoco.mj_comPos(self.m, d)
    sites = np.array([d.site_xpos[s].copy() for s in self.site_ids])
    assert self.foot_world is not None, "call reset() first"
    anchors = [i for i in range(2) if self.contact[i] and contact[i]]
    if not anchors:
      anchors = [i for i in range(2) if self.contact[i]] or [0, 1]
    base = np.mean([self.foot_world[i] - sites[i] for i in anchors], axis=0)
    for i in range(2):
      if contact[i] and not self.contact[i]:  # touchdown: record where it landed
        self.foot_world[i] = base + sites[i]
        # Flat ground: a landed sole is at its start height, so small height
        # errors do not pile up step after step.
        self.foot_world[i][2] = self.foot_z0[i]
    self.contact = np.array(contact, dtype=bool)
    com = base + d.subtree_com[self.base_id]
    if self.com is not None:
      v = (com - self.com) / self.dt
      self.vcom = (1.0 - self.alpha) * self.vcom + self.alpha * v
    self.com = com
    return com.copy(), self.vcom.copy()


@dataclass
class FeedbackGains:
  gyro: tuple[float, float] = (0.0, 0.0)  # (roll, pitch) rad per rad/s
  load: float = 1.0  # 0 off, 1 full load compensation


class BalanceFeedback:
  def __init__(
    self,
    model: mujoco.MjModel,
    dadr: np.ndarray,
    site_ids: list[int],
    gains: FeedbackGains | None = None,
  ):
    self.m = model
    self.dadr = dadr
    self.legs = (dadr[:5], dadr[5:])
    self.site_ids = site_ids
    self.g = gains or FeedbackGains()
    acts = range(model.nu)
    self.kp = np.array([model.actuator_gainprm[a, 0] for a in acts])
    self.mass = float(model.body_subtreemass[1])
    self._jacp = np.zeros((3, model.nv))
    self._jacr = np.zeros((3, model.nv))

  def offsets(
    self,
    d_plan: mujoco.MjData,
    contact: np.ndarray,
    gyro_body: np.ndarray,
    com_acc: np.ndarray,
    zmp: np.ndarray,
    feet: np.ndarray,
    wrench: np.ndarray | None = None,
  ) -> np.ndarray:
    """Joint target offsets (10,), computed at the planned configuration.

    ``d_plan`` must hold the IK solution with kinematics and comPos done.
    ``wrench`` (12,: F_L, F_R, M_L, M_R, world frame, force on the robot) is
    the ground reaction of a force MPC; without it the load term uses the
    planned CoM acceleration split between the feet by the ZMP.
    """
    m = self.m
    out = np.zeros(10)
    jacs = []
    for i in range(2):
      mujoco.mj_jacSite(m, d_plan, self._jacp, self._jacr, self.site_ids[i])
      jacs.append((self._jacp.copy(), self._jacr.copy()))

    if any(self.g.gyro):
      dtheta = -np.array([self.g.gyro[0] * gyro_body[0], self.g.gyro[1] * gyro_body[1], 0.0])
      for i in range(2):
        if not contact[i]:
          continue
        jp, jr = jacs[i]
        j_leg = np.vstack([jp[:, self.legs[i]], jr[:, self.legs[i]]])
        j_rot = np.vstack([jp[:, 3:6], jr[:, 3:6]])  # base angular dofs (body frame)
        out[5 * i : 5 * i + 5] += -np.linalg.pinv(j_leg) @ (j_rot @ dtheta)

    if self.g.load > 0.0:
      bias = np.zeros(m.nv)
      qvel = d_plan.qvel.copy()
      d_plan.qvel[:] = 0.0
      mujoco.mj_rne(m, d_plan, 0, bias)  # gravity only (zero velocity)
      d_plan.qvel[:] = qvel
      force = self.mass * np.array([com_acc[0], com_acc[1], GRAVITY])
      share = np.array(contact, dtype=float)
      if contact[0] and contact[1]:
        span = feet[0][1] - feet[1][1]
        lam = np.clip((zmp[1] - feet[1][1]) / span, 0.0, 1.0) if abs(span) > 1e-6 else 0.5
        share = np.array([lam, 1.0 - lam])
      for i in range(2):
        leg = self.legs[i]
        jp, jr = jacs[i][0][:, leg], jacs[i][1][:, leg]
        if wrench is None:
          tau = bias[leg] - jp.T @ (share[i] * force)
        else:
          tau = bias[leg] - jp.T @ wrench[3 * i : 3 * i + 3] - jr.T @ wrench[6 + 3 * i : 9 + 3 * i]
        out[5 * i : 5 * i + 5] += self.g.load * tau / self.kp[5 * i : 5 * i + 5]
    return out
