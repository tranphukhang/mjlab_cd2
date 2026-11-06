"""SRBD-MPC that follows the ZMP walker's plan, for position servos.

Layering (no RL):

    ZMP-MPC + capture point (``zmp_walker.py``)
      -> footholds, contact schedule, swing trajectories, CoM trajectory
    SRBD-MPC (``srbd_mpc.py``), this module
      -> ground reaction force and moment per stance foot that track the CoM
         plan and keep the body level, from the measured state
    joint targets for the servos
      -> whole-body IK of the plan, plus ``tau / kp`` with
         ``tau = bias - J^T [F; M]`` (``walk_feedback.BalanceFeedback``)

The ZMP plan also gives the force reference ``u_ref``: total force
``m (a_plan + g)`` split between the stance feet by the planned ZMP, with the
center of pressure of each foot at the ZMP (single support) or under the ZMP
along x (double support). R penalizes ``u - u_ref``, so with no tracking error
the SRBD-MPC returns the plan's forces and the robot does what the ZMP walker
alone does; tracking errors in CoM, roll and pitch then change the forces.

A position servo has no torque input: the wrench only reaches the hardware as
a target offset ``tau / kp``. That works as a feedforward at the servo
bandwidth (about 3 Hz for the STS3215), not as torque control.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from open_duck_mjlab.mpc.srbd_mpc import NX, MpcParams, SrbdMpc
from open_duck_mjlab.mpc.zmp_walker import GRAVITY, ZmpWalker

# Weights: roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, 1.
TRACK_Q = (30.0, 30.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0)
TRACK_R = (1e-2, 1.0)  # force, moment (penalize u - u_ref)


def composite_inertia(m: mujoco.MjModel, d: mujoco.MjData, base_id: int) -> np.ndarray:
  """Inertia of the whole robot about its CoM, base frame (3, 3).

  ``d`` must have kinematics and comPos done.
  """
  com = d.subtree_com[base_id]
  inertia = np.zeros((3, 3))
  for b in range(1, m.nbody):
    r_b = d.ximat[b].reshape(3, 3)
    r = d.xipos[b] - com
    inertia += r_b @ np.diag(m.body_inertia[b]) @ r_b.T
    inertia += m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
  r_base = d.xmat[base_id].reshape(3, 3)
  return r_base.T @ inertia @ r_base


def _rpy(r: np.ndarray) -> np.ndarray:
  return np.array([
    np.arctan2(r[2, 1], r[2, 2]),
    -np.arcsin(np.clip(r[2, 0], -1.0, 1.0)),
    np.arctan2(r[1, 0], r[0, 0]),
  ])


class SrbdTracker:
  def __init__(self, walker: ZmpWalker, params: MpcParams | None = None):
    """``walker.reset`` must have been called (its IK data holds the start pose)."""
    self.w = walker
    m, d = walker.m, walker.ik_data
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    inertia = composite_inertia(m, d, walker.base_id)
    p = params or MpcParams(q_weights=TRACK_Q, r_force=TRACK_R[0], r_moment=TRACK_R[1], horizon=10)
    p = dataclasses.replace(
      p, mass=float(m.body_subtreemass[walker.base_id]), inertia=tuple(np.diag(inertia))
    )
    if abs(p.dt - walker.dt) > 1e-9:
      raise ValueError("SRBD-MPC dt must match the ZMP walker dt")
    self.mpc = SrbdMpc(p)
    self.p = p
    self.u_ref0 = np.zeros(12)
    self.last_wrench = np.zeros(12)

  def _force_reference(self, contact: np.ndarray, feet: np.ndarray, acc: np.ndarray,
                       zmp: np.ndarray) -> np.ndarray:
    p = self.p
    force = p.mass * np.array([acc[0], acc[1], GRAVITY])
    u = np.zeros(12)
    if contact[0] and contact[1]:
      span = feet[0][1] - feet[1][1]
      lam = np.clip((zmp[1] - feet[1][1]) / span, 0.0, 1.0) if abs(span) > 1e-6 else 0.5
      share = (lam, 1.0 - lam)
    else:
      share = (float(contact[0]), float(contact[1]))
    for i in range(2):
      if not contact[i]:
        continue
      f = share[i] * force
      cop = zmp - feet[i][:2]
      if contact[0] and contact[1]:
        cop[1] = 0.0  # double support: each sole pushes under its own center line
      cop[0] = np.clip(cop[0], -p.cop_heel, p.cop_toe)
      cop[1] = np.clip(cop[1], -p.cop_half_width, p.cop_half_width)
      u[3 * i : 3 * i + 3] = f
      u[6 + 3 * i : 9 + 3 * i] = np.cross(np.array([cop[0], cop[1], 0.0]), f)
    return u

  def solve(
    self,
    t: float,
    quat: np.ndarray,
    gyro_body: np.ndarray,
    com: np.ndarray,
    vcom: np.ndarray,
  ) -> np.ndarray:
    """Wrench ``[F_L, F_R, M_L, M_R]`` (12,, world) for the step starting at t.

    Call right after ``walker.step(t, ...)``. ``quat``/``gyro_body`` are the
    IMU orientation and body-frame angular rate; ``com``/``vcom`` the CoM
    estimate (world).
    """
    w, p = self.w, self.p
    n = p.horizon
    states = w.plan_states
    if states.shape[1] < n:
      raise ValueError("ZMP walker horizon is shorter than the SRBD horizon")
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat)
    r = r.reshape(3, 3)
    x0 = np.concatenate([_rpy(r), com, r @ gyro_body, vcom, [1.0]])

    z_ref = float(w.debug["com"][2])
    x_ref = np.zeros((n, NX))
    x_ref[:, 3] = states[0, :n, 0]
    x_ref[:, 4] = states[1, :n, 0]
    x_ref[:, 5] = z_ref
    x_ref[:, 9] = states[0, :n, 1]
    x_ref[:, 10] = states[1, :n, 1]
    x_ref[:, 12] = 1.0

    contact, feet = w.schedule(t, n)
    foot_rel = feet - x_ref[:, None, 3:6]
    u_ref = np.zeros((n, 12))
    for k in range(n):
      # Input k acts over [t + k dt, t + (k+1) dt]: use the plan at its start.
      st = states[:, k - 1] if k > 0 else states[:, 0]
      zmp = st[:, 0] - st[:, 2] / w.omega**2
      u_ref[k] = self._force_reference(contact[k], feet[k], st[:, 2], zmp)
    self.u_ref0 = u_ref[0]
    wrench = self.mpc.solve(x0, x_ref, contact, foot_rel, np.zeros(2), u_ref)
    if not self.mpc.last_status.startswith("solved"):
      wrench = u_ref[0]  # fall back to the plan's forces
    self.last_wrench = wrench
    return wrench
