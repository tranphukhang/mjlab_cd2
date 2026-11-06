"""Convex single-rigid-body MPC (force and moment) for the Open Duck biped.

The whole robot is treated as one rigid body of mass ``m`` and inertia ``I``
about its center of mass, pushed by a ground reaction force ``F_i`` and moment
``M_i`` at each foot (Di Carlo et al., IROS 2018; Li and Nguyen, CDC 2021; the
baseline MPC of Kamohara et al., ICRA 2026). Linearizing the rotation about
the current yaw gives a linear time-varying model

    d/dt [Theta, p, omega, v, 1] = A x + B u,   u = [F_L, F_R, M_L, M_R]

    Theta_dot = Rz(yaw)^T omega            (small roll and pitch)
    p_dot     = v
    omega_dot = I_w^-1 sum_i (r_i x F_i + M_i)
    v_dot     = sum_i F_i / m + g

with ``r_i`` the vector from the CoM to foot ``i``. Euler discretization over
an ``N``-step horizon turns tracking a reference into a sparse QP solved with
OSQP. Per stance foot the QP enforces a friction pyramid, a vertical force
range and a center-of-pressure box: the Open Duck foot is a narrow sole with
no ankle roll joint, so the moment about the foot axes must keep the CoP on
the sole. Swing feet get zero force and moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

NX = 13  # roll, pitch, yaw, p (3), omega (3, world), v (3), constant 1
NU = 12  # F_left, F_right, M_left, M_right, world frame
GRAVITY = 9.81


@dataclass
class MpcParams:
  horizon: int = 20
  dt: float = 0.02
  mass: float = 1.5
  # Body-frame inertia about the CoM at the home pose (diagonal terms).
  inertia: tuple[float, float, float] = (0.0123, 0.0078, 0.0065)
  mu: float = 0.7
  f_max: float = 30.0
  # Center-of-pressure box relative to the foot site, foot frame (m). The
  # sole spans -0.030..+0.061 m along x and +-0.012 m along y; keep a margin.
  cop_toe: float = 0.05
  cop_heel: float = 0.022
  cop_half_width: float = 0.008
  # |M_z| <= coeff * F_z (torsional friction of a small sole).
  yaw_moment_coeff: float = 0.005
  # Tracking weights: roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, 1.
  q_weights: tuple[float, ...] = (
    200.0, 200.0, 50.0, 20.0, 20.0, 400.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 0.0
  )
  r_force: float = 1e-4
  r_moment: float = 1e-2


def _skew(v: np.ndarray) -> np.ndarray:
  return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_z(yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SrbdMpc:
  def __init__(self, params: MpcParams | None = None):
    self.p = params or MpcParams()
    self._inertia_body = np.diag(self.p.inertia)
    self.last_solve_ms = 0.0
    self.last_status = ""

  def _discrete_dynamics(
    self, yaw: float, foot_rel: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    p = self.p
    rz = _rot_z(yaw)
    inertia_inv = np.linalg.inv(rz @ self._inertia_body @ rz.T)
    a = np.zeros((NX, NX))
    a[0:3, 6:9] = rz.T
    a[3:6, 9:12] = np.eye(3)
    a[11, 12] = -GRAVITY
    b = np.zeros((NX, NU))
    for i in range(2):
      b[6:9, 3 * i : 3 * i + 3] = inertia_inv @ _skew(foot_rel[i])
      b[9:12, 3 * i : 3 * i + 3] = np.eye(3) / p.mass
      b[6:9, 6 + 3 * i : 9 + 3 * i] = inertia_inv
    return np.eye(NX) + a * p.dt, b * p.dt

  def solve(
    self,
    x0: np.ndarray,
    x_ref: np.ndarray,
    contact: np.ndarray,
    foot_rel: np.ndarray,
    foot_yaw: np.ndarray,
  ) -> np.ndarray:
    """Return the first-step input ``[F_L, F_R, M_L, M_R]`` (12,).

    Args:
      x0: current state (13,).
      x_ref: reference states x_1..x_N (N, 13).
      contact: stance flags per step and foot, (N, 2) bool, (left, right).
      foot_rel: foot position minus CoM per step and foot, (N, 2, 3), world.
      foot_yaw: yaw of each foot's sole frame (2,).
    """
    p = self.p
    n = p.horizon
    nxv, nuv = NX * n, NU * n
    nvar = nxv + nuv

    # Cost 1/2 z^T P z + q^T z.
    q_diag = np.tile(np.asarray(p.q_weights), n)
    r_one = np.array([p.r_force] * 6 + [p.r_moment] * 6)
    p_mat = sp.diags(np.concatenate([q_diag, np.tile(r_one, n)]) * 2.0, format="csc")
    q_vec = np.concatenate([-2.0 * q_diag * x_ref.reshape(-1), np.zeros(nuv)])

    # Dynamics: x_{k+1} - A_k x_k - B_k u_k = 0 (x_0 is given).
    rows, cols, vals = [], [], []
    beq = np.zeros(nxv)

    def put(r0: int, c0: int, block: np.ndarray) -> None:
      rr, cc = np.nonzero(np.abs(block) > 0.0)
      rows.extend(rr + r0)
      cols.extend(cc + c0)
      vals.extend(block[rr, cc])

    for k in range(n):
      ad, bd = self._discrete_dynamics(x_ref[k, 2], foot_rel[k])
      put(NX * k, NX * k, np.eye(NX))
      put(NX * k, nxv + NU * k, -bd)
      if k == 0:
        beq[0:NX] = ad @ x0
      else:
        put(NX * k, NX * (k - 1), -ad)
    a_eq = sp.csc_matrix((vals, (rows, cols)), shape=(nxv, nvar))

    # Per step and foot: 6 box rows on u, 4 friction rows, 6 CoP/moment rows.
    rows, cols, vals = [], [], []
    lo, hi = [], []
    row = 0
    inf = np.inf
    for k in range(n):
      for i in range(2):
        f0 = nxv + NU * k + 3 * i  # Fx, Fy, Fz
        m0 = nxv + NU * k + 6 + 3 * i  # Mx, My, Mz
        stance = bool(contact[k, i])
        # Box bounds on the six wrench components.
        for j in range(3):
          rows.append(row)
          cols.append(f0 + j)
          vals.append(1.0)
          if not stance:
            lo.append(0.0)
            hi.append(0.0)
          elif j == 2:
            lo.append(0.0)
            hi.append(p.f_max)
          else:
            lo.append(-inf)
            hi.append(inf)
          row += 1
        for j in range(3):
          rows.append(row)
          cols.append(m0 + j)
          vals.append(1.0)
          lo.append(-inf if stance else 0.0)
          hi.append(inf if stance else 0.0)
          row += 1
        # Friction pyramid |Fx|, |Fy| <= mu Fz.
        for j in range(2):
          for sign in (1.0, -1.0):
            rows += [row, row]
            cols += [f0 + j, f0 + 2]
            vals += [sign, -p.mu]
            lo.append(-inf)
            hi.append(0.0)
            row += 1
        # CoP box in the foot frame: M_f = Rz(yaw_f)^T M. A CoP offset (cx, cy)
        # gives M_fx = cy Fz and M_fy = -cx Fz.
        rz_t = _rot_z(foot_yaw[i]).T
        limits = (
          (0, p.cop_half_width, p.cop_half_width),  # -w Fz <= M_fx <= w Fz
          (1, p.cop_toe, p.cop_heel),  # -toe Fz <= M_fy <= heel Fz
          (2, p.yaw_moment_coeff, p.yaw_moment_coeff),
        )
        for axis, neg, pos in limits:
          for sign, lim in ((1.0, pos), (-1.0, neg)):
            for j in range(3):
              coeff = sign * rz_t[axis, j]
              if coeff != 0.0:
                rows.append(row)
                cols.append(m0 + j)
                vals.append(coeff)
            rows.append(row)
            cols.append(f0 + 2)
            vals.append(-lim)
            lo.append(-inf)
            hi.append(0.0)
            row += 1
    a_in = sp.csc_matrix((vals, (rows, cols)), shape=(row, nvar))

    a_all = sp.vstack([a_eq, a_in], format="csc")
    l_all = np.concatenate([beq, np.asarray(lo)])
    u_all = np.concatenate([beq, np.asarray(hi)])

    solver = osqp.OSQP()
    solver.setup(
      p_mat,
      q_vec,
      a_all,
      l_all,
      u_all,
      verbose=False,
      eps_abs=1e-5,
      eps_rel=1e-5,
      max_iter=8000,
      polishing=True,
    )
    res = solver.solve()
    self.last_status = res.info.status
    self.last_solve_ms = 1e3 * (res.info.setup_time + res.info.solve_time)
    if res.x is None or not np.all(np.isfinite(res.x)):
      return np.zeros(NU)
    return res.x[nxv : nxv + NU].copy()
