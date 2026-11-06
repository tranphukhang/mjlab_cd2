"""Linear-inverted-pendulum ZMP MPC (Wieber 2006 style) for CoM planning.

Per horizontal axis the CoM is a triple integrator driven by jerk ``u``:

    c_{k+1} = A c_k + B u_k,   c = (position, velocity, acceleration)
    zmp_k   = position_k - (h / g) acceleration_k

Over an ``N``-step horizon the QP

    min  sum q_z (zmp - zmp_ref)^2 + q_v (vel - vel_ref)^2 + r u^2
    s.t. zmp_min <= zmp <= zmp_max   (inside the support polygon)

is solved in condensed form (decision variables: the ``N`` jerks), with
OSQP. The support polygon is the stance sole in single support and the box
spanning both soles in double support (exact in the heading frame when the
feet are side by side; conservative enough for straight walking).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

GRAVITY = 9.81


@dataclass
class ZmpMpcParams:
  horizon: int = 80
  dt: float = 0.02
  com_height: float = 0.151
  q_zmp: float = 1.0
  q_vel: float = 1e-3
  r_jerk: float = 1e-6


class ZmpMpc:
  """One instance per axis (heading x, heading y)."""

  def __init__(self, params: ZmpMpcParams | None = None):
    self.p = params or ZmpMpcParams()
    n, dt, h = self.p.horizon, self.p.dt, self.p.com_height
    a = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    b = np.array([dt**3 / 6.0, 0.5 * dt * dt, dt])
    c_zmp = np.array([1.0, 0.0, -h / GRAVITY])
    # Condensed prediction: state_k+1 = Phi_k x0 + Gam_k U.
    self.phi = np.zeros((n, 3, 3))
    self.gam = np.zeros((n, 3, n))
    ak = np.eye(3)
    for k in range(n):
      ak = a @ ak
      self.phi[k] = ak
      for j in range(k + 1):
        self.gam[k, :, j] = np.linalg.matrix_power(a, k - j) @ b
    self.z_x = np.einsum("i,kij->kj", c_zmp, self.phi)  # (n, 3)
    self.z_u = np.einsum("i,kij->kj", c_zmp, self.gam)  # (n, n)
    self.v_x = self.phi[:, 1, :]
    self.v_u = self.gam[:, 1, :]
    p = self.p
    hess = p.q_zmp * self.z_u.T @ self.z_u + p.q_vel * self.v_u.T @ self.v_u
    hess += p.r_jerk * np.eye(n)
    self._p = sp.csc_matrix(np.triu(2.0 * hess))
    self._a = sp.csc_matrix(self.z_u)
    self._solver: osqp.OSQP | None = None
    self.last_status = ""

  def solve(
    self,
    x0: np.ndarray,
    zmp_ref: np.ndarray,
    zmp_min: np.ndarray,
    zmp_max: np.ndarray,
    vel_ref: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (jerk sequence (N,), predicted states (N, 3))."""
    p = self.p
    zf = self.z_x @ x0
    vf = self.v_x @ x0
    q = 2.0 * (p.q_zmp * self.z_u.T @ (zf - zmp_ref) + p.q_vel * self.v_u.T @ (vf - vel_ref))
    lo = zmp_min - zf
    hi = zmp_max - zf
    if self._solver is None:
      self._solver = osqp.OSQP()
      self._solver.setup(
        self._p, q, self._a, lo, hi, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
        max_iter=4000, polishing=False, warm_starting=True,
      )
    else:
      self._solver.update(q=q, l=lo, u=hi)
    res = self._solver.solve()
    self.last_status = res.info.status
    u = res.x if res.x is not None and np.all(np.isfinite(res.x)) else np.zeros(p.horizon)
    states = self.phi @ x0 + self.gam @ u
    return u, states
