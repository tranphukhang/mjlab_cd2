"""ZMP-MPC walking pattern generator with whole-body IK for position servos.

Pipeline, run every control tick (50 Hz, like the RL policy):

1. Footstep plan from the velocity command on a fixed gait clock (double
   support, right swing, double support, left swing).
2. ZMP MPC (``zmp_mpc.py``) per axis: CoM jerk that keeps the ZMP inside the
   support polygon while tracking a ZMP reference near the sole centers.
3. Balance feedback from the measured CoM (estimated on the robot from the
   IMU and joint encoders, see ``walk_feedback.py``):
   - DCM feedback shifts the commanded CoM (Kajita-style ZMP feedback through
     position control);
   - capture-point stepping moves the landing point of the swinging foot by
     the DCM error. The leg has no ankle roll, so foot placement is the main
     way to balance sideways.
4. Swing foot trajectories between footsteps.
5. Whole-body IK (damped least squares on MuJoCo Jacobians): CoM, base
   orientation and both foot poses to joint angles, sent to the STS3215
   position servos (``walk_feedback.py`` adds gyro damping and load
   compensation on top).

Walking is straight along +x; turning is not planned yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from open_duck_mjlab.mpc.zmp_mpc import ZmpMpc, ZmpMpcParams

LEGS = (
  ("L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle"),
  ("R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"),
)
FOOT_SITES = ("L6_IK", "R6_IK")
BASE_BODY = "base_link"
GRAVITY = 9.81


@dataclass
class WalkParams:
  period: float = 0.4  # one gait cycle = two steps
  double_support: float = 0.1  # fraction of the period, per DS phase
  swing_height: float = 0.02
  # ZMP reference relative to the foot site (x forward, y inward). The sole
  # spans -0.030..+0.061 m along x and +-0.012 m along y around the site.
  zmp_offset_x: float = 0.005
  zmp_offset_in: float = 0.004
  # ZMP box around the foot site: keeps a few mm from the sole edges.
  zmp_box_back: float = 0.022
  zmp_box_front: float = 0.045
  zmp_box_half_width: float = 0.009
  # Seconds of double support that move the ZMP onto the first stance foot.
  start_shift: float = 0.4
  # Extra CoM height while walking. The home pose bends the knees to 1.415 of
  # 1.571 rad and needs about 0.015 m to leave room for the swing; the
  # crouch10 start pose (knees at 40 deg) needs none.
  com_raise: float = 0.0
  com_raise_time: float = 0.5
  # DCM feedback on the commanded CoM, (x, y). 0 runs the plan open loop.
  dcm_gain: tuple[float, float] = (0.0, 0.0)
  # Capture-point stepping: landing offset = gain * DCM error, (x, y), with
  # a per-axis limit and a minimum distance between the feet along y.
  step_gain: tuple[float, float] = (0.0, 0.3)
  step_limit: tuple[float, float] = (0.04, 0.03)
  min_step_width: float = 0.12
  # At each new step, shift the whole plan (CoM state and footsteps) so the
  # stance foot matches its measured position, along x only: the plan then
  # follows where the robot really is instead of drifting away from it.
  reanchor_x: bool = True


@dataclass
class IkWeights:
  com: float = 100.0
  foot_pos: float = 100.0
  foot_rot: float = 30.0
  base_rot: tuple[float, float, float] = (0.05, 5.0, 5.0)  # roll, pitch, yaw
  posture: float = 1e-3  # pull toward the home joint angles
  damping: float = 1e-4
  iterations: int = 8


@dataclass
class _Plan:
  zref: np.ndarray
  zmin: np.ndarray
  zmax: np.ndarray
  feet: list[list[np.ndarray]] = field(default_factory=list)


def _smoothstep(s: float) -> float:
  s = min(max(s, 0.0), 1.0)
  return 3.0 * s * s - 2.0 * s**3


class ZmpWalker:
  def __init__(
    self,
    model: mujoco.MjModel,
    params: WalkParams | None = None,
    mpc_params: ZmpMpcParams | None = None,
    ik: IkWeights | None = None,
  ):
    self.m = model
    self.p = params or WalkParams()
    self.mpc_params = mpc_params or ZmpMpcParams()
    self.mpc: list[ZmpMpc] = []
    self.dt = self.mpc_params.dt
    self.ik_w = ik or IkWeights()
    self.base_id = model.body(BASE_BODY).id
    self.site_ids = [model.site(s).id for s in FOOT_SITES]
    names = [j for leg in LEGS for j in leg]
    jids = [model.joint(n).id for n in names]
    self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
    self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
    self.qmin = model.jnt_range[jids, 0]
    self.qmax = model.jnt_range[jids, 1]
    self.ik_data = mujoco.MjData(model)
    self.walk_start: float | None = None
    self.t_first: float | None = None
    self.com_state = np.zeros((2, 3))  # per axis: position, velocity, acceleration
    self.feet0 = np.zeros((2, 3))
    self.foot_rot0 = np.tile(np.eye(3), (2, 1, 1))
    self.committed: list[list[np.ndarray]] = [[], []]
    self.nominal_landing: dict[int, np.ndarray] = {}
    self.q_home = np.zeros(len(names))
    self.com_height = 0.0
    self.omega = 1.0
    self.vx = 0.0
    self.vy = 0.0
    self.contact = np.array([True, True])
    self.debug: dict = {}
    self.last_plan: _Plan | None = None
    self.plan_states = np.zeros((2, 0, 3))  # per axis, CoM (pos, vel, acc) at t + (k+1) dt

  # ------------------------------------------------------------------ setup
  def reset(self, d: mujoco.MjData) -> None:
    """Start from the measured standing pose (feet on the floor)."""
    mujoco.mj_forward(self.m, d)
    self.ik_data.qpos[:] = d.qpos
    self.ik_data.qvel[:] = 0.0
    self.q_home = d.qpos[self.qadr].copy()
    self.feet0 = np.array([d.site_xpos[s].copy() for s in self.site_ids])
    # Keep each sole's roll and pitch from the start pose, with zero yaw.
    for i, s in enumerate(self.site_ids):
      r = d.site_xmat[s].reshape(3, 3)
      yaw = np.arctan2(r[1, 0], r[0, 0])
      c, sn = np.cos(-yaw), np.sin(-yaw)
      self.foot_rot0[i] = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ r
    com = d.subtree_com[self.base_id].copy()
    self.com_height = com[2]
    ground = self.feet0[:, 2].mean()
    h = com[2] + self.p.com_raise - ground
    self.omega = np.sqrt(GRAVITY / h)
    self.mpc = [
      ZmpMpc(ZmpMpcParams(**{**self.mpc_params.__dict__, "com_height": h})) for _ in range(2)
    ]
    self.com_state[:] = 0.0
    self.com_state[:, 0] = com[:2]
    self.committed = [[self.feet0[0].copy()], [self.feet0[1].copy()]]
    self.nominal_landing = {}
    self.walk_start = None
    self.t_first = None
    self._last_step = -3

  # ------------------------------------------------------------- footsteps
  def _zmp_point(self, foot: np.ndarray, side: int) -> np.ndarray:
    inward = -1.0 if side == 0 else 1.0  # left foot is at +y
    return foot[:2] + np.array([self.p.zmp_offset_x, inward * self.p.zmp_offset_in])

  def _box(self, feet: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around the given soles (foot site positions)."""
    p = self.p
    c = np.array(feet)[:, :2]
    lo = c.min(0) - np.array([p.zmp_box_back, p.zmp_box_half_width])
    hi = c.max(0) + np.array([p.zmp_box_front, p.zmp_box_half_width])
    return lo, hi

  def _phase(self, t: float) -> tuple[int, float]:
    """(step index, time into the step); -2 standing, -1 initial weight shift."""
    if self.walk_start is None:
      return -2, 0.0
    tau = t - self.walk_start
    if tau < self.p.start_shift:
      return -1, tau
    tau -= self.p.start_shift
    half = 0.5 * self.p.period
    j = int(tau // half)
    return j, tau - j * half

  @staticmethod
  def _stance(j: int) -> int:
    """Step j: right foot swings on even steps (left stance), left on odd."""
    return 0 if j % 2 == 0 else 1

  def _next_landing(self, feet: list[list[np.ndarray]], j: int) -> None:
    """Append both feet's positions at the start of step j + 1."""
    stance = self._stance(j)
    swing = 1 - stance
    half = 0.5 * self.p.period
    new = feet[stance][j].copy()
    new[0] += self.vx * half
    new[1] += (self.feet0[swing][1] - self.feet0[stance][1]) + self.vy * half
    new[2] = self.feet0[swing][2]
    feet[stance].append(feet[stance][j].copy())
    feet[swing].append(new)

  def _footsteps(self, t: float, t_end: float) -> list[list[np.ndarray]]:
    """feet[f][j]: position of foot f at the start of step j.

    The landing of the step in progress is committed when the step starts
    (capture-point stepping may still move it); later footsteps follow the
    current command.
    """
    j_now, _ = self._phase(t)
    while len(self.committed[0]) < j_now + 2:
      j = len(self.committed[0]) - 1
      self._next_landing(self.committed, j)
      self.nominal_landing[j + 1] = self.committed[1 - self._stance(j)][j + 1].copy()
    feet = [list(self.committed[0]), list(self.committed[1])]
    j_end, _ = self._phase(t_end)
    while len(feet[0]) < j_end + 2:
      self._next_landing(feet, len(feet[0]) - 1)
    return feet

  def _adjust_landing(self, t: float, dcm_err: np.ndarray) -> None:
    """Capture-point stepping: move the current swing landing by the DCM error."""
    p = self.p
    j, local = self._phase(t)
    if j < 0 or not any(p.step_gain) or j + 1 not in self.nominal_landing:
      return
    stance = self._stance(j)
    swing = 1 - stance
    adj = np.clip(np.asarray(p.step_gain) * dcm_err, -np.asarray(p.step_limit), p.step_limit)
    land = self.nominal_landing[j + 1].copy()
    land[:2] += adj
    stance_y = self.committed[stance][j][1]
    if swing == 1:  # right foot lands to the right of the left foot
      land[1] = min(land[1], stance_y - p.min_step_width)
    else:
      land[1] = max(land[1], stance_y + p.min_step_width)
    self.committed[swing][j + 1] = land

  def _plan(self, t: float, n: int) -> _Plan:
    p = self.p
    ds = p.double_support * p.period
    feet = self._footsteps(t, t + n * self.dt)
    zref = np.zeros((n, 2))
    zmin = np.zeros((n, 2))
    zmax = np.zeros((n, 2))
    for k in range(n):
      j, local = self._phase(t + (k + 1) * self.dt)
      if j < 0:
        pts = [self._zmp_point(self.feet0[0], 0), self._zmp_point(self.feet0[1], 1)]
        mid = 0.5 * (pts[0] + pts[1])
        s = _smoothstep(local / p.start_shift) if j == -1 else 0.0
        zref[k] = (1 - s) * mid + s * pts[self._stance(0)]
        zmin[k], zmax[k] = self._box(list(self.feet0))
        continue
      stance = self._stance(j)
      st = self._zmp_point(feet[stance][j], stance)
      if local < ds:
        # Weight moves from the previous stance foot (now at feet[swing][j]).
        old = self._zmp_point(feet[1 - stance][j], 1 - stance)
        s = _smoothstep(local / ds) if j > 0 else 1.0
        zref[k] = (1 - s) * old + s * st
        zmin[k], zmax[k] = self._box([feet[0][j], feet[1][j]])
      else:
        zref[k] = st
        zmin[k], zmax[k] = self._box([feet[stance][j]])
    return _Plan(zref, zmin, zmax, feet)

  def _feet_at(self, t: float, feet: list[list[np.ndarray]]) -> np.ndarray:
    p = self.p
    j, local = self._phase(t)
    self.contact = np.array([True, True])
    if j < 0:
      return self.feet0.copy()
    half = 0.5 * p.period
    ds = p.double_support * p.period
    stance = self._stance(j)
    swing = 1 - stance
    out = np.zeros((2, 3))
    out[stance] = feet[stance][j]
    if local < ds:
      out[swing] = feet[swing][j]
    else:
      self.contact[swing] = False
      s = (local - ds) / (half - ds)
      p0, p1 = feet[swing][j], feet[swing][j + 1]
      out[swing] = p0 + (p1 - p0) * _smoothstep(s)
      out[swing][2] = p0[2] + p.swing_height * 64.0 * s**3 * (1.0 - s) ** 3
    return out

  # ---------------------------------------------------------------- update
  def step(
    self,
    t: float,
    vx: float = 0.0,
    vy: float = 0.0,
    walk: bool = False,
    com_meas: np.ndarray | None = None,
    vcom_meas: np.ndarray | None = None,
    feet_meas: np.ndarray | None = None,
  ) -> np.ndarray:
    """Advance the plan to time t + dt; return leg joint targets (10,).

    ``com_meas``/``vcom_meas`` (world frame) enable the balance feedback.
    With ``feet_meas`` (estimated foot positions, (2, 3)) the DCM error is
    taken relative to the support feet, so slow drift between the estimate
    and the plan (foot slip, odometry error) does not build up as feedback.
    """
    if self.t_first is None:
      self.t_first = t
    if walk and self.walk_start is None:
      self.walk_start = t
    self.vx, self.vy = (vx, vy) if walk else (0.0, 0.0)

    dcm_err = np.zeros(2)
    if com_meas is not None and vcom_meas is not None:
      dcm_meas = com_meas[:2] + vcom_meas[:2] / self.omega
      dcm_plan = self.com_state[:, 0] + self.com_state[:, 1] / self.omega
      dcm_err = dcm_meas - dcm_plan
      self._footsteps(t, t)  # make sure the current landing is committed
      j_now, _ = self._phase(t)
      if feet_meas is not None and self.p.reanchor_x and j_now >= 0 and j_now != self._last_step:
        stance = self._stance(j_now)
        shift = feet_meas[stance][0] - self.committed[stance][j_now][0]
        for f in range(2):
          for k in range(j_now, len(self.committed[f])):
            self.committed[f][k][0] += shift
        for k in list(self.nominal_landing):
          if k >= j_now:
            self.nominal_landing[k][0] += shift
        self.com_state[0, 0] += shift
        dcm_err[0] -= shift
      self._last_step = j_now
      if feet_meas is not None:
        support = [i for i in range(2) if self.contact[i]]
        j, _ = self._phase(t)
        planned = (
          [self.committed[i][j] for i in support] if j >= 0 else [self.feet0[i] for i in support]
        )
        dcm_err -= np.mean([feet_meas[i][:2] for i in support], axis=0) - np.mean(
          [f[:2] for f in planned], axis=0
        )
      self._adjust_landing(t + self.dt, dcm_err)

    plan = self._plan(t, self.mpc_params.horizon)
    vel_ref = (self.vx, self.vy)
    status = []
    self.plan_states = np.zeros((2, len(plan.zref), 3))
    for ax in range(2):
      _, states = self.mpc[ax].solve(
        self.com_state[ax],
        plan.zref[:, ax],
        plan.zmin[:, ax],
        plan.zmax[:, ax],
        np.full(len(plan.zref), vel_ref[ax]),
      )
      self.com_state[ax] = states[0]
      self.plan_states[ax] = states
      status.append(self.mpc[ax].last_status)
    self.last_plan = plan
    rise = _smoothstep((t - self.t_first) / self.p.com_raise_time)
    com = np.array([*self.com_state[:, 0], self.com_height + rise * self.p.com_raise])
    com[:2] -= np.asarray(self.p.dcm_gain) * dcm_err
    feet = self._feet_at(t + self.dt, plan.feet)
    zmp = self.com_state[:, 0] - self.com_state[:, 2] / self.omega**2
    q = self._ik(com, feet)
    self.debug.update(
      zref=plan.zref[0], zmin=plan.zmin[0], zmax=plan.zmax[0], zmp=zmp, com=com,
      feet=feet, status=status, dcm_err=dcm_err, com_acc=self.com_state[:, 2].copy(),
    )
    return q

  def schedule(self, t: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Planned contacts (n, 2) and foot positions (n, 2, 3) at t + k dt.

    Uses the footsteps of the last ``step`` call. A swinging foot is given
    its landing point (it carries no force while it swings).
    """
    assert self.last_plan is not None, "call step() first"
    feet = self.last_plan.feet
    ds = self.p.double_support * self.p.period
    contact = np.ones((n, 2), dtype=bool)
    pos = np.zeros((n, 2, 3))
    for k in range(n):
      j, local = self._phase(t + k * self.dt)
      if j < 0:
        pos[k] = self.feet0
        continue
      j = min(j, len(feet[0]) - 2)
      stance = self._stance(j)
      swing = 1 - stance
      pos[k, stance] = feet[stance][j]
      if local < ds:
        pos[k, swing] = feet[swing][j]
      else:
        contact[k, swing] = False
        pos[k, swing] = feet[swing][j + 1]
    return contact, pos

  # -------------------------------------------------------------------- IK
  def _ik(self, com: np.ndarray, feet: np.ndarray) -> np.ndarray:
    m, d, w = self.m, self.ik_data, self.ik_w
    nv = m.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    jcom = np.zeros((3, nv))
    eye = np.eye(3)
    jpost = np.zeros((len(self.dadr), nv))
    jpost[np.arange(len(self.dadr)), self.dadr] = 1.0
    err = np.zeros(1)
    for _ in range(w.iterations):
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      rows, errs, wts = [], [], []
      mujoco.mj_jacSubtreeCom(m, d, jcom, self.base_id)
      rows.append(jcom.copy())
      errs.append(com - d.subtree_com[self.base_id])
      wts.append(np.full(3, w.com))
      for i in range(2):
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site_ids[i])
        rows.append(jacp.copy())
        errs.append(feet[i] - d.site_xpos[self.site_ids[i]])
        wts.append(np.full(3, w.foot_pos))
        rows.append(jacr.copy())
        errs.append(_rot_error(self.foot_rot0[i], d.site_xmat[self.site_ids[i]].reshape(3, 3)))
        wts.append(np.full(3, w.foot_rot))
      mujoco.mj_jacBody(m, d, jacp, jacr, self.base_id)
      rows.append(jacr.copy())
      errs.append(_rot_error(eye, d.xmat[self.base_id].reshape(3, 3)))
      wts.append(np.asarray(w.base_rot))
      rows.append(jpost)
      errs.append(self.q_home - d.qpos[self.qadr])
      wts.append(np.full(len(self.dadr), w.posture))
      jac = np.vstack(rows)
      err = np.concatenate(errs)
      wt = np.concatenate(wts)
      hess = jac.T @ (wt[:, None] * jac) + w.damping * np.eye(nv)
      dq = np.linalg.solve(hess, jac.T @ (wt * err))
      mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
      d.qpos[self.qadr] = np.clip(d.qpos[self.qadr], self.qmin, self.qmax)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    self.debug["ik_err"] = float(np.abs(err[:15]).max())  # CoM + feet
    return d.qpos[self.qadr].copy()


def _rot_error(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
  """Small-angle rotation vector taking r_cur to r_des (world frame)."""
  e = r_des @ r_cur.T
  return 0.5 * np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])
