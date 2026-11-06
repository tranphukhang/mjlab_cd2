"""Model-based walking controller for Open Duck: SRBD-MPC plus leg control.

Layers, top to bottom:

1. Gait clock: a fixed contact schedule (double support, right swing, double
   support, left swing), the same timing as the RL GaitPhase task.
2. SRBD-MPC (``srbd_mpc.py``): ground reaction force and moment per stance
   foot that track a CoM velocity, height and heading reference.
3. Swing feet: Raibert/capture-point foothold, smooth swing trajectory and a
   Cartesian PD on the foot site (position plus keeping the sole flat).
4. Joint torques: stance tau = -J^T [F; M] + bias, swing tau = J^T wrench +
   bias, where ``bias`` (MuJoCo ``qfrc_bias``) cancels leg gravity and
   Coriolis terms.

The MJCF keeps its STS3215 position actuators (``kp (ctrl - q)``, ``kv = 0``),
so a joint torque ``tau`` is applied in simulation by commanding
``ctrl = q + (tau + damping * qdot) / kp`` every physics step. This is exact in
MuJoCo; the real servos only accept position targets, so this torque mode is
for simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from open_duck_mjlab.mpc.srbd_mpc import NX, MpcParams, SrbdMpc

LEGS = (
  ("L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle"),
  ("R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"),
)
FOOT_SITES = ("L6_IK", "R6_IK")  # sole frame: x forward, y left, z up at home
BASE_BODY = "base_link"


@dataclass
class GaitParams:
  period: float = 0.40
  double_support: float = 0.10  # fraction of the period, per DS phase
  swing_height: float = 0.03
  # Before the first lift-off both feet stay down while the CoM starts
  # rocking on the LIPM sway orbit (it cannot reach the orbit velocity within
  # one short double-support phase).
  rock_periods: float = 2.0
  # Capture-point foothold feedback gain, about sqrt(z_com / g).
  foothold_gain: float = 0.12
  # Foot site minus CoM at the home pose, body frame (x, y).
  nominal_foot_offset: tuple[tuple[float, float], tuple[float, float]] = (
    (0.005, 0.0907),
    (0.005, -0.0907),
  )
  # No footstep planning: each swing foot lands back where it lifted off and
  # keeps its lift-off orientation (the MPC alone keeps the body balanced).
  fixed_footholds: bool = False


@dataclass
class SwingGains:
  kp: float = 400.0  # N/m
  kd: float = 8.0  # N s/m
  kp_rot: float = 3.0  # N m/rad
  kd_rot: float = 0.08  # N m s/rad


@dataclass
class Command:
  vx: float = 0.0
  vy: float = 0.0
  wz: float = 0.0
  stand: bool = True


@dataclass
class ControllerState:
  yaw_des: float = 0.0
  swing_start: list[np.ndarray | None] = field(default_factory=lambda: [None, None])
  swing_start_rot: list[np.ndarray | None] = field(default_factory=lambda: [None, None])
  foothold: list[np.ndarray | None] = field(default_factory=lambda: [None, None])
  wrench: np.ndarray = field(default_factory=lambda: np.zeros(12))
  last_mpc_time: float = -1.0
  stand_time: float = 0.0


def _rot_z(yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy(r: np.ndarray) -> np.ndarray:
  roll = np.arctan2(r[2, 1], r[2, 2])
  pitch = -np.arcsin(np.clip(r[2, 0], -1.0, 1.0))
  yaw = np.arctan2(r[1, 0], r[0, 0])
  return np.array([roll, pitch, yaw])


def _wrap(a: float) -> float:
  return (a + np.pi) % (2.0 * np.pi) - np.pi


class MpcWalkingController:
  def __init__(
    self,
    model: mujoco.MjModel,
    mpc_params: MpcParams | None = None,
    gait: GaitParams | None = None,
    swing_gains: SwingGains | None = None,
    com_height: float = 0.151,
    mpc_period: float = 0.01,
    inertia_ff: bool = True,
  ):
    self.m = model
    self.mpc = SrbdMpc(mpc_params)
    self.gait = gait or GaitParams()
    self.gains = swing_gains or SwingGains()
    self.com_height = com_height
    self.mpc_period = mpc_period
    self.inertia_ff = inertia_ff
    self.base_id = model.body(BASE_BODY).id
    self.site_ids = [model.site(s).id for s in FOOT_SITES]
    self.dof = [np.array([model.jnt_dofadr[model.joint(j).id] for j in leg]) for leg in LEGS]
    self.qadr = [np.array([model.jnt_qposadr[model.joint(j).id] for j in leg]) for leg in LEGS]
    # Actuator order matches LEGS (L then R); read kp and joint damping.
    names = [j for leg in LEGS for j in leg]
    self.act = np.array([model.actuator(n).id for n in names])
    self.kp = model.actuator_gainprm[self.act, 0].copy()
    self.damping = np.array([model.dof_damping[model.jnt_dofadr[model.joint(n).id]] for n in names])
    self.st = ControllerState()
    self._jacp = np.zeros((3, model.nv))
    self._jacr = np.zeros((3, model.nv))

  # ---------------------------------------------------------------- gait clock
  def contact_at(self, t: float, cmd: Command) -> np.ndarray:
    """Planned stance flags (left, right) at time ``t``."""
    if cmd.stand or t - self.st.stand_time < self.gait.rock_periods * self.gait.period:
      return np.array([True, True])
    phase = ((t - self.st.stand_time) / self.gait.period) % 1.0
    d = self.gait.double_support
    right_swing = d <= phase < 0.5
    left_swing = phase >= 0.5 + d
    return np.array([not left_swing, not right_swing])

  def swing_progress(self, t: float, leg: int, cmd: Command) -> tuple[float, float]:
    """(progress in [0, 1], remaining swing time) for a swinging leg."""
    phase = ((t - self.st.stand_time) / self.gait.period) % 1.0
    d = self.gait.double_support
    start = d if leg == 1 else 0.5 + d
    frac = (phase - start) / (0.5 - d)
    frac = float(np.clip(frac, 0.0, 1.0))
    return frac, (1.0 - frac) * (0.5 - d) * self.gait.period

  # --------------------------------------------------------------- estimation
  def measure(self, d: mujoco.MjData) -> dict:
    mujoco.mj_subtreeVel(self.m, d)
    r = d.xmat[self.base_id].reshape(3, 3)
    feet = np.array([d.site_xpos[s] for s in self.site_ids])
    foot_yaw = np.array([_rpy(d.site_xmat[s].reshape(3, 3))[2] for s in self.site_ids])
    return {
      "com": d.subtree_com[self.base_id].copy(),
      "vcom": d.subtree_linvel[self.base_id].copy(),
      "R": r,
      "rpy": _rpy(r),
      "omega": r @ d.qvel[3:6],
      "feet": feet,
      "foot_yaw": foot_yaw,
    }

  # ---------------------------------------------------------------- footholds
  def _nominal_foot(self, com_xy: np.ndarray, yaw: float, leg: int) -> np.ndarray:
    off = np.array([*self.gait.nominal_foot_offset[leg], 0.0])
    p = np.array([com_xy[0], com_xy[1], 0.0]) + _rot_z(yaw) @ off
    p[2] = 0.0
    return p

  def _foothold(
    self, s: dict, leg: int, t: float, t_remaining: float, cmd: Command
  ) -> np.ndarray:
    """Touchdown point of the swinging foot, world frame.

    Forward (heading x): Raibert, centered under the predicted CoM plus
    ``v T_stance / 2`` and capture-point feedback on the velocity error.
    Lateral (heading y): one step width from the stance foot, plus feedback
    on the deviation of the lateral CoM velocity from the LIPM sway orbit.
    """
    yaw = s["rpy"][2]
    ex = np.array([np.cos(yaw), np.sin(yaw)])
    ey = np.array([-np.sin(yaw), np.cos(yaw)])
    stance_time = (0.5 + self.gait.double_support) * self.gait.period
    v_along = ex @ s["vcom"][:2]
    along = ex @ s["com"][:2] + v_along * t_remaining
    along += self.gait.nominal_foot_offset[leg][0]
    along += 0.5 * stance_time * v_along
    along += self.gait.foothold_gain * (v_along - cmd.vx)
    other = 1 - leg
    width = self.gait.nominal_foot_offset[leg][1] - self.gait.nominal_foot_offset[other][1]
    lateral = ey @ s["feet"][other][:2] + width
    _, vy_ref = self._sway(t)
    lateral += self.gait.foothold_gain * (ey @ s["vcom"][:2] - cmd.vy - vy_ref)
    xy = along * ex + lateral * ey
    return np.array([xy[0], xy[1], 0.0])

  # ------------------------------------------------------------ lateral sway
  def _sway(self, t: float) -> tuple[float, float]:
    """Lateral CoM offset from the feet midline and its rate (LIPM orbit).

    With feet ``+-a`` apart and the support switching every half period
    ``T/2``, the periodic linear-inverted-pendulum solution about the stance
    foot is ``a - y_min cosh(w (tau - T/4))`` with ``y_min = a / cosh(w T/4)``:
    the CoM crosses the midline at each support switch moving toward the new
    stance foot, and gravity turns it around at mid-stance.
    """
    g = self.gait
    if t < self.st.stand_time:
      return 0.0, 0.0
    period = g.period
    omega = np.sqrt(9.81 / self.com_height)
    a = g.nominal_foot_offset[0][1]
    y_min = a / np.cosh(omega * period / 4.0)
    u = (((t - self.st.stand_time) / period) - 0.5 * g.double_support) % 1.0
    left = u < 0.5
    tau = (u if left else u - 0.5) * period - period / 4.0
    y = a - y_min * np.cosh(omega * tau)
    vy = -y_min * omega * np.sinh(omega * tau)
    ramp_time = max(g.rock_periods, 1.0) * period
    ramp = float(np.clip((t - self.st.stand_time) / ramp_time, 0.0, 1.0))
    sign = 1.0 if left else -1.0
    return sign * y * ramp, sign * vy * ramp

  # ---------------------------------------------------------------------- MPC
  def _run_mpc(self, s: dict, t: float, cmd: Command) -> None:
    p = self.mpc.p
    n, dt = p.horizon, p.dt
    yaw0 = s["rpy"][2]
    self.st.yaw_des = _wrap(self.st.yaw_des + cmd.wz * self.mpc_period)
    yaw_ref0 = yaw0 + _wrap(self.st.yaw_des - yaw0)
    v_des = _rot_z(yaw0) @ np.array([cmd.vx, cmd.vy, 0.0])

    x0 = np.concatenate([s["rpy"], s["com"], s["omega"], s["vcom"], [1.0]])
    x_ref = np.zeros((n, NX))
    contact = np.zeros((n, 2), dtype=bool)
    foot_pos = np.zeros((n, 2, 3))
    ex = np.array([np.cos(yaw0), np.sin(yaw0)])
    ey = np.array([-np.sin(yaw0), np.cos(yaw0)])
    now = self.contact_at(t, cmd)
    if cmd.stand:
      # Hold the CoM over the middle of the two soles.
      sole_mid = 0.5 * (s["feet"][0] + s["feet"][1])[:2] + 0.012 * ex
      along0, mid_lat = ex @ sole_mid, ey @ sole_mid
    else:
      # Lateral reference: feet midline plus the LIPM sway. A swinging foot
      # counts at its planned touchdown point.
      feet = [
        self.st.foothold[i] if not now[i] and self.st.foothold[i] is not None else s["feet"][i]
        for i in range(2)
      ]
      along0, mid_lat = ex @ s["com"][:2], 0.5 * ey @ (feet[0][:2] + feet[1][:2])
      if self.gait.fixed_footholds:
        # Stepping in place: keep the CoM over the middle of the soles.
        along0 = ex @ (0.5 * (feet[0][:2] + feet[1][:2])) - self.gait.nominal_foot_offset[0][0]
    # Per foot: number of contact switches seen so far along the horizon.
    switches = [0, 0]
    for k in range(n):
      tk = t + (k + 1) * dt
      sway, sway_v = (0.0, 0.0) if cmd.stand else self._sway(tk)
      along = along0 + cmd.vx * (k + 1) * dt
      lat = mid_lat + sway + cmd.vy * (k + 1) * dt
      xy = along * ex + lat * ey
      vxy = cmd.vx * ex + (cmd.vy + sway_v) * ey
      x_ref[k, 0:3] = (0.0, 0.0, yaw_ref0 + cmd.wz * (k + 1) * dt)
      x_ref[k, 3:6] = (xy[0], xy[1], self.com_height)
      x_ref[k, 6:9] = (0.0, 0.0, cmd.wz)
      x_ref[k, 9:12] = (vxy[0], vxy[1], 0.0)
      x_ref[k, 12] = 1.0
      contact[k] = self.contact_at(t + k * dt, cmd)
      for i in range(2):
        prev = now[i] if k == 0 else contact[k - 1, i]
        if contact[k, i] != prev:
          switches[i] += 1
        if now[i] and switches[i] == 0:
          foot_pos[k, i] = s["feet"][i]  # current stance foot
        elif not now[i] and switches[i] == 1 and self.st.foothold[i] is not None:
          foot_pos[k, i] = self.st.foothold[i]  # landing of the current swing
        else:
          foot_pos[k, i] = self._nominal_foot(x_ref[k, 3:5], x_ref[k, 2], i)
    foot_rel = foot_pos - x_ref[:, None, 3:6]
    self.st.wrench = self.mpc.solve(x0, x_ref, contact, foot_rel, s["foot_yaw"])

  # ------------------------------------------------------------------ control
  def compute(self, d: mujoco.MjData, t: float, cmd: Command) -> np.ndarray:
    """Return actuator ctrl (10,). Call every physics step after mj_step1."""
    s = self.measure(d)
    if self.st.last_mpc_time < 0.0:
      self.st.yaw_des = s["rpy"][2]
    if cmd.stand:
      self.st.stand_time = t  # gait clock starts when stepping starts
    contact = self.contact_at(t, cmd)
    v_des = _rot_z(s["rpy"][2]) @ np.array([cmd.vx, cmd.vy, 0.0])

    # Swing bookkeeping and foothold update.
    for i in range(2):
      if contact[i]:
        self.st.swing_start[i] = None
        self.st.foothold[i] = None
      else:
        if self.st.swing_start[i] is None:
          self.st.swing_start[i] = s["feet"][i].copy()
          self.st.swing_start_rot[i] = d.site_xmat[self.site_ids[i]].reshape(3, 3).copy()
        if self.gait.fixed_footholds:
          self.st.foothold[i] = self.st.swing_start[i].copy()
        else:
          _, t_rem = self.swing_progress(t, i, cmd)
          self.st.foothold[i] = self._foothold(s, i, t, t_rem, cmd)

    if self.st.last_mpc_time < 0.0 or t - self.st.last_mpc_time >= self.mpc_period - 1e-9:
      self._run_mpc(s, t, cmd)
      self.st.last_mpc_time = t

    jacs = []
    for i in range(2):
      mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, self.site_ids[i])
      jacs.append((self._jacp.copy(), self._jacr.copy()))
    if self.inertia_ff:
      tau_ff = self._inertia_feedforward(d, s, contact, jacs)
    else:
      tau_ff = np.zeros(self.m.nv)

    tau = np.zeros(10)
    for i in range(2):
      self._jacp[:], self._jacr[:] = jacs[i]
      jp = self._jacp[:, self.dof[i]]
      jr = self._jacr[:, self.dof[i]]
      bias = d.qfrc_bias[self.dof[i]] + tau_ff[self.dof[i]]
      if contact[i]:
        f = self.st.wrench[3 * i : 3 * i + 3]
        mom = self.st.wrench[6 + 3 * i : 9 + 3 * i]
        tau_leg = -(jp.T @ f + jr.T @ mom) + bias
      else:
        p_des, v_des_foot = self._swing_target(t, i, cmd)
        foot_vel = jp @ d.qvel[self.dof[i]] + self._jacp[:, :6] @ d.qvel[:6]
        force = self.gains.kp * (p_des - s["feet"][i]) + self.gains.kd * (v_des_foot - foot_vel)
        if self.gait.fixed_footholds:
          r_des = self.st.swing_start_rot[i]
        else:
          r_des = _rot_z(s["rpy"][2])
        r_cur = d.site_xmat[self.site_ids[i]].reshape(3, 3)
        r_err = r_des @ r_cur.T
        e_rot = 0.5 * np.array(
          [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
        )
        w_foot = jr @ d.qvel[self.dof[i]] + self._jacr[:, :6] @ d.qvel[:6]
        moment = self.gains.kp_rot * e_rot - self.gains.kd_rot * w_foot
        tau_leg = jp.T @ force + jr.T @ moment + bias
      tau[5 * i : 5 * i + 5] = tau_leg

    qd = np.concatenate([d.qvel[self.dof[0]], d.qvel[self.dof[1]]])
    q = np.concatenate([d.qpos[self.qadr[0]], d.qpos[self.qadr[1]]])
    self.last_tau = tau
    self.last_contact = contact
    return q + (tau + self.damping * qd) / self.kp

  def _swing_target(self, t: float, leg: int, cmd: Command) -> tuple[np.ndarray, np.ndarray]:
    frac, _ = self.swing_progress(t, leg, cmd)
    t_swing = (0.5 - self.gait.double_support) * self.gait.period
    p0 = self.st.swing_start[leg]
    pf = self.st.foothold[leg]
    blend = 3.0 * frac**2 - 2.0 * frac**3
    dblend = (6.0 * frac - 6.0 * frac**2) / t_swing
    p = p0 + (pf - p0) * blend
    v = (pf - p0) * dblend
    bump = 64.0 * frac**3 * (1.0 - frac) ** 3
    dbump = 64.0 * (3.0 * frac**2 * (1.0 - frac) ** 3 - 3.0 * frac**3 * (1.0 - frac) ** 2) / t_swing
    p[2] = p0[2] + (pf[2] - p0[2]) * blend + self.gait.swing_height * bump
    v[2] = (pf[2] - p0[2]) * dblend + self.gait.swing_height * dbump
    return p, v

  def _inertia_feedforward(
    self,
    d: mujoco.MjData,
    s: dict,
    contact: np.ndarray,
    jacs: list[tuple[np.ndarray, np.ndarray]],
  ) -> np.ndarray:
    """Joint torques ``M qddot`` for the motion the MPC asks for.

    The single-rigid-body model ignores the legs' own inertia and, above all,
    the servo rotor inertia (MJCF armature 0.027 kg m^2 per joint, reflected
    through the gearbox), which is large next to the 1.5 kg robot: moving the
    CoM sideways at a few m/s^2 needs nearly 1 N m at each hip roll just to
    spin the rotors. Desired accelerations: the base follows the MPC wrench,
    stance legs keep their feet still, swing legs get zero (their PD loop
    handles them).
    """
    p = self.mpc.p
    w = self.st.wrench
    force = w[0:3] + w[3:6]
    a_com = force / p.mass + np.array([0.0, 0.0, -9.81])
    r_body = s["R"]
    inertia_w = r_body @ np.diag(p.inertia) @ r_body.T
    torque = w[6:9] + w[9:12]
    for i in range(2):
      torque = torque + np.cross(s["feet"][i] - s["com"], w[3 * i : 3 * i + 3])
    alpha = np.linalg.solve(inertia_w, torque)
    qacc = np.zeros(self.m.nv)
    qacc[0:3] = a_com + np.cross(alpha, d.xpos[self.base_id] - s["com"])
    qacc[3:6] = r_body.T @ alpha
    for i in range(2):
      if not contact[i]:
        continue
      jacp, jacr = jacs[i]
      j_leg = np.vstack([jacp[:, self.dof[i]], jacr[:, self.dof[i]]])
      j_base = np.vstack([jacp[:, :6], jacr[:, :6]])
      qacc[self.dof[i]] = -np.linalg.pinv(j_leg) @ (j_base @ qacc[:6])
    out = np.zeros(self.m.nv)
    mujoco.mj_mulM(self.m, d, out, qacc)
    return out
