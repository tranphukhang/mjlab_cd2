"""Run a model-based walking controller on Open Duck in plain MuJoCo.

Controllers:
  zmp   LIPM-ZMP MPC + whole-body IK, joint position targets (default; the
        same interface as the real servos)
  srbd  single-rigid-body force MPC in torque mode (needs --armature 0.002)

Phases: stand still, step in place, then walk at ``--vx``. Prints tracking and
stability metrics, and optionally records a video or opens the MuJoCo viewer.

The zmp controller uses robot-like sensing only (IMU + joint angles, leg
odometry for the CoM). ``--latency`` and ``--gyro-noise`` emulate the real
servo bus and IMU.

  pixi run python scripts/mpc_walk.py --latency 0.02 --gyro-noise 0.05 --duration 60
  pixi run python scripts/mpc_walk.py --video videos/mpc_walk.mp4
  pixi run python scripts/mpc_walk.py --viewer
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from open_duck_mjlab.mpc.controller import Command, MpcWalkingController
from open_duck_mjlab.mpc.walk_feedback import BalanceFeedback, ComEstimator, FeedbackGains
from open_duck_mjlab.mpc.zmp_walker import WalkParams, ZmpWalker
from open_duck_mjlab.open_duck.open_duck_constants import (
  FOOT_GEOMS,
  INIT_STATE,
  JOINT_NAMES,
  get_spec,
)

def build_model() -> mujoco.MjModel:
  """Open Duck on a flat floor with the same foot contacts as the RL tasks."""
  spec = get_spec()
  for g in spec.geoms:
    if g.name in FOOT_GEOMS:
      g.contype = 1
      g.conaffinity = 1
      g.condim = 3
      g.priority = 1
      g.friction = [0.8, 0.005, 0.0001]
  tex = spec.add_texture()
  tex.name = "grid"
  tex.type = mujoco.mjtTexture.mjTEXTURE_2D
  tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  tex.rgb1 = [0.85, 0.85, 0.85]
  tex.rgb2 = [0.65, 0.65, 0.65]
  tex.width = 300
  tex.height = 300
  mat = spec.add_material()
  mat.name = "grid"
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  mat.texrepeat = [8, 8]
  mat.texuniform = True
  floor = spec.worldbody.add_geom()
  floor.name = "floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = [0, 0, 0.01]
  floor.material = "grid"
  floor.contype = 1
  floor.conaffinity = 1
  light = spec.worldbody.add_light()
  light.pos = [0, 0, 3]
  light.dir = [0, 0, -1]
  light.diffuse = [0.7, 0.7, 0.7]
  return spec.compile()

class ZmpController:
  """Model-based walker at the policy rate (50 Hz), holding its joint targets.

  Uses only robot-like sensing: IMU orientation and gyro (optionally noisy)
  and joint angles; the CoM comes from leg odometry (``--estimator kin``) or,
  for debugging, from the simulator (``--estimator true``).
  """

  def __init__(
    self,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    params: WalkParams,
    gains: FeedbackGains,
    estimator: str = "kin",
    gyro_noise: float = 0.0,
    seed: int = 0,
  ):
    self.m = m
    self.walker = ZmpWalker(m, params)
    self.walker.reset(d)
    w = self.walker
    self.qadr = w.qadr
    self.control_dt = w.dt
    self.feedback = BalanceFeedback(m, w.dadr, w.site_ids, gains)
    self.estimator = ComEstimator(m, w.qadr, w.site_ids, w.base_id, w.dt) if estimator == "kin" else None
    if self.estimator is not None:
      self.estimator.reset(w.feet0)
    self.gyro_noise = gyro_noise
    self.rng = np.random.default_rng(seed)
    self.next_t = -1.0
    self.ctrl = d.ctrl.copy()
    self.last_solve_ms = 0.0
    self.base = m.body("base_link").id

  def compute(self, d: mujoco.MjData, t: float, cmd: Command) -> np.ndarray:
    if t >= self.next_t - 1e-9:
      tic = time.perf_counter()
      w = self.walker
      gyro = d.qvel[3:6] + self.gyro_noise * self.rng.standard_normal(3)
      if self.estimator is not None:
        com, vcom = self.estimator.update(d.qpos[3:7], d.qpos[self.qadr], w.contact)
        feet = self.estimator.foot_world.copy()
      else:
        mujoco.mj_subtreeVel(self.m, d)
        com, vcom = d.subtree_com[self.base].copy(), d.subtree_linvel[self.base].copy()
        feet = np.array([d.site_xpos[s].copy() for s in w.site_ids])
      q = w.step(
        t, cmd.vx, cmd.vy, walk=not cmd.stand, com_meas=com, vcom_meas=vcom, feet_meas=feet
      )
      dbg = w.debug
      q = q + self.feedback.offsets(
        w.ik_data, w.contact, gyro, dbg["com_acc"], dbg["zmp"], dbg["feet"]
      )
      self.ctrl = q
      self.last_solve_ms = 1e3 * (time.perf_counter() - tic)
      self.next_t = t + self.control_dt
    return self.ctrl

  def measure(self, d: mujoco.MjData) -> dict:
    r = d.xmat[self.base].reshape(3, 3)
    rpy = np.array([
      np.arctan2(r[2, 1], r[2, 2]),
      -np.arcsin(np.clip(r[2, 0], -1.0, 1.0)),
      np.arctan2(r[1, 0], r[0, 0]),
    ])
    # Free-joint linear velocity is in the world frame.
    return {"rpy": rpy, "com": d.subtree_com[self.base].copy(), "vcom": d.qvel[0:3].copy()}


POSES = {
  "home": dict(INIT_STATE.joint_pos),
  # Hip roll 10 deg inward, hip pitch 20, knee 40, ankle 20: taller CoM,
  # feet closer together, less servo load than the home pose.
  "crouch10": {
    "L_hip_yaw": 0.0, "L_hip_roll": -np.radians(10), "L_hip_pitch": np.radians(20),
    "L_knee": -np.radians(40), "L_ankle": np.radians(20),
    "R_hip_yaw": 0.0, "R_hip_roll": np.radians(10), "R_hip_pitch": -np.radians(20),
    "R_knee": np.radians(40), "R_ankle": -np.radians(20),
  },
}


def reset(m: mujoco.MjModel, d: mujoco.MjData, pose: str = "home") -> None:
  """Put the robot on the floor in a standing pose with matching targets."""
  mujoco.mj_resetData(m, d)
  d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  q = POSES[pose]
  for name in JOINT_NAMES:
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = q[name]
  d.ctrl[:] = [q[n] for n in JOINT_NAMES]
  mujoco.mj_forward(m, d)
  sites = [m.site(n).id for n in ("L6_IK", "R6_IK")]
  d.qpos[2] += 0.002 - min(d.site_xpos[s][2] for s in sites)
  mujoco.mj_forward(m, d)

def command_at(t: float, args: argparse.Namespace) -> Command:
  if t < args.stand:
    return Command(stand=True)
  if t < args.stand + args.in_place:
    return Command(stand=False)
  ramp = min(1.0, (t - args.stand - args.in_place) / 1.0)
  return Command(vx=args.vx * ramp, vy=0.0, wz=args.wz * ramp, stand=False)

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--duration", type=float, default=10.0)
  ap.add_argument("--stand", type=float, default=1.0)
  ap.add_argument("--in-place", type=float, default=2.0)
  ap.add_argument("--vx", type=float, default=0.1)
  ap.add_argument("--wz", type=float, default=0.0)
  ap.add_argument("--video", type=str, default="")
  ap.add_argument("--viewer", action="store_true")
  ap.add_argument("--settle", type=float, default=0.3, help="position-hold before the controller")
  ap.add_argument("--controller", choices=("zmp", "srbd"), default="zmp")
  ap.add_argument("--period", type=float, default=WalkParams.period)
  ap.add_argument("--swing-height", type=float, default=WalkParams.swing_height)
  ap.add_argument("--double-support", type=float, default=WalkParams.double_support)
  ap.add_argument("--dcm-gain", type=float, nargs=2, default=WalkParams.dcm_gain, metavar=("X", "Y"))
  ap.add_argument("--step-gain", type=float, nargs=2, default=WalkParams.step_gain, metavar=("X", "Y"))
  ap.add_argument("--com-raise", type=float, default=WalkParams.com_raise)
  ap.add_argument("--no-reanchor-x", dest="reanchor_x", action="store_false")
  ap.add_argument("--gyro", type=float, nargs=2, default=FeedbackGains.gyro, metavar=("ROLL", "PITCH"))
  ap.add_argument("--load", type=float, default=FeedbackGains.load, help="load compensation 0..1")
  ap.add_argument("--estimator", choices=("kin", "true"), default="kin")
  ap.add_argument("--gyro-noise", type=float, default=0.0, help="rad/s std")
  ap.add_argument("--latency", type=float, default=0.0, help="command delay, s")
  ap.add_argument("--pose", choices=tuple(POSES), default=None, help="default: crouch10 (zmp), home (srbd)")
  ap.add_argument("--armature", type=float, default=None, help="override joint armature")
  args = ap.parse_args()

  m = build_model()
  if args.armature is not None:
    m.dof_armature[6:] = args.armature
  d = mujoco.MjData(m)
  if args.pose is None:
    args.pose = "crouch10" if args.controller == "zmp" else "home"
  reset(m, d, args.pose)
  base = m.body("base_link").id
  dt = m.opt.timestep

  # Let the feet settle on the floor under the home position targets.
  for _ in range(int(args.settle / dt)):
    mujoco.mj_step(m, d)
  t0 = d.time
  if args.controller == "zmp":
    params = WalkParams(
      period=args.period,
      swing_height=args.swing_height,
      double_support=args.double_support,
      dcm_gain=tuple(args.dcm_gain),
      step_gain=tuple(args.step_gain),
      com_raise=args.com_raise,
      reanchor_x=args.reanchor_x,
    )
    gains = FeedbackGains(gyro=tuple(args.gyro), load=args.load)
    ctrl = ZmpController(m, d, params, gains, args.estimator, args.gyro_noise)
  else:
    ctrl = MpcWalkingController(m)

  renderer = cam = None
  frames: list[np.ndarray] = []
  fps = 30
  if args.video:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base
    cam.distance = 0.9
    cam.elevation = -15
    cam.azimuth = 135
  viewer = None
  if args.viewer:

    viewer = mujoco.viewer.launch_passive(m, d)

  log = {
    "t": [], "vx": [], "vx_cmd": [], "z": [], "roll": [], "pitch": [], "yaw": [],
    "solve_ms": [], "x": [], "foot_z": [],
  }
  foot_sites = [m.site(n).id for n in ("L6_IK", "R6_IK")]
  foot_z0 = np.array([d.site_xpos[i][2] for i in foot_sites])
  fell = None
  pending: list[tuple[float, np.ndarray]] = []  # delayed commands (send time, ctrl)
  n_steps = int(args.duration / dt)
  wall = time.perf_counter()
  for step in range(n_steps):
    t = d.time - t0
    cmd = command_at(t, args)
    mujoco.mj_step1(m, d)
    u = ctrl.compute(d, t, cmd)
    if args.latency > 0.0:
      if not pending or pending[-1][1] is not u:
        pending.append((t, u))
      while len(pending) > 1 and pending[1][0] <= t - args.latency:
        pending.pop(0)
      if pending[0][0] <= t - args.latency:
        d.ctrl[:] = pending[0][1]
    else:
      d.ctrl[:] = u
    mujoco.mj_step2(m, d)

    if step % max(1, int(round(0.01 / dt))) == 0:
      s = ctrl.measure(d)
      yaw = s["rpy"][2]
      log["t"].append(t)
      log["vx"].append(np.array([np.cos(yaw), np.sin(yaw)]) @ s["vcom"][:2])
      log["vx_cmd"].append(cmd.vx)
      log["z"].append(s["com"][2])
      log["roll"].append(s["rpy"][0])
      log["pitch"].append(s["rpy"][1])
      log["yaw"].append(yaw)
      log["x"].append(d.subtree_com[base][0])
      log["foot_z"].append([d.site_xpos[i][2] for i in foot_sites] - foot_z0)
      log["solve_ms"].append(ctrl.last_solve_ms if args.controller == "zmp" else ctrl.mpc.last_solve_ms)
      if fell is None and (d.xpos[base][2] < 0.12 or abs(s["rpy"][0]) > 0.7 or abs(s["rpy"][1]) > 0.7):
        fell = t
        print(f"FELL at t={t:.2f}s (base z {d.xpos[base][2]:.3f}, rpy {np.round(s['rpy'], 2)})")
        if not args.viewer:
          break
    if renderer is not None and step % int(round(1.0 / (fps * dt))) == 0:
      renderer.update_scene(d, cam)
      frames.append(renderer.render())
    if viewer is not None:
      viewer.sync()
      if not viewer.is_running():
        break
      time.sleep(max(0.0, (d.time - t0) - (time.perf_counter() - wall)))

  wall = time.perf_counter() - wall
  arr = {k: np.asarray(v) for k, v in log.items()}
  walk = arr["t"] > args.stand + args.in_place + 1.0
  print(f"simulated {arr['t'][-1]:.2f}s in {wall:.1f}s wall; controller {np.mean(arr['solve_ms']):.1f} ms/tick mean")
  print(f"CoM height mean {arr['z'].mean():.3f} m (min {arr['z'].min():.3f})")
  print(f"roll |max| {np.degrees(np.abs(arr['roll']).max()):.1f} deg, pitch |max| {np.degrees(np.abs(arr['pitch']).max()):.1f} deg")
  if walk.any():
    err = arr["vx"][walk] - arr["vx_cmd"][walk]
    print(f"walking: vx mean {arr['vx'][walk].mean():.3f} (cmd {arr['vx_cmd'][walk].mean():.3f}), |err| mean {np.abs(err).mean():.3f} m/s")
    print(f"yaw drift {np.degrees(arr['yaw'][walk][-1] - arr['yaw'][walk][0]):.1f} deg over {arr['t'][walk][-1] - arr['t'][walk][0]:.1f}s")
  steps = arr["t"] > args.stand
  if steps.any():
    fz = arr["foot_z"][steps]
    print(f"stepping: foot clearance p90 L {np.percentile(fz[:, 0], 90) * 1e3:.1f} mm, R {np.percentile(fz[:, 1], 90) * 1e3:.1f} mm")
  print(f"CoM moved {arr['x'][-1] - arr['x'][0]:+.3f} m forward")
  print("result:", "FELL" if fell is not None else "OK (no fall)")

  if renderer is not None and frames:
    import mediapy

    mediapy.write_video(args.video, frames, fps=fps)
    print("video:", args.video, len(frames), "frames")

if __name__ == "__main__":
  main()
