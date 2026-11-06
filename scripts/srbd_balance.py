"""SRBD-MPC balance test on Open Duck: stand, then step in place.

No footstep planning: each swing foot lands back where it lifted off. The
MPC (surface contact, CoP box per sole) is the only balance mechanism.

Default start pose: hip roll 10 deg inward, hip pitch 20 deg, knee 40 deg,
ankle 20 deg. Default Q penalizes only z, roll, pitch, yaw (weight 1); R is
unchanged.

  pixi run python scripts/srbd_balance.py                    # stand + step in place
  pixi run python scripts/srbd_balance.py --step-at 99       # stand only
  pixi run python scripts/srbd_balance.py --q 1,1,1,10,10,1,0,0,0,1,1,0,0
  pixi run python scripts/srbd_balance.py --armature 0.002 --video videos/srbd.mp4
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from mpc_walk import build_model
from open_duck_mjlab.mpc.controller import Command, GaitParams, MpcWalkingController
from open_duck_mjlab.mpc.srbd_mpc import MpcParams
from open_duck_mjlab.open_duck.open_duck_constants import INIT_STATE, JOINT_NAMES

# State order: roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, 1.
Q_ORIENTATION_HEIGHT = (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def pose_targets(name: str) -> np.ndarray:
  if name == "home":
    return np.array([INIT_STATE.joint_pos[n] for n in JOINT_NAMES])
  r = np.radians
  q = {
    "L_hip_yaw": 0.0, "L_hip_roll": -r(10), "L_hip_pitch": r(20), "L_knee": -r(40), "L_ankle": r(20),
    "R_hip_yaw": 0.0, "R_hip_roll": r(10), "R_hip_pitch": -r(20), "R_knee": r(40), "R_ankle": -r(20),
  }
  return np.array([q[n] for n in JOINT_NAMES])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pose", choices=("crouch10", "home"), default="crouch10")
  ap.add_argument("--q", default="", help="13 comma-separated Q weights (default z/roll/pitch/yaw = 1)")
  ap.add_argument("--horizon", type=int, default=20)
  ap.add_argument("--armature", type=float, default=None)
  ap.add_argument("--duration", type=float, default=6.0)
  ap.add_argument("--step-at", type=float, default=1.0, help="time stepping in place starts")
  ap.add_argument("--period", type=float, default=0.4)
  ap.add_argument("--swing-height", type=float, default=0.02)
  ap.add_argument("--inertia-ff", action="store_true", help="rotor-inertia feedforward")
  ap.add_argument("--video", default="")
  args = ap.parse_args()

  m = build_model()
  if args.armature is not None:
    m.dof_armature[6:] = args.armature
  d = mujoco.MjData(m)
  sites = [m.site("L6_IK").id, m.site("R6_IK").id]
  base = m.body("base_link").id
  adr = [m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]

  # Place the robot on the floor in the start pose and let it settle.
  q0 = pose_targets(args.pose)
  d.qpos[3] = 1.0
  d.qpos[adr] = q0
  mujoco.mj_forward(m, d)
  d.qpos[2] += 0.003 - min(d.site_xpos[s][2] for s in sites)
  d.ctrl[:] = q0
  for _ in range(int(0.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)
  com = d.subtree_com[base].copy()
  feet = np.array([d.site_xpos[s] for s in sites])
  offset = tuple((float(f[0] - com[0]), float(f[1] - com[1])) for f in feet)
  print(f"pose {args.pose}: CoM z {com[2]:.3f} m, feet y {feet[0][1]:+.3f} {feet[1][1]:+.3f} m")

  qw = tuple(float(x) for x in args.q.split(",")) if args.q else Q_ORIENTATION_HEIGHT
  ctrl = MpcWalkingController(
    m,
    MpcParams(q_weights=qw, horizon=args.horizon),
    GaitParams(
      nominal_foot_offset=offset, fixed_footholds=True, period=args.period,
      swing_height=args.swing_height, rock_periods=1.0,
    ),
    com_height=com[2],
    inertia_ff=args.inertia_ff,
  )

  renderer = cam = None
  frames: list[np.ndarray] = []
  if args.video:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base
    cam.distance = 0.8
    cam.elevation = -10
    cam.azimuth = 160

  t0 = d.time
  fz0 = feet[:, 2].copy()
  log = []
  fell = None
  dt = m.opt.timestep
  for step in range(int(args.duration / dt)):
    t = d.time - t0
    cmd = Command(stand=t < args.step_at)
    mujoco.mj_step1(m, d)
    d.ctrl[:] = ctrl.compute(d, t, cmd)
    mujoco.mj_step2(m, d)
    if step % max(1, int(round(0.01 / dt))) == 0:
      r = d.xmat[base].reshape(3, 3)
      roll, pitch = np.degrees([np.arctan2(r[2, 1], r[2, 2]), -np.arcsin(r[2, 0])])
      fz = np.array([d.site_xpos[s][2] for s in sites]) - fz0
      log.append((t, roll, pitch, fz[0], fz[1], d.subtree_com[base][2], np.abs(ctrl.last_tau).max()))
      if d.xpos[base][2] < 0.12 or abs(roll) > 40 or abs(pitch) > 40:
        fell = t
        break
    if renderer is not None and step % int(round(1.0 / (30 * dt))) == 0:
      renderer.update_scene(d, cam)
      frames.append(renderer.render())

  arr = np.array(log)
  print("result:", f"FELL at {fell:.2f} s" if fell is not None else "no fall")
  print(f"roll |max| {np.abs(arr[:, 1]).max():.1f} deg, pitch |max| {np.abs(arr[:, 2]).max():.1f} deg, "
        f"CoM z {arr[:, 5].min():.3f}..{arr[:, 5].max():.3f} m, |tau| p95 {np.percentile(arr[:, 6], 95):.2f} N m")
  stepping = arr[:, 0] > args.step_at
  if stepping.any():
    print(f"stepping {arr[stepping, 0][-1] - args.step_at:.1f} s: foot clearance p90 "
          f"L {np.percentile(arr[stepping, 3], 90) * 1e3:.1f} mm, R {np.percentile(arr[stepping, 4], 90) * 1e3:.1f} mm")
  if renderer is not None and frames:
    import mediapy

    mediapy.write_video(args.video, frames, fps=30)
    print("video:", args.video)


if __name__ == "__main__":
  main()
