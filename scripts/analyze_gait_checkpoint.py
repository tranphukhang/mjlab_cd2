#!/usr/bin/env python3
"""Evaluate a trained Open Duck policy and export gait/MPC-SRBD data.

The script runs one deterministic MJLab environment, loads an RSL-RL actor
checkpoint, and records one continuous episode.  It exports:

- timeseries.csv: raw policy-step data (CoM/base/feet/joints/actions/contacts/GRF)
- gait_events.csv: touchdown/liftoff events and per-step/stride metrics
- gait_summary.json / gait_summary.csv: scalar gait statistics
- gait_cycle_mean.csv: phase-normalized mean gait cycle (when >= 1 full stride)
- mpc_srbd_reference.csv / .npz: one representative stride for SRBD-MPC
- raw_rollout.npz: all numeric arrays for later Python analysis

The MPC export contains the quantities that are directly useful as references:
CoM position/velocity, base orientation/angular velocity, foothold positions,
contact schedule, and measured foot contact forces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Importing the package registers the custom task with MJLab.
import open_duck_mjlab  # noqa: F401
from open_duck_mjlab.open_duck import FOOT_GEOMS, FOOT_SITES, JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


TASK_ID = "Mjlab-Velocity-Flat-OpenDuckMiniV2"


def _np(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()


def _scalar(x: Any) -> float:
  if isinstance(x, torch.Tensor):
    return float(x.detach().cpu().item())
  return float(x)


def quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
  """Convert MuJoCo quaternion [w, x, y, z] to roll/pitch/yaw [rad]."""
  w, x, y, z = [float(v) for v in q]
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)
  return np.array([roll, pitch, yaw], dtype=np.float64)


def resolve_name_index(names: list[str] | tuple[str, ...], target: str) -> int:
  """Resolve a local/contact name, accepting optional entity prefixes."""
  for i, name in enumerate(names):
    if name == target or name.endswith("/" + target) or name.endswith(target):
      return i
  raise KeyError(f"Could not find {target!r} in names={list(names)}")


def filter_short_runs(x: np.ndarray, min_samples: int = 2) -> np.ndarray:
  """Remove one-sample contact chatter while keeping longer contact phases."""
  y = np.asarray(x, dtype=bool).copy()
  if len(y) < 3 or min_samples <= 1:
    return y

  changed = True
  while changed:
    changed = False
    start = 0
    while start < len(y):
      end = start + 1
      while end < len(y) and y[end] == y[start]:
        end += 1
      run_len = end - start
      if run_len < min_samples:
        left = y[start - 1] if start > 0 else None
        right = y[end] if end < len(y) else None
        if left is not None and right is not None and left == right:
          y[start:end] = left
          changed = True
        elif left is not None and right is None:
          y[start:end] = left
          changed = True
        elif left is None and right is not None:
          y[start:end] = right
          changed = True
      start = end
  return y


def safe_mean(values: list[float]) -> float | None:
  vals = np.asarray(values, dtype=np.float64)
  vals = vals[np.isfinite(vals)]
  return float(np.mean(vals)) if vals.size else None


def safe_std(values: list[float]) -> float | None:
  vals = np.asarray(values, dtype=np.float64)
  vals = vals[np.isfinite(vals)]
  return float(np.std(vals)) if vals.size else None


def symmetry_index(left: float | None, right: float | None) -> float | None:
  if left is None or right is None:
    return None
  den = abs(left) + abs(right)
  if den < 1e-12:
    return 0.0
  return 200.0 * abs(left - right) / den


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    path.write_text("", encoding="utf-8")
    return
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def build_events(
  t: np.ndarray,
  contact_l: np.ndarray,
  contact_r: np.ndarray,
  foot_l: np.ndarray,
  foot_r: np.ndarray,
  warmup_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Extract touchdown/liftoff events and standard spatiotemporal gait metrics."""
  contacts = {
    "L": filter_short_runs(contact_l),
    "R": filter_short_runs(contact_r),
  }
  feet = {"L": foot_l, "R": foot_r}

  events_raw: list[tuple[int, str, str]] = []
  for foot in ("L", "R"):
    c = contacts[foot]
    for i in range(1, len(c)):
      if (not c[i - 1]) and c[i]:
        events_raw.append((i, foot, "touchdown"))
      elif c[i - 1] and (not c[i]):
        events_raw.append((i, foot, "liftoff"))
  events_raw.sort(key=lambda z: (z[0], 0 if z[2] == "liftoff" else 1))

  last_td: dict[str, tuple[float, np.ndarray] | None] = {"L": None, "R": None}
  last_lo: dict[str, tuple[float, np.ndarray] | None] = {"L": None, "R": None}
  last_td_any: tuple[str, float, np.ndarray] | None = None
  rows: list[dict[str, Any]] = []

  step_lengths_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  step_times_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  stride_lengths_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  stride_times_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  stance_times_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  swing_times_by_foot: dict[str, list[float]] = {"L": [], "R": []}
  step_widths: list[float] = []

  for idx, foot, event in events_raw:
    ti = float(t[idx])
    p = feet[foot][idx].copy()
    row: dict[str, Any] = {
      "sample": int(idx),
      "time_s": ti,
      "foot": foot,
      "event": event,
      "foot_x_m": float(p[0]),
      "foot_y_m": float(p[1]),
      "foot_z_m": float(p[2]),
      "step_length_x_m": "",
      "step_width_m": "",
      "step_time_s": "",
      "stride_length_x_m": "",
      "stride_time_s": "",
      "stance_time_s": "",
      "swing_time_s": "",
    }

    if event == "touchdown":
      same_prev = last_td[foot]
      if same_prev is not None:
        stride_time = ti - same_prev[0]
        stride_length = float(p[0] - same_prev[1][0])
        row["stride_time_s"] = stride_time
        row["stride_length_x_m"] = stride_length
        if ti >= warmup_s:
          stride_times_by_foot[foot].append(stride_time)
          stride_lengths_by_foot[foot].append(stride_length)

      if last_lo[foot] is not None and last_lo[foot][0] > (same_prev[0] if same_prev else -1e9):
        swing_time = ti - last_lo[foot][0]
        row["swing_time_s"] = swing_time
        if ti >= warmup_s:
          swing_times_by_foot[foot].append(swing_time)

      if last_td_any is not None and last_td_any[0] != foot:
        step_time = ti - last_td_any[1]
        step_length = float(p[0] - last_td_any[2][0])
        step_width = float(abs(p[1] - last_td_any[2][1]))
        row["step_time_s"] = step_time
        row["step_length_x_m"] = step_length
        row["step_width_m"] = step_width
        if ti >= warmup_s:
          step_times_by_foot[foot].append(step_time)
          step_lengths_by_foot[foot].append(step_length)
          step_widths.append(step_width)

      last_td[foot] = (ti, p)
      last_td_any = (foot, ti, p)

    else:
      if last_td[foot] is not None:
        stance_time = ti - last_td[foot][0]
        row["stance_time_s"] = stance_time
        if ti >= warmup_s:
          stance_times_by_foot[foot].append(stance_time)
      last_lo[foot] = (ti, p)

    rows.append(row)

  all_step_lengths = step_lengths_by_foot["L"] + step_lengths_by_foot["R"]
  all_step_times = step_times_by_foot["L"] + step_times_by_foot["R"]
  all_stride_lengths = stride_lengths_by_foot["L"] + stride_lengths_by_foot["R"]
  all_stride_times = stride_times_by_foot["L"] + stride_times_by_foot["R"]

  mean_step_time = safe_mean(all_step_times)
  cadence = 60.0 / mean_step_time if mean_step_time and mean_step_time > 0 else None

  mean_stance_l = safe_mean(stance_times_by_foot["L"])
  mean_stance_r = safe_mean(stance_times_by_foot["R"])
  mean_stride_l = safe_mean(stride_times_by_foot["L"])
  mean_stride_r = safe_mean(stride_times_by_foot["R"])

  metrics = {
    "n_touchdowns": sum(1 for r in rows if r["event"] == "touchdown"),
    "n_valid_steps_after_warmup": len(all_step_times),
    "mean_step_length_m": safe_mean(all_step_lengths),
    "std_step_length_m": safe_std(all_step_lengths),
    "mean_step_width_m": safe_mean(step_widths),
    "std_step_width_m": safe_std(step_widths),
    "mean_step_time_s": mean_step_time,
    "std_step_time_s": safe_std(all_step_times),
    "cadence_steps_per_min": cadence,
    "mean_stride_length_m": safe_mean(all_stride_lengths),
    "mean_stride_time_s": safe_mean(all_stride_times),
    "left_step_length_m": safe_mean(step_lengths_by_foot["L"]),
    "right_step_length_m": safe_mean(step_lengths_by_foot["R"]),
    "left_step_time_s": safe_mean(step_times_by_foot["L"]),
    "right_step_time_s": safe_mean(step_times_by_foot["R"]),
    "left_stride_length_m": safe_mean(stride_lengths_by_foot["L"]),
    "right_stride_length_m": safe_mean(stride_lengths_by_foot["R"]),
    "left_stride_time_s": mean_stride_l,
    "right_stride_time_s": mean_stride_r,
    "left_stance_time_s": mean_stance_l,
    "right_stance_time_s": mean_stance_r,
    "left_swing_time_s": safe_mean(swing_times_by_foot["L"]),
    "right_swing_time_s": safe_mean(swing_times_by_foot["R"]),
    "left_duty_factor": (
      mean_stance_l / mean_stride_l
      if mean_stance_l is not None and mean_stride_l and mean_stride_l > 0
      else None
    ),
    "right_duty_factor": (
      mean_stance_r / mean_stride_r
      if mean_stance_r is not None and mean_stride_r and mean_stride_r > 0
      else None
    ),
  }
  metrics["step_length_symmetry_percent"] = symmetry_index(
    metrics["left_step_length_m"], metrics["right_step_length_m"]
  )
  metrics["step_time_symmetry_percent"] = symmetry_index(
    metrics["left_step_time_s"], metrics["right_step_time_s"]
  )
  return rows, metrics


def interp_columns(
  t_src: np.ndarray,
  x_src: np.ndarray,
  t_dst: np.ndarray,
) -> np.ndarray:
  if x_src.ndim == 1:
    return np.interp(t_dst, t_src, x_src)
  out = np.empty((len(t_dst), x_src.shape[1]), dtype=np.float64)
  for j in range(x_src.shape[1]):
    out[:, j] = np.interp(t_dst, t_src, x_src[:, j])
  return out


def build_mean_cycle(
  data: dict[str, np.ndarray],
  events: list[dict[str, Any]],
  warmup_s: float,
  phase_points: int = 101,
) -> dict[str, np.ndarray] | None:
  """Average left-touchdown-to-left-touchdown strides onto a 0..1 phase grid."""
  left_td = [
    int(r["sample"])
    for r in events
    if r["foot"] == "L"
    and r["event"] == "touchdown"
    and float(r["time_s"]) >= warmup_s
  ]
  if len(left_td) < 2:
    return None

  phase = np.linspace(0.0, 1.0, phase_points)
  cycle_keys = [
    "joint_pos",
    "joint_vel",
    "com_vel",
    "base_rpy",
    "base_ang_vel_w",
    "foot_l_force",
    "foot_r_force",
  ]
  cycles: dict[str, list[np.ndarray]] = {k: [] for k in cycle_keys}
  cycles.update(
    {
      "com_rel": [],
      "foot_l_rel_origin": [],
      "foot_r_rel_origin": [],
      "foot_l_rel_com": [],
      "foot_r_rel_com": [],
      "contact_l": [],
      "contact_r": [],
    }
  )
  stride_times: list[float] = []

  t = data["time"]
  for i0, i1 in zip(left_td[:-1], left_td[1:]):
    if i1 - i0 < 4:
      continue
    ts = t[i0 : i1 + 1]
    if ts[-1] <= ts[0]:
      continue
    tau = (ts - ts[0]) / (ts[-1] - ts[0])
    stride_times.append(float(ts[-1] - ts[0]))

    origin = data["com_pos"][i0].copy()
    com_rel = data["com_pos"][i0 : i1 + 1] - origin
    foot_l_rel_origin = data["foot_l_pos"][i0 : i1 + 1] - origin
    foot_r_rel_origin = data["foot_r_pos"][i0 : i1 + 1] - origin
    foot_l_rel_com = (
      data["foot_l_pos"][i0 : i1 + 1] - data["com_pos"][i0 : i1 + 1]
    )
    foot_r_rel_com = (
      data["foot_r_pos"][i0 : i1 + 1] - data["com_pos"][i0 : i1 + 1]
    )

    cycles["com_rel"].append(interp_columns(tau, com_rel, phase))
    cycles["foot_l_rel_origin"].append(
      interp_columns(tau, foot_l_rel_origin, phase)
    )
    cycles["foot_r_rel_origin"].append(
      interp_columns(tau, foot_r_rel_origin, phase)
    )
    cycles["foot_l_rel_com"].append(interp_columns(tau, foot_l_rel_com, phase))
    cycles["foot_r_rel_com"].append(interp_columns(tau, foot_r_rel_com, phase))

    for key in cycle_keys:
      cycles[key].append(interp_columns(tau, data[key][i0 : i1 + 1], phase))

    cycles["contact_l"].append(
      interp_columns(tau, data["contact_l"][i0 : i1 + 1].astype(float), phase)
    )
    cycles["contact_r"].append(
      interp_columns(tau, data["contact_r"][i0 : i1 + 1].astype(float), phase)
    )

  if not stride_times:
    return None

  result: dict[str, np.ndarray] = {
    "phase": phase,
    "mean_stride_time_s": np.array([np.mean(stride_times)], dtype=np.float64),
    "n_cycles": np.array([len(stride_times)], dtype=np.int64),
  }
  for key, values in cycles.items():
    arr = np.stack(values, axis=0)
    result[key + "_mean"] = np.mean(arr, axis=0)
    result[key + "_std"] = np.std(arr, axis=0)
  return result


def save_cycle_csv(
  path: Path,
  cycle: dict[str, np.ndarray],
  joint_names: list[str],
) -> None:
  phase = cycle["phase"]
  stride_time = float(cycle["mean_stride_time_s"][0])
  rows: list[dict[str, Any]] = []
  for i, ph in enumerate(phase):
    row: dict[str, Any] = {
      "phase": float(ph),
      "time_in_stride_s": float(ph * stride_time),
    }
    for axis, name in enumerate(("x", "y", "z")):
      row[f"com_{name}_rel_mean_m"] = float(cycle["com_rel_mean"][i, axis])
      row[f"com_v{name}_mean_mps"] = float(cycle["com_vel_mean"][i, axis])
      row[f"L_foot_{name}_rel_com_mean_m"] = float(
        cycle["foot_l_rel_com_mean"][i, axis]
      )
      row[f"R_foot_{name}_rel_com_mean_m"] = float(
        cycle["foot_r_rel_com_mean"][i, axis]
      )
    row["roll_mean_rad"] = float(cycle["base_rpy_mean"][i, 0])
    row["pitch_mean_rad"] = float(cycle["base_rpy_mean"][i, 1])
    row["yaw_mean_rad"] = float(cycle["base_rpy_mean"][i, 2])
    row["L_contact_probability"] = float(cycle["contact_l_mean"][i])
    row["R_contact_probability"] = float(cycle["contact_r_mean"][i])
    for j, joint in enumerate(joint_names):
      row[f"q_{joint}_mean_rad"] = float(cycle["joint_pos_mean"][i, j])
      row[f"dq_{joint}_mean_radps"] = float(cycle["joint_vel_mean"][i, j])
    rows.append(row)
  write_dict_csv(path, rows)


def save_mpc_reference(
  csv_path: Path,
  npz_path: Path,
  cycle: dict[str, np.ndarray],
) -> None:
  """Save one phase-normalized representative stride for SRBD-MPC."""
  phase = cycle["phase"]
  stride_time = float(cycle["mean_stride_time_s"][0])
  t_stride = phase * stride_time

  contact_l_prob = cycle["contact_l_mean"].reshape(-1)
  contact_r_prob = cycle["contact_r_mean"].reshape(-1)
  contact_l = contact_l_prob >= 0.5
  contact_r = contact_r_prob >= 0.5

  rows: list[dict[str, Any]] = []
  for i in range(len(phase)):
    row: dict[str, Any] = {
      "phase": float(phase[i]),
      "time_s": float(t_stride[i]),
      "contact_L": int(contact_l[i]),
      "contact_R": int(contact_r[i]),
      "contact_L_probability": float(contact_l_prob[i]),
      "contact_R_probability": float(contact_r_prob[i]),
    }
    for axis, name in enumerate(("x", "y", "z")):
      row[f"pCoM_{name}_rel_m"] = float(cycle["com_rel_mean"][i, axis])
      row[f"vCoM_{name}_mps"] = float(cycle["com_vel_mean"][i, axis])
      row[f"pL_{name}_rel_origin_m"] = float(
        cycle["foot_l_rel_origin_mean"][i, axis]
      )
      row[f"pR_{name}_rel_origin_m"] = float(
        cycle["foot_r_rel_origin_mean"][i, axis]
      )
      row[f"pL_{name}_rel_CoM_m"] = float(cycle["foot_l_rel_com_mean"][i, axis])
      row[f"pR_{name}_rel_CoM_m"] = float(cycle["foot_r_rel_com_mean"][i, axis])
      row[f"F_L_{name}_N"] = float(cycle["foot_l_force_mean"][i, axis])
      row[f"F_R_{name}_N"] = float(cycle["foot_r_force_mean"][i, axis])
    row["roll_rad"] = float(cycle["base_rpy_mean"][i, 0])
    row["pitch_rad"] = float(cycle["base_rpy_mean"][i, 1])
    row["yaw_rad"] = float(cycle["base_rpy_mean"][i, 2])
    row["omega_x_radps"] = float(cycle["base_ang_vel_w_mean"][i, 0])
    row["omega_y_radps"] = float(cycle["base_ang_vel_w_mean"][i, 1])
    row["omega_z_radps"] = float(cycle["base_ang_vel_w_mean"][i, 2])
    rows.append(row)
  write_dict_csv(csv_path, rows)

  np.savez_compressed(
    npz_path,
    phase=phase,
    time_s=t_stride,
    stride_time_s=np.array([stride_time]),
    pCoM_rel=cycle["com_rel_mean"],
    vCoM=cycle["com_vel_mean"],
    eulerCoM=cycle["base_rpy_mean"],
    wCoM=cycle["base_ang_vel_w_mean"],
    pL_rel_origin=cycle["foot_l_rel_origin_mean"],
    pR_rel_origin=cycle["foot_r_rel_origin_mean"],
    pL_rel_CoM=cycle["foot_l_rel_com_mean"],
    pR_rel_CoM=cycle["foot_r_rel_com_mean"],
    contact_L=contact_l.astype(np.int8),
    contact_R=contact_r.astype(np.int8),
    contact_L_probability=contact_l_prob,
    contact_R_probability=contact_r_prob,
    F_L=cycle["foot_l_force_mean"],
    F_R=cycle["foot_r_force_mean"],
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Analyze a trained Open Duck gait and export SRBD-MPC references."
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--duration", type=float, default=15.0, help="Rollout seconds")
  parser.add_argument(
    "--warmup", type=float, default=1.0, help="Ignore this initial time in metrics"
  )
  parser.add_argument(
    "--vx", type=float, default=0.15, help="Fixed forward command in m/s"
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--output-dir", type=Path, default=Path("gait_analysis"), help="Output root"
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  args = parser.parse_args()

  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  run_name = checkpoint.parent.name
  out_dir = (args.output_dir / f"{run_name}_{checkpoint.stem}_vx{args.vx:.3f}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  env_cfg.auto_reset = False

  # Force the requested deterministic command.  This is independent of the
  # play default so the analysis command is explicit in every saved result.
  command_cfg = env_cfg.commands["twist"]
  command_cfg.rel_standing_envs = 0.0
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_world_envs = 0.0
  command_cfg.rel_forward_envs = 0.0
  command_cfg.heading_command = False
  command_cfg.ranges.lin_vel_x = (args.vx, args.vx)
  command_cfg.ranges.lin_vel_y = (0.0, 0.0)
  command_cfg.ranges.ang_vel_z = (0.0, 0.0)
  command_cfg.ranges.heading = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  base_env = vec_env.unwrapped
  robot = base_env.scene["robot"]
  contact_sensor = base_env.scene["feet_ground_contact"]

  joint_ids, joint_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
  foot_site_ids, foot_site_names = robot.find_sites(FOOT_SITES, preserve_order=True)
  if len(joint_ids) != 10:
    raise RuntimeError(f"Expected 10 joints, got {joint_names}")
  if len(foot_site_ids) != 2:
    raise RuntimeError(f"Expected 2 foot sites, got {foot_site_names}")

  contact_names = list(contact_sensor.primary_names)
  contact_l_id = resolve_name_index(contact_names, FOOT_GEOMS[0])
  contact_r_id = resolve_name_index(contact_names, FOOT_GEOMS[1])

  root_body_id = robot.indexing.root_body_id
  if isinstance(root_body_id, torch.Tensor):
    root_body_id = int(root_body_id.item())
  else:
    root_body_id = int(root_body_id)

  dt = float(base_env.step_dt)
  n_steps = max(1, int(round(args.duration / dt)))
  obs = vec_env.get_observations()

  records: dict[str, list[np.ndarray | float | bool]] = {
    "time": [],
    "reward": [],
    "command": [],
    "com_pos": [],
    "base_pos": [],
    "base_quat": [],
    "base_rpy": [],
    "base_lin_vel_w": [],
    "base_ang_vel_w": [],
    "joint_pos": [],
    "joint_vel": [],
    "action": [],
    "actuator_force": [],
    "foot_l_pos": [],
    "foot_r_pos": [],
    "foot_l_vel": [],
    "foot_r_vel": [],
    "contact_l": [],
    "contact_r": [],
    "foot_l_force": [],
    "foot_r_force": [],
  }

  terminated_early = False
  termination_time_s: float | None = None

  print(f"[INFO] Checkpoint : {checkpoint}")
  print(f"[INFO] Device     : {args.device}")
  print(f"[INFO] Command vx : {args.vx:.3f} m/s")
  print(f"[INFO] Rollout    : {args.duration:.2f} s, dt={dt:.4f} s")
  print(f"[INFO] Output     : {out_dir}")

  with torch.inference_mode():
    for k in range(n_steps):
      action = policy(obs)
      obs, reward, done, _extras = vec_env.step(action)

      ti = (k + 1) * dt
      root_pose = _np(robot.data.root_link_pose_w[0])
      root_vel = _np(robot.data.root_link_vel_w[0])
      com_pos = _np(robot.data.data.subtree_com[0, root_body_id])
      q = _np(robot.data.joint_pos[0, joint_ids])
      dq = _np(robot.data.joint_vel[0, joint_ids])
      site_pose = _np(robot.data.site_pose_w[0, foot_site_ids])
      site_vel = _np(robot.data.site_vel_w[0, foot_site_ids])
      act_force = _np(robot.data.actuator_force[0])

      cdata = contact_sensor.data
      found = _np(cdata.found[0])
      force = _np(cdata.force[0])
      c_l = bool(found[contact_l_id] > 0)
      c_r = bool(found[contact_r_id] > 0)
      f_l = force[contact_l_id].copy()
      f_r = force[contact_r_id].copy()
      cmd = _np(base_env.command_manager.get_command("twist")[0])

      records["time"].append(ti)
      records["reward"].append(_scalar(reward[0]))
      records["command"].append(cmd.copy())
      records["com_pos"].append(com_pos.copy())
      records["base_pos"].append(root_pose[:3].copy())
      records["base_quat"].append(root_pose[3:7].copy())
      records["base_rpy"].append(quat_wxyz_to_rpy(root_pose[3:7]))
      records["base_lin_vel_w"].append(root_vel[:3].copy())
      records["base_ang_vel_w"].append(root_vel[3:6].copy())
      records["joint_pos"].append(q.copy())
      records["joint_vel"].append(dq.copy())
      records["action"].append(_np(action[0]).copy())
      records["actuator_force"].append(act_force.copy())
      records["foot_l_pos"].append(site_pose[0, :3].copy())
      records["foot_r_pos"].append(site_pose[1, :3].copy())
      records["foot_l_vel"].append(site_vel[0, :3].copy())
      records["foot_r_vel"].append(site_vel[1, :3].copy())
      records["contact_l"].append(c_l)
      records["contact_r"].append(c_r)
      records["foot_l_force"].append(f_l)
      records["foot_r_force"].append(f_r)

      if bool(done[0].item()):
        terminated_early = True
        termination_time_s = ti
        print(f"[WARN] Episode terminated at t={ti:.3f} s; stopping rollout.")
        break

  vec_env.close()

  data: dict[str, np.ndarray] = {}
  for key, values in records.items():
    data[key] = np.asarray(values)
  if len(data["time"]) < 2:
    raise RuntimeError("Rollout produced fewer than 2 samples; cannot analyze gait.")

  # Whole-body CoM velocity from the subtree CoM trajectory.  This is the quantity
  # required by the SRBD state, rather than the root-link velocity.
  data["com_vel"] = np.gradient(data["com_pos"], data["time"], axis=0)

  contact_l_filtered = filter_short_runs(data["contact_l"], min_samples=2)
  contact_r_filtered = filter_short_runs(data["contact_r"], min_samples=2)
  data["contact_l_filtered"] = contact_l_filtered
  data["contact_r_filtered"] = contact_r_filtered

  timeseries_rows: list[dict[str, Any]] = []
  for i, ti in enumerate(data["time"]):
    row: dict[str, Any] = {
      "time_s": float(ti),
      "reward": float(data["reward"][i]),
      "cmd_vx_mps": float(data["command"][i, 0]),
      "cmd_vy_mps": float(data["command"][i, 1]),
      "cmd_wz_radps": float(data["command"][i, 2]),
      "contact_L": int(data["contact_l"][i]),
      "contact_R": int(data["contact_r"][i]),
      "contact_L_filtered": int(contact_l_filtered[i]),
      "contact_R_filtered": int(contact_r_filtered[i]),
    }
    for a, name in enumerate(("x", "y", "z")):
      row[f"pCoM_{name}_m"] = float(data["com_pos"][i, a])
      row[f"vCoM_{name}_mps"] = float(data["com_vel"][i, a])
      row[f"base_{name}_m"] = float(data["base_pos"][i, a])
      row[f"base_v{name}_mps"] = float(data["base_lin_vel_w"][i, a])
      row[f"L_foot_{name}_m"] = float(data["foot_l_pos"][i, a])
      row[f"R_foot_{name}_m"] = float(data["foot_r_pos"][i, a])
      row[f"L_foot_v{name}_mps"] = float(data["foot_l_vel"][i, a])
      row[f"R_foot_v{name}_mps"] = float(data["foot_r_vel"][i, a])
      row[f"F_L_{name}_N"] = float(data["foot_l_force"][i, a])
      row[f"F_R_{name}_N"] = float(data["foot_r_force"][i, a])
    row["base_qw"] = float(data["base_quat"][i, 0])
    row["base_qx"] = float(data["base_quat"][i, 1])
    row["base_qy"] = float(data["base_quat"][i, 2])
    row["base_qz"] = float(data["base_quat"][i, 3])
    row["roll_rad"] = float(data["base_rpy"][i, 0])
    row["pitch_rad"] = float(data["base_rpy"][i, 1])
    row["yaw_rad"] = float(data["base_rpy"][i, 2])
    row["omega_x_radps"] = float(data["base_ang_vel_w"][i, 0])
    row["omega_y_radps"] = float(data["base_ang_vel_w"][i, 1])
    row["omega_z_radps"] = float(data["base_ang_vel_w"][i, 2])
    for j, joint in enumerate(joint_names):
      row[f"q_{joint}_rad"] = float(data["joint_pos"][i, j])
      row[f"dq_{joint}_radps"] = float(data["joint_vel"][i, j])
      row[f"action_{joint}"] = float(data["action"][i, j])
      if j < data["actuator_force"].shape[1]:
        row[f"actuator_force_{joint}"] = float(data["actuator_force"][i, j])
    timeseries_rows.append(row)
  write_dict_csv(out_dir / "timeseries.csv", timeseries_rows)

  event_rows, gait_metrics = build_events(
    data["time"],
    contact_l_filtered,
    contact_r_filtered,
    data["foot_l_pos"],
    data["foot_r_pos"],
    args.warmup,
  )
  write_dict_csv(out_dir / "gait_events.csv", event_rows)

  valid = data["time"] >= args.warmup
  if np.count_nonzero(valid) < 2:
    valid = np.ones_like(data["time"], dtype=bool)
  t_valid = data["time"][valid]
  com_valid = data["com_pos"][valid]
  vcom_valid = data["com_vel"][valid]
  cmd_valid = data["command"][valid]

  elapsed = float(t_valid[-1] - t_valid[0]) if len(t_valid) > 1 else dt
  achieved_speed_x = float((com_valid[-1, 0] - com_valid[0, 0]) / max(elapsed, 1e-9))
  mean_vx = float(np.mean(vcom_valid[:, 0]))
  rmse_vx = float(np.sqrt(np.mean((vcom_valid[:, 0] - cmd_valid[:, 0]) ** 2)))

  summary: dict[str, Any] = {
    "checkpoint": str(checkpoint),
    "task": TASK_ID,
    "command_vx_mps": float(args.vx),
    "policy_dt_s": dt,
    "requested_duration_s": float(args.duration),
    "recorded_duration_s": float(data["time"][-1]),
    "warmup_s": float(args.warmup),
    "terminated_early": bool(terminated_early),
    "termination_time_s": termination_time_s,
    "achieved_speed_x_displacement_mps": achieved_speed_x,
    "mean_vCoM_x_mps": mean_vx,
    "std_vCoM_x_mps": float(np.std(vcom_valid[:, 0])),
    "rmse_vCoM_x_tracking_mps": rmse_vx,
    "forward_displacement_after_warmup_m": float(com_valid[-1, 0] - com_valid[0, 0]),
    "lateral_drift_after_warmup_m": float(com_valid[-1, 1] - com_valid[0, 1]),
    "mean_CoM_height_m": float(np.mean(com_valid[:, 2])),
    "std_CoM_height_m": float(np.std(com_valid[:, 2])),
    "mean_abs_roll_deg": float(np.degrees(np.mean(np.abs(data["base_rpy"][valid, 0])))),
    "mean_abs_pitch_deg": float(np.degrees(np.mean(np.abs(data["base_rpy"][valid, 1])))),
    **gait_metrics,
  }

  cycle = build_mean_cycle(data, event_rows, args.warmup)
  if cycle is not None:
    summary["mean_cycle_available"] = True
    summary["mean_cycle_count"] = int(cycle["n_cycles"][0])
    summary["mean_cycle_stride_time_s"] = float(cycle["mean_stride_time_s"][0])
    save_cycle_csv(out_dir / "gait_cycle_mean.csv", cycle, list(joint_names))
    save_mpc_reference(
      out_dir / "mpc_srbd_reference.csv",
      out_dir / "mpc_srbd_reference.npz",
      cycle,
    )
  else:
    summary["mean_cycle_available"] = False
    summary["mean_cycle_count"] = 0
    summary["mean_cycle_stride_time_s"] = None

  with (out_dir / "gait_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
  write_dict_csv(out_dir / "gait_summary.csv", [summary])

  np.savez_compressed(
    out_dir / "raw_rollout.npz",
    **data,
    joint_names=np.asarray(joint_names),
    foot_site_names=np.asarray(foot_site_names),
    checkpoint=np.asarray([str(checkpoint)]),
    command_vx_mps=np.asarray([args.vx]),
    step_dt_s=np.asarray([dt]),
  )

  print("\n================ GAIT SUMMARY ================")
  print(f"Recorded duration       : {summary['recorded_duration_s']:.3f} s")
  print(f"Achieved forward speed  : {achieved_speed_x:.4f} m/s")
  print(f"Mean CoM vx             : {mean_vx:.4f} m/s")
  print(f"vx tracking RMSE        : {rmse_vx:.4f} m/s")
  print(f"Valid steps             : {summary['n_valid_steps_after_warmup']}")
  print(f"Mean step length        : {summary['mean_step_length_m']}")
  print(f"Mean step time          : {summary['mean_step_time_s']}")
  print(f"Cadence                 : {summary['cadence_steps_per_min']}")
  print(f"Mean stride length      : {summary['mean_stride_length_m']}")
  print(f"Mean stride time        : {summary['mean_stride_time_s']}")
  print(f"L/R duty factor         : {summary['left_duty_factor']} / {summary['right_duty_factor']}")
  print(f"Mean gait cycle ready   : {summary['mean_cycle_available']}")
  print(f"Output directory        : {out_dir}")
  print("================================================")

  if not summary["mean_cycle_available"]:
    print(
      "[WARN] No complete left-foot stride was found after warmup. "
      "Raw data and gait events were still saved, but an SRBD stride template "
      "could not be generated."
    )


if __name__ == "__main__":
  main()
