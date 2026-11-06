"""Check the left/right mirror against a physically mirrored simulator state.

Run: pixi run python tests/test_symmetry.py
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from tensordict import TensorDict

import open_duck_mjlab  # noqa: F401
from open_duck_mjlab import symmetry
from open_duck_mjlab.env_cfgs import GAIT_PERIOD_S

TASK = "Mjlab-Velocity-Flat-OpenDuckMiniV2-GaitPhase"
POLAR = torch.tensor([1.0, -1.0, 1.0])
AXIAL = torch.tensor([-1.0, 1.0, -1.0])


def main() -> None:
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  dev = env.device
  env.reset()
  torch.manual_seed(0)
  with torch.inference_mode():
    for _ in range(15):  # Random motion so every term is non-trivial.
      env.step(0.5 * torch.randn(env.num_envs, 10, device=dev))

  robot = env.scene["robot"]
  cmd_term = env.command_manager.get_term("twist")
  with torch.inference_mode():
    om = env.observation_manager  # compute() returns a per-step cache.
    obs = {g: om.compute_group(g).clone() for g in ("actor", "critic")}
    root = torch.cat(
      (
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w,
        robot.data.root_link_ang_vel_w,
      ),
      dim=1,
    ).clone()
    q, qd = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()

    # Physically mirror the state about the world x-z plane.
    m_root = root.clone()
    m_root[:, 0:3] *= POLAR.to(dev)
    m_root[:, 3:7] *= torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)  # (w,x,y,z)
    m_root[:, 7:10] *= POLAR.to(dev)
    m_root[:, 10:13] *= AXIAL.to(dev)
    robot.write_root_state_to_sim(m_root)
    robot.write_joint_state_to_sim(
      symmetry.mirror_actions(q), symmetry.mirror_actions(qd)
    )
    cmd_term.vel_command_b[:] *= torch.tensor([1.0, -1.0, -1.0], device=dev)
    env.action_manager.action[:] = symmetry.mirror_actions(env.action_manager.action)
    half_cycle = round(0.5 * GAIT_PERIOD_S / env.step_dt)
    env.episode_length_buf += half_cycle
    env.sim.forward()
    obs_m = {g: om.compute_group(g) for g in ("actor", "critic")}

  ok = True
  for group in ("actor", "critic"):
    pred = symmetry.mirror_obs(env, group, obs[group])
    names = env.observation_manager.active_terms[group]
    dims = [d[0] for d in env.observation_manager.group_obs_term_dim[group]]
    off = 0
    for name, d in zip(names, dims, strict=True):
      sl = slice(off, off + d)
      off += d
      if name.startswith("foot_"):  # Contact/ray sensors only refresh on a step.
        continue
      err = (pred[:, sl] - obs_m[group][:, sl]).abs().max().item()
      flag = "OK " if err < 1e-4 else "BAD"
      ok &= err < 1e-4
      print(f"  {flag} {group:6s} {name:20s} max err {err:.2e}")

  # Involution and augmentation shapes.
  td = TensorDict({k: obs[k] for k in ("actor", "critic")}, batch_size=[env.num_envs])
  aug, act = symmetry.mirror_augmentation(env, td, env.action_manager.action)
  assert aug is not None and act is not None
  assert aug.batch_size[0] == 2 * env.num_envs and act.shape[0] == 2 * env.num_envs
  for group in ("actor", "critic"):
    twice = symmetry.mirror_obs(env, group, symmetry.mirror_obs(env, group, obs[group]))
    ok &= torch.allclose(twice, obs[group])
  # Home pose (zero relative joints) mirrors onto itself.
  ok &= torch.allclose(symmetry.mirror_actions(torch.zeros(1, 10)), torch.zeros(1, 10))
  print("ALL OK" if ok else "FAILED")
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
