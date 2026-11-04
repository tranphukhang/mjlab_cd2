# mjlab_cd2 — Open Duck Mini V2 RL with MJLab

Baseline MJLab workspace for training an Open Duck Mini V2 biped with PPO on flat terrain.

## Current milestone

The first milestone is deliberately simple:

1. load the Open Duck v4 MJCF,
2. spawn one robot,
3. reset to the crouched home pose,
4. verify the 10 position actuators,
5. run the built-in MJLab velocity task,
6. train forward/backward velocity tracking on flat terrain.

Lateral velocity, yaw turning, rough terrain, and MPC integration are intentionally postponed until this baseline works.

## Project structure

```text
src/open_duck_mjlab/
  __init__.py                         # MJLab task registration
  env_cfgs.py                         # Flat velocity task + rewards
  rl_cfg.py                           # PPO configuration
  open_duck/
    __init__.py
    open_duck_constants.py            # EntityCfg, XML actuator wrapper, home pose
    xmls/
      open_duck_mini_v4.xml           # RL-ready Open Duck v4 model
      scene_flat_terrain_v4.xml       # Standalone MuJoCo sanity-check scene

xmls/assets_v3/                       # Existing Open Duck STL assets
```

The loader first looks for package-local `assets_v3/*.STL`; if they are not present it automatically falls back to the existing repository-level `xmls/assets_v3/` directory.

## Robot/action convention

The policy controls 10 joints in this order:

```text
L_hip_yaw
L_hip_roll
L_hip_pitch
L_knee
L_ankle
R_hip_yaw
R_hip_roll
R_hip_pitch
R_knee
R_ankle
```

MJLab uses residual joint-position actions around the home pose:

```text
q_des = q_home + action_scale * action
```

The MJCF keeps the v4 STS3215 baseline actuator model:

```text
kp = 13.37
kv = 0.0
force range = [-3.23, 3.23]
```

The simulation step is kept at `0.0005 s` and `decimation=40`, giving a 50 Hz policy/control rate.

## Baseline reward set

The initial reward recipe is intentionally conservative:

| Term | Weight |
|---|---:|
| linear velocity tracking | +2.0 |
| angular velocity tracking | +2.0 |
| upright | +2.0 |
| home-pose regularization | +1.0 |
| foot air time | +1.0 |
| body angular velocity | -0.05 |
| angular momentum | -0.02 |
| foot clearance | -0.5 |
| foot swing-height error | -0.25 |
| foot slip | -0.1 |
| action rate | -0.1 |
| joint-position limits | -1.0 |

At stage 0 the command distribution is restricted to forward/backward motion:

```text
vx      = [-0.20, 0.25] m/s
vy      = 0
wz      = 0
standing environments = 20%
```

## Pixi + gait-first task (Linux or Windows, NVIDIA GPU)

`pixi.toml` installs Python 3.12, PyTorch (CUDA 12.8, NVIDIA driver >= 570) and
this package in editable mode, on `linux-64` and `win-64` (tested natively on
Windows 11 with an RTX 5080 Laptop GPU, no WSL). The download is about 3.5-4.5 GB,
mostly PyTorch and the NVIDIA CUDA libraries.

Install pixi (then open a new terminal):

```bash
curl -fsSL https://pixi.sh/install.sh | bash                          # Linux
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"   # Windows
```

```bash
pixi install
pixi run check         # prints: cuda True <GPU name>
pixi run smoke         # zero-action play, 1 env
pixi run train-smoke   # 10 PPO iterations, 64 envs
pixi run train         # full run, 8192 envs, 3000 iterations (~75 min on RTX 5080 Laptop)
pixi run tb            # TensorBoard (training logs offline, no W&B)
```

Extra arguments are appended to a task, e.g.
`pixi run train --agent.max-iterations 1500`.

Follow training live (Windows, key metrics only):

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\watch_train.ps1 -Log train.log
```

where `train.log` is the redirected output of `pixi run train > train.log`.

Play a checkpoint in the browser viewer (http://localhost:8080). The viewer's
Checkpoints tab picks up new checkpoints while training is still running, and
Commands → Twist → Enable lets you drive the robot with sliders:

```bash
pixi run play Mjlab-Velocity-Flat-OpenDuckMiniV2-GaitFirst \
  --checkpoint-file logs/rsl_rl/open_duck_velocity/<run>/model_2999.pt \
  --num-envs 16 --viewer viser
```

Use `--viewer native` for a MuJoCo window instead. Record a 10 s video to
`logs/rsl_rl/open_duck_velocity/<run>/videos/play/`:

```bash
pixi run play Mjlab-Velocity-Flat-OpenDuckMiniV2-GaitFirst \
  --checkpoint-file logs/rsl_rl/open_duck_velocity/<run>/model_2999.pt \
  --num-envs 1 --video True --video-length 500 --video-width 1280 --video-height 720
```

`src/open_duck_mjlab/viser_compat.py` works around an mjlab 1.6 bug where the
Viser velocity joystick crashes when a command range max is below 0.1 (our
tasks use zero lateral and yaw ranges).

Reference run (RTX 5080 Laptop, Windows 11, 8192 envs, 3000 iterations,
1 h 15 min): no falls, `error_vel_xy` ~0.09, mean foot air time ~0.095 s.

`Mjlab-Velocity-Flat-OpenDuckMiniV2-GaitFirst` learns stepping first and
velocity tracking second, using reward/command curricula keyed on PPO
iteration (24 env steps each):

| | Phase A: gait (iter 0-400) | Phase B: blend (400-1000) | Phase C: tracking (1000+) |
|---|---|---|---|
| vx command (m/s) | 0.05 - 0.15 | 0.0 - 0.25 | -0.20 - 0.30 |
| standing envs | 0% | 0% | 10% |
| air_time weight | 2.0 | 1.0 | 0.5 |
| single_foot_contact weight | 1.0 | 0.5 | 0.25 |
| foot_swing_height weight | -0.2 | -0.1 | -0.1 |
| track_linear_velocity weight / std | 1.0 / sqrt(0.25) | 3.0 / sqrt(0.10) | 4.0 / sqrt(0.05) |
| action_rate_l2 weight | -0.01 | -0.02 | -0.05 |

In TensorBoard, `Episode_Reward/air_time` and `Episode_Reward/single_foot_contact`
should rise first, then `Episode_Reward/track_linear_velocity` after iteration
~400. `Curriculum/*` shows the active weights.

## 1. Clone and create an environment

Ubuntu + NVIDIA GPU is recommended.

```bash
git clone https://github.com/tranphukhang/mjlab_cd2.git
cd mjlab_cd2

conda create -n mjlab_cd2 python=3.11 -y
conda activate mjlab_cd2
```

Check the GPU:

```bash
nvidia-smi
```

## 2. Install the package

Using uv:

```bash
pip install uv
uv sync
```

Or editable pip installation inside the conda environment:

```bash
pip install -e .
```

Verify imports:

```bash
python -c "import mjlab; import open_duck_mjlab; print('MJLab/OpenDuck import OK')"
```

## 3. Standalone MuJoCo sanity check

Before debugging RL, verify that the MJCF itself compiles:

```bash
python - <<'PY'
import mujoco
from pathlib import Path

xml = Path('src/open_duck_mjlab/open_duck/xmls/scene_flat_terrain_v4.xml')
model = mujoco.MjModel.from_xml_path(str(xml))
print('nq =', model.nq)
print('nv =', model.nv)
print('nu =', model.nu)
print('MJCF compile OK')
PY
```

Expected actuator count:

```text
nu = 10
```

## 4. MJLab zero-action smoke test

Run one robot first:

```bash
uv run play Mjlab-Velocity-Flat-OpenDuckMiniV2 \
  --agent zero \
  --env.scene.num-envs 1
```

This test is for integration only. The robot may fall under zero actions depending on actuator tuning; the important checks are that the MJCF compiles, the task registers, observations are finite, and the 10 actuator dimensions are correct.

## 5. Short PPO smoke train

Do not start with thousands of environments immediately.

```bash
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-OpenDuckMiniV2 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 10
```

Only after this runs cleanly should the number of environments and iterations be increased.

Example next step:

```bash
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-OpenDuckMiniV2 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 3000
```

## Notes on the RL MJCF

`open_duck_mini_v4.xml` is derived from the v4 model in `open_duck_mini_v2` and keeps its kinematics, masses, joint limits, and position actuators. The RL copy only adds interface elements required by MJLab's velocity task:

- `imu_ang_vel` gyro,
- `imu_lin_vel` velocimeter,
- `imu_lin_acc` accelerometer,
- `root_angmom` subtree angular-momentum sensor,
- explicit names for the two foot collision geoms.

These additions are for observation/contact bookkeeping; they do not change the robot's kinematic chain.

## Next milestones

After the flat forward/backward baseline is stable:

1. tune actuator/action scale and reward terms,
2. enable yaw-rate commands,
3. enable lateral commands,
4. add stronger domain randomization,
5. add push recovery,
6. add rough terrain if needed,
7. export/evaluate the policy for sim-to-real,
8. only then study coupling with the existing SRBD-MPC pipeline.
