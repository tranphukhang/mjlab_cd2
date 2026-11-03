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
      assets_v3/                      # Copy STL files here
```

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

## 2. Copy the STL assets

This repository was initialized through GitHub's text-file API, so binary STL files were not copied automatically.

If the original repository is next to this repository:

```bash
cp ../open_duck_mini_v2/xmls/assets_v3/*.STL \
  src/open_duck_mjlab/open_duck/xmls/assets_v3/
```

Or clone the source repository and copy them:

```bash
git clone https://github.com/tranphukhang/open_duck_mini_v2.git /tmp/open_duck_mini_v2
cp /tmp/open_duck_mini_v2/xmls/assets_v3/*.STL \
  src/open_duck_mjlab/open_duck/xmls/assets_v3/
```

Then commit the assets normally with git.

## 3. Install the package

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

## 4. Standalone MuJoCo sanity check

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

## 5. MJLab zero-action smoke test

Run one robot first:

```bash
uv run play Mjlab-Velocity-Flat-OpenDuckMiniV2 \
  --agent zero \
  --env.scene.num-envs 1
```

This test is for integration only. The robot may fall under zero actions depending on actuator tuning; the important checks are that the MJCF compiles, the task registers, observations are finite, and the 10 actuator dimensions are correct.

## 6. Short PPO smoke train

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
