from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from . import viser_compat  # noqa: F401
from .env_cfgs import open_duck_flat_env_cfg, open_duck_gait_first_env_cfg
from .rl_cfg import open_duck_gait_first_ppo_runner_cfg, open_duck_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-OpenDuckMiniV2",
  env_cfg=open_duck_flat_env_cfg(),
  play_env_cfg=open_duck_flat_env_cfg(play=True),
  rl_cfg=open_duck_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-OpenDuckMiniV2-GaitFirst",
  env_cfg=open_duck_gait_first_env_cfg(),
  play_env_cfg=open_duck_gait_first_env_cfg(play=True),
  rl_cfg=open_duck_gait_first_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
