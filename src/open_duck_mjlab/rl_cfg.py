"""PPO configuration for Open Duck Mini V2 velocity tracking."""

from dataclasses import asdict, dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class SymmetryCfg:
  """rsl_rl symmetry extension settings (Mittal et al., ICRA 2024)."""

  data_augmentation_func: str = "open_duck_mjlab.symmetry:mirror_augmentation"
  use_data_augmentation: bool = True
  """Append left/right mirrored samples to every PPO mini-batch."""
  use_mirror_loss: bool = True
  mirror_loss_coeff: float = 5.0
  """Weight of the MSE between the policy on mirrored obs and mirrored actions."""


@dataclass
class OpenDuckPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO config plus the ``symmetry_cfg`` rsl_rl already accepts."""

  symmetry_cfg: SymmetryCfg | None = None


def open_duck_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the Stage-0 PPO configuration for MJLab 1.6.0."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="open_duck_velocity",
    run_name="stage0_walk",
    wandb_project="mjlab_open_duck",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1_000,
  )


def open_duck_gait_first_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO configuration for the gait-first curriculum task.

  Logs to TensorBoard so training needs no network connection.
  """
  cfg = open_duck_ppo_runner_cfg()
  cfg.run_name = "gait_first"
  cfg.max_iterations = 3_000
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg


def open_duck_gait_first_zmp_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO configuration for fine-tuning with ZMP and friction-cone penalties."""
  cfg = open_duck_gait_first_ppo_runner_cfg()
  cfg.run_name = "gait_first_zmp"
  cfg.max_iterations = 1_000
  return cfg


def open_duck_gait_phase_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fine-tune of the ZMP policy with the gait clock and contact schedule."""
  cfg = open_duck_gait_first_ppo_runner_cfg()
  cfg.run_name = "gait_phase"
  cfg.max_iterations = 1_500
  # Without symmetry the policy limped (left swing 0.24 s, right 0.14 s) even
  # though the model is mirror-symmetric.
  cfg.algorithm = OpenDuckPpoAlgorithmCfg(
    **asdict(cfg.algorithm), symmetry_cfg=SymmetryCfg()
  )
  return cfg
