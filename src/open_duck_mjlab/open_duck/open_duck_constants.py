"""Open Duck Mini V2 robot configuration for MJLab."""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parents[2]
OPEN_DUCK_XML = _HERE / "xmls" / "open_duck_mini_v4.xml"
_PACKAGE_ASSETS = OPEN_DUCK_XML.parent / "assets_v3"
_REPO_ASSETS = _REPO_ROOT / "xmls" / "assets_v3"


JOINT_NAMES = (
  "L_hip_yaw",
  "L_hip_roll",
  "L_hip_pitch",
  "L_knee",
  "L_ankle",
  "R_hip_yaw",
  "R_hip_roll",
  "R_hip_pitch",
  "R_knee",
  "R_ankle",
)

FOOT_SITES = ("L6", "R6")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
FOOT_BODIES = ("L5_ankle", "R5_ankle")
BASE_BODY = "base_link"


def _asset_dir() -> Path:
  """Return the STL directory, preferring package-local assets when present."""
  if _PACKAGE_ASSETS.exists() and any(_PACKAGE_ASSETS.glob("*.STL")):
    return _PACKAGE_ASSETS
  if _REPO_ASSETS.exists() and any(_REPO_ASSETS.glob("*.STL")):
    return _REPO_ASSETS
  raise FileNotFoundError(
    "Open Duck STL assets not found. Expected them in either "
    f"{_PACKAGE_ASSETS} or {_REPO_ASSETS}."
  )


def _inject_mesh_assets(spec: mujoco.MjSpec) -> None:
  """Load the mesh bytes required by the MJCF into ``spec.assets``.

  MJLab 1.6.0 does not provide the newer ``mjlab.utils.os.update_assets`` helper,
  so the referenced mesh files are injected directly using their MJCF file names.
  """
  asset_dir = _asset_dir()
  for mesh in spec.meshes:
    if not mesh.file:
      continue
    mesh_path = asset_dir / Path(mesh.file).name
    if not mesh_path.exists():
      raise FileNotFoundError(
        f"Mesh referenced by MJCF was not found: {mesh.file} (looked in {asset_dir})"
      )
    spec.assets[mesh.file] = mesh_path.read_bytes()


def get_spec() -> mujoco.MjSpec:
  """Load a fresh MuJoCo specification for the robot."""
  if not OPEN_DUCK_XML.exists():
    raise FileNotFoundError(f"Open Duck MJCF not found: {OPEN_DUCK_XML}")
  spec = mujoco.MjSpec.from_file(str(OPEN_DUCK_XML))
  _inject_mesh_assets(spec)
  return spec


# Home pose copied from scene_flat_terrain_v4.xml.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.24),
  joint_pos={
    "L_hip_yaw": 0.0,
    "L_hip_roll": 0.0,
    "L_hip_pitch": 0.626,
    "L_knee": -1.3963,
    "L_ankle": 0.7854,
    "R_hip_yaw": 0.0,
    "R_hip_roll": 0.0,
    "R_hip_pitch": -0.626,
    "R_knee": 1.3963,
    "R_ankle": -0.7854,
  },
  joint_vel={".*": 0.0},
)


# The current v4 model only has explicit collision geometry on the feet.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=FOOT_GEOMS,
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)


# Keep the position actuators defined in the MJCF (STS3215 baseline model).
OPEN_DUCK_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
  soft_joint_pos_limit_factor=0.9,
)


def get_open_duck_robot_cfg() -> EntityCfg:
  """Return a fresh Open Duck Mini V2 EntityCfg."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=OPEN_DUCK_ARTICULATION,
  )


# Joint-position residual action scale in radians.
# Narrow yaw/roll joints receive smaller residuals than pitch/knee/ankle joints.
OPEN_DUCK_ACTION_SCALE = {
  ".*hip_yaw": 0.20,
  ".*hip_roll": 0.15,
  ".*hip_pitch": 0.35,
  ".*knee": 0.45,
  ".*ankle": 0.30,
}
