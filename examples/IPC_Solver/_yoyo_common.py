"""
Shared constants and utilities for yoyo replay and showcase scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
YOYO_ASSETS_DIR = _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo" / "results" / "v3"

# Available trajectories
TRAJECTORIES = {
    "v3": _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo" / "results" / "v3" / "seq",
    "long_sleep": _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo" / "results" / "long_sleep" / "seq",
    "long_sleep_full": _REPO_ROOT / "IPC-Samples" / "python" / "Yoyo" / "results" / "long_sleep" / "seq_full",
}
DEFAULT_SEQ_DIR = TRAJECTORIES["v3"]

# Map from seq object names (underscores) to GLB asset names (hyphens)
GLB_NAME_MAP = {
    "yoyo_ball": "yoyo-ball",
}

# Camera pose used by ipc_robot_yoyo (the final pose for ipc_show_yoyo orbit)
REPLAY_CAMERA_POS = (0.55, -0.35, 0.15)
REPLAY_CAMERA_LOOKAT = (0.25, 0.0, -0.05)
REPLAY_CAMERA_UP = (0.0, 0.0, 1.0)
REPLAY_CAMERA_FOV = 35
END_CAMERA_POS = (0.55, 0.35, 0.15)

# Shell surface parameters
# Metallic/roughness are baked into GLB assets via export_yoyo_glb.py
# Only opacity is set at runtime (to selectively make one side transparent)
SHELL_OPACITY_TRANSPARENT = 0.05
INTERNAL_OPACITY_TRANSPARENT = 0.5

# Env map candidates (tried in order, first existing wins)
ENV_MAP_CANDIDATES = [
    _REPO_ROOT / "genesis" / "assets" / "textures" / "indoor_bright.png",
]


def tf_to_pos_quat(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position (3,) and quaternion (w,x,y,z) from a 4x4 transform."""
    pos = tf[:3, 3].copy()
    R = tf[:3, :3]
    r = Rotation.from_matrix(R)
    xyzw = r.as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return pos, quat


def find_env_texture():
    """Find the best available env map texture. Returns a gs texture object."""
    import genesis as gs

    for candidate in ENV_MAP_CANDIDATES:
        if candidate.exists():
            print(f"[render] Using env map: {candidate.name}")
            return gs.textures.ImageTexture(image_path=str(candidate))
    return gs.textures.ColorTexture(color=(0.75, 0.8, 0.9))


def make_raytracer(**kwargs):
    """Create a RayTracer renderer with env map and default lights."""
    import genesis as gs
    from genesis.options.renderers import SphereLight

    env_texture = find_env_texture()
    defaults = dict(
        logging_level="warning",
        tracing_depth=32,
        env_surface=gs.surfaces.Emission(emissive_texture=env_texture),
        env_radius=5.0,
        env_euler=(0, 0, 20),
        lights=[
            SphereLight(pos=(0.3, -0.4, 0.5), radius=0.06, color=(1.0, 0.98, 0.95), intensity=15.0),
            SphereLight(pos=(-0.2, 0.3, 0.3), radius=0.08, color=(0.9, 0.92, 1.0), intensity=8.0),
        ],
    )
    defaults.update(kwargs)
    return gs.renderers.RayTracer(**defaults)


def load_seq_data(seq_dir: Path, meta: dict):
    """Load rigid and FEM trajectory data from a sequence directory.

    Returns (rigid_data, fem_data) dicts mapping name -> numpy array.
    """
    rigid_data: dict[str, np.ndarray] = {}
    fem_data: dict[str, np.ndarray] = {}
    for name, info in meta["objects"].items():
        npy_path = seq_dir / info["data"]
        if not npy_path.exists():
            print(f"[warn] {npy_path} not found, skipping {name}")
            continue
        arr = np.load(str(npy_path))
        if info["type"] == "rigid":
            rigid_data[name] = arr
        else:
            fem_data[name] = arr
    return rigid_data, fem_data


def resolve_mesh_path(name: str, seq_dir: Path) -> Path | None:
    """Resolve a mesh path for a yoyo part, preferring GLB over OBJ."""
    glb_name = GLB_NAME_MAP.get(name, name)
    glb_path = YOYO_ASSETS_DIR / f"{glb_name}.glb"
    obj_path = seq_dir / name / "mesh.obj"
    if glb_path.exists():
        return glb_path
    if obj_path.exists():
        return obj_path
    return None
