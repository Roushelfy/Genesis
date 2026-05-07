"""Replay Robowits tabletop manipulation trajectories (MARVIN_PIKA bimanual).

Replays NPZ trajectories (from convert_mcap_to_npz.py) in Genesis with the
correct per-task scene geometry.  Supports all 16 non-excluded Robowits tasks.

Usage
-----
    # Interactive viewer
    python replay_robowits.py --task 02

    # Render to video
    python replay_robowits.py --task 02 --render

    # With Nyx renderer
    python replay_robowits.py --task 02 --render --nyx

    # Custom trajectory path
    python replay_robowits.py --task 02 --traj /path/to/02.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch


class _LightDef(TypedDict):
    pos: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    intensity: float


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]  # Genesis-IPC root
_DEMO = _REPO / "DemoAssets"
_TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"

# Ensure _replay_common is importable
sys.path.insert(0, str(_REPO / "examples" / "IPC_Solver"))
from _replay_common import TrajectoryReplay, marvin_urdf  # noqa: E402

# ---------------------------------------------------------------------------
# Shared assets
# ---------------------------------------------------------------------------

TABLE_GLB = str(_DEMO / "coat_hanger" / "work_table.glb")
MARVIN_PIKA_URDF = marvin_urdf("marvin_pika")

# HuggingFace download helper (lazy, cached)
_HF_CACHE: dict[str, str] = {}


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a single file from HuggingFace, return local path (cached)."""
    key = f"{repo_id}/{filename}"
    if key not in _HF_CACHE:
        from huggingface_hub import hf_hub_download

        _HF_CACHE[key] = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    return _HF_CACHE[key]


def _bk(uuid_glb: str) -> str:
    """Download a BlenderKit asset GLB from HuggingFace."""
    return _hf_download("Genesis-Intelligence/blender_kit", uuid_glb)


def _rw(path: str) -> str:
    """Download a RoboWits customized asset GLB from HuggingFace."""
    return _hf_download("Genesis-Intelligence/RoboWits_customized_assets", path)


def _envmap(filename: str) -> str:
    """Download a 4K HDR envmap from Digital_twin_asset, return local path."""
    return _hf_download("Genesis-Intelligence/Digital_twin_asset", f"envmap/4K/{filename}")


# ---------------------------------------------------------------------------
# Classic mode (--classic): old MarvinPika URDF + polyhaven small_empty_room_1
# ---------------------------------------------------------------------------

# polyhaven CC0; downloaded on first use, cached under DemoAssets/_polyhaven_cache/.
_POLYHAVEN_4K_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/small_empty_room_1_4k.exr"
_CLASSIC_ENVMAP_NAME = "small_empty_room_1_4k.exr"
_CLASSIC_ENVMAP_REGISTRY: tuple[float, float] = (320.0, 1.0)  # (yaw, multiplier)

# Local archived old MarvinPika URDF — predates the new camera-base assembly.
CLASSIC_MARVIN_PIKA_URDF = str(
    _REPO / "data" / "archive_marvin_demoassets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
)


def _classic_envmap_path() -> str:
    """Return the cached path to the polyhaven small_empty_room_1 EXR."""
    cache_dir = _DEMO / "_polyhaven_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _CLASSIC_ENVMAP_NAME
    if not target.exists():
        import urllib.request

        print(f"[classic] downloading {_POLYHAVEN_4K_URL} -> {target}")
        urllib.request.urlretrieve(_POLYHAVEN_4K_URL, target)
    return str(target)


# Mirror of gs-core/.../envmaps/registry.py: per-EXR (rotation_deg, multiplier).
# Keyed by filename so the task -> envmap mapping below stays orthogonal to
# the registry values.
_ENVMAP_REGISTRY: dict[str, tuple[float, float]] = {
    "abandoned_factory_canteen_02_4k.exr": (0.0, 1.0),
    "abandoned_greenhouse_4k.exr": (0.0, 1.0),
    "abandoned_hall_01_4k.exr": (0.0, 1.0),
    "art_studio_4k.exr": (0.0, 1.0),
    "ballroom_4k.exr": (0.0, 1.0),
    "blue_photo_studio_4k.exr": (0.0, 1.0),
    "bright_church_1.exr": (0.0, 1e-2),
    "bright_church_3.exr": (0.0, 1e-1),
    "bright_house_garage.exr": (0.0, 1e-2),
    "bright_house_living_room_2.exr": (0.0, 1e-1),
    "bright_house_living_room_3.exr": (0.0, 1e-1),
    "bright_house_staircase.exr": (0.0, 1e-1),
    "bright_labratory.exr": (0.0, 1.0),
    "bright_library.exr": (0.0, 1e-1),
    "bright_school_science_room.exr": (0.0, 1e-1),
    "brown_photostudio_02_4k.exr": (0.0, 1.0),
    "brown_photostudio_05_4k.exr": (0.0, 1.0),
    "brown_photostudio_06_4k.exr": (0.0, 1.0),
    "brown_photostudio_07_4k.exr": (0.0, 1.0),
    "burnt_warehouse_4k.exr": (0.0, 1.0),
    "empty_warehouse_01_4k.exr": (0.0, 1.0),
    "fireplace_4k.exr": (0.0, 1.0),
    "gear_store_4k.exr": (0.0, 1.0),
    "lythwood_lounge_4k.exr": (0.0, 1.0),
    "machine_shop_01_4k.exr": (0.0, 1.0),
    "machine_shop_03_4k.exr": (0.0, 1.0),
    "marry_hall_4k.exr": (0.0, 1.0),
    "mirrored_hall_4k.exr": (0.0, 1.0),
    "neon_photostudio_4k.exr": (0.0, 1.0),
    "paris_robot_room_softbox.exr": (0.0, 4.0),
    "poly_haven_studio_4k.exr": (0.0, 1.0),
    "robot_room_all_lights_16k.exr": (0.0, 1.0),
    "robot_room_spot01_16k.exr": (0.0, 1.0),
    "robot_room_spot02_16k.exr": (0.0, 1.0),
    "robot_room_top_light_16k.exr": (0.0, 1.0),
    "robot_room_window_16k.exr": (0.0, 1.0),
    "san_carlos_left_marvin_modified.exr": (0.349066, 1.0),
    "small_hangar_01_4k.exr": (0.0, 1.0),
    "unfinished_office_night_4k.exr": (0.0, 1.0),
}

# ---------------------------------------------------------------------------
# Lighting tuning knobs (all dimming/scaling scalars in one place)
# ---------------------------------------------------------------------------
# Global dim on the registry envmap multiplier so the HDR sky doesn't
# overpower our key/fill/rim sphere lights under exposure=0.5/aces.
_OVERALL_INTENSITY = 0.5
# Global dim on per-light Luisa intensity (applied in `_LIGHTS` below).
_ENV_LIGHT_EFFECT = 0.1
# Luisa SphereLight intensity vs Nyx PointLight intensity use different
# physical units; this scalar maps Luisa values to Nyx so the two renderers
# look comparably bright. Pos/radius/color/type pass through unchanged.
_LUISA_TO_NYX_INTENSITY_SCALE = 0.3

# Task -> EXR filename (key into _ENVMAP_REGISTRY above).
_TASK_ENVMAP: dict[str, str] = {
    "02": "brown_photostudio_07_4k.exr",
    "03": "machine_shop_01_4k.exr",
    "04": "poly_haven_studio_4k.exr",
    "06": "art_studio_4k.exr",
    "09": "lythwood_lounge_4k.exr",
    "11": "empty_warehouse_01_4k.exr",
    "13": "brown_photostudio_05_4k.exr",
    "14": "blue_photo_studio_4k.exr",
    "16": "gear_store_4k.exr",
    "17": "small_hangar_01_4k.exr",
    "18": "abandoned_factory_canteen_02_4k.exr",
    "19": "ballroom_4k.exr",
    "24": "machine_shop_03_4k.exr",
    "27": "abandoned_hall_01_4k.exr",
    "28": "marry_hall_4k.exr",
    "29": "brown_photostudio_06_4k.exr",
}


# ---------------------------------------------------------------------------
# Per-task entity definitions
# ---------------------------------------------------------------------------
# Each task maps to a list of entity dicts.  Keys:
#   name     : entity name (must match NPZ rigid_{name} key after s/ /_/g)
#   morph    : callable returning gs.morphs.* (deferred so gs is imported lazily)
#   material : callable returning gs.materials.* (or None for default Rigid)
#   surface  : callable returning gs.surfaces.* (or None for default)


def _task_02():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "deep narrow container",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=str(_DEMO / "repaired" / "deep_narrow_container.glb"),
                scale=0.6,
                pos=(0.50, 0.00, 0.83992),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Glass(color=(0.9, 0.95, 1.0), double_sided=True),
        },
        {
            "name": "cube",
            "morph": gs.morphs.Box(
                pos=(0.50, 0.00, 0.79), euler=(0, 0, 0), size=(0.015, 0.015, 0.015), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(color=(0.8, 0.6, 0.4), double_sided=True),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.68, 0.12, 0.761), euler=(0, 0, 0), size=(0.12, 0.12, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.1, 0.8, 0.1), double_sided=True),
        },
    ]


def _task_03():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "left boundary block",
            "morph": gs.morphs.Box(
                pos=(0.40, -0.056, 0.86), euler=(0, 0, 0), size=(0.14, 0.05, 0.2), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=500.0, friction=1.0),
            "surface": gs.surfaces.Rough(color=(0.6, 0.6, 0.6), double_sided=False),
        },
        {
            "name": "right boundary block",
            "morph": gs.morphs.Box(
                pos=(0.40, 0.056, 0.86), euler=(0, 0, 0), size=(0.14, 0.05, 0.2), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=500.0, friction=1.0),
            "surface": gs.surfaces.Rough(color=(0.6, 0.6, 0.6), double_sided=False),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.60, 0.00, 0.7605), euler=(0, 0, 0), size=(0.12, 0.10, 0.001), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(rho=100.0, friction=0.5),
            "surface": gs.surfaces.Default(color=(0.1, 0.8, 0.1), roughness=0.9),
        },
        {
            "name": "lemon",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("c45b108b-2163-469c-8ed4-dcb82260d83f/obj.glb"),
                scale=0.6,
                pos=(0.40, 0.0, 0.77964),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=1200.0, friction=0.8),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


def _task_04():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "bank card",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("9b4245ca-361f-4940-a23d-090b1e547a52/obj.glb"),
                scale=(0.8, 0.8, 1.2),
                pos=(0.46, 0.0, 0.843),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.4),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb"),
                scale=2,
                pos=(0.48, 0.08, 0.85),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "small table",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("17674e05-b713-4b2a-a04c-49aee4d4d401/obj.glb"),
                scale=0.5,
                pos=(0.46, 0, 0.76338),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "target cube",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.86), euler=(0, 0, 0), size=(0.15, 0.15, 0.2), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.4),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.9592), euler=(0, 0, 0), size=(0.15, 0.15, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4),
        },
    ]


def _task_06():
    import genesis as gs

    return [
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.96, 0.0, 0.761), euler=(0, 0, 0), size=(0.06, 0.10, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.1, 0.7, 0.2), double_sided=False),
        },
        {
            "name": "white block 4",
            "morph": gs.morphs.Box(
                pos=(0.50, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        },
        {
            "name": "white block 3",
            "morph": gs.morphs.Box(
                pos=(0.735, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        },
        {
            "name": "white block 2",
            "morph": gs.morphs.Box(
                pos=(0.795, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        },
        {
            "name": "white block 1",
            "morph": gs.morphs.Box(
                pos=(0.855, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.95, 0.95), double_sided=False),
        },
        {
            "name": "red block",
            "morph": gs.morphs.Box(
                pos=(0.915, 0.0, 0.805), euler=(0, 0, 0), size=(0.02, 0.05, 0.09), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Smooth(color=(0.9, 0.1, 0.1), double_sided=False),
        },
    ]


def _task_09():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "cup",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb"),
                scale=1.1,
                pos=(0.38, 0.00, 0.76 + 0.0459 * 1.1),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "slope",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=str(_DEMO / "repaired" / "slope.glb"),
                scale=(0.5, 1.5, 0.6),
                pos=(0.58, 0.0, 0.76 + 0.0586 * 0.5),
                euler=(0, 0, 90),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "coaster",
            "morph": gs.morphs.Cylinder(
                pos=(0.36, -0.12, 0.764), euler=(0, 0, 0), radius=0.04, height=0.02, fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=1.5),
            "surface": gs.surfaces.Default(color=(0.6, 0.4, 0.2), roughness=0.7),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.60, 0.0, 0.79988), euler=(-10.3, 0.0, 0.0), size=(0.13, 0.16, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.2),
        },
    ]


def _task_10():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "container",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("382432e6-4f0d-44b6-98f9-2f3a013a47e2/obj.glb"),
                scale=1.2,
                pos=(0.58, 0.00, 0.81892),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "dustpan",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/dustpan.STL"),
                scale=0.001,
                pos=(0.67, 0.35, 0.9696),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(color=(0.9, 0.8, 0.6), double_sided=True),
        },
        {
            "name": "screw",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb"),
                scale=1.5,
                pos=(0.56, -0.16, 0.7674),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(color=(0.6, 0.7, 0.9), double_sided=True),
        },
    ]


def _task_11():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "tall cardboard box",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("8fb31a9b-33d5-4246-997f-84307520c1a0/obj.glb"),
                scale=1.2,
                pos=(0.42, 0.0, 0.8422),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "support block left",
            "morph": gs.morphs.Box(
                pos=(0.55, -0.15, 0.8025), euler=(0, 0, 0), size=(0.10, 0.06, 0.085), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6),
        },
        {
            "name": "support block right",
            "morph": gs.morphs.Box(
                pos=(0.55, 0.15, 0.8025), euler=(0, 0, 0), size=(0.10, 0.06, 0.085), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.6, 0.6), roughness=0.6),
        },
        {
            "name": "beam plank",
            "morph": gs.morphs.Box(
                pos=(0.55, 0.0, 0.855), euler=(0, 0, 0), size=(0.12, 0.36, 0.02), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.7, 0.55, 0.3), roughness=0.7),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.55, 0.0, 0.761), euler=(0, 0, 0), size=(0.10, 0.12, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4),
        },
    ]


def _task_13():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    lid_file = _bk("c1a05823-5cb6-4e64-b3f3-5fd5a86cfc0b/obj.glb")
    return [
        {
            "name": "pot without lid",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("3aee9353-e21a-40d4-b160-e2a0af1fff7f/obj.glb"),
                scale=0.208,
                pos=(0.52, 0.0, 0.8292),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "small lid",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=lid_file,
                scale=(0.142, 0.142, 0.5),
                pos=(0.36, -0.12, 0.7668),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(rho=500, friction=5.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "lid medium",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=lid_file,
                scale=(0.221, 0.221, 0.5),
                pos=(0.70, 0.14, 0.7707),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(rho=500, friction=5.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "lid large",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=lid_file,
                scale=(0.315, 0.31, 0.55),
                pos=(0.36, 0.44, 0.7752),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(rho=500, friction=5.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


def _task_14():
    import genesis as gs

    return [
        {
            "name": "apex target",
            "morph": gs.morphs.Cylinder(
                pos=(0.605, 0.0, 0.761), euler=(0, 0, 0), radius=0.015, height=0.002, fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(rho=200.0, friction=0.8),
            "surface": gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.4),
        },
        {
            "name": "base cube 1",
            "morph": gs.morphs.Box(
                pos=(0.52, -0.05, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=1.0),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        },
        {
            "name": "base cube 2",
            "morph": gs.morphs.Box(
                pos=(0.68, 0.05, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=1.0),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        },
        {
            "name": "apex cube",
            "morph": gs.morphs.Box(
                pos=(0.58, 0.12, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=1.0),
            "surface": gs.surfaces.Default(color=(0.9, 0.1, 0.1), roughness=0.4),
        },
    ]


def _task_16():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "target patch",
            "morph": gs.morphs.Cylinder(
                pos=(0.52, 0.0, 0.7625), euler=(0, 0, 0), radius=0.10, height=0.005, fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.0, 0.8, 0.1), roughness=0.6),
        },
        {
            "name": "stabilizing ring",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("adbecefa-2e19-4438-b6e1-59e6f4390122/obj.glb"),
                scale=(0.7, 0.8, 0.7),
                pos=(0.45, -0.15, 0.808),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "tall object with curved bottom",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("4c5e9552-87c0-46e3-8e40-8c736fee4ff1/obj.glb"),
                scale=2,
                pos=(0.60, 0.2, 0.768),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


def _task_17():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "tower base",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/tower_base.glb"),
                scale=(1.0, 1.5, 1.0),
                pos=(0.605, 0.0, 0.803),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "large ring",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/large_ring.glb"),
                scale=(1.2, 1.5, 1.2),
                pos=(0.605, 0.0, 0.802),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "medium ring",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/medium_ring.glb"),
                scale=(1.2, 1.5, 1.2),
                pos=(0.605, 0.0, 0.835),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "small ring",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/small_ring.glb"),
                scale=(1.2, 1.5, 1.2),
                pos=(0.69, 0.08, 0.7704),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
                decimate=True,
                decimate_face_num=100,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "ball",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb"),
                scale=0.2,
                pos=(0.34, 0.0, 0.773),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


def _task_18():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    coacd_fine = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=200, max_convex_hull=50, decimate=True)
    return [
        {
            "name": "collection_zone",
            "morph": gs.morphs.Box(
                pos=(0.55, 0.0, 0.761), euler=(0, 0, 0), size=(0.22, 0.18, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        },
        {
            "name": "hole_plate",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd_fine,
                file=str(_DEMO / "repaired" / "holeplate.glb"),
                scale=(1.2, 1.2, 1.0),
                pos=(0.55, 0.0, 0.7823),
                euler=(0, 90, -90),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "cylindrical_peg",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("ecfc80a0-4318-4ac8-8ec0-fa1e355d1521/obj.glb"),
                scale=0.8,
                pos=(0.67, -0.18, 0.78),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=200.0, friction=0.8),
            "surface": gs.surfaces.Smooth(color=(0.8, 0.8, 0.0), double_sided=True),
        },
    ]


def _task_19():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    bowl_file = _bk("c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb")
    return [
        {
            "name": "large bowl",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=bowl_file,
                scale=1.688,
                pos=(0.605, 0.0, 0.8204),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "medium bowl",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=bowl_file,
                scale=1.227,
                pos=(0.40, 0.10, 0.8042),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


def _task_24():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    cup_file = _bk("94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb")
    return [
        {
            "name": "heavy book",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("196d92f1-10b5-4563-ae7f-0b26b615ce51/obj.glb"),
                scale=0.7,
                pos=(0.70, -0.32, 0.8042),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=100),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "cup support 1",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=cup_file,
                scale=1.0,
                pos=(0.64, -0.10, 0.8082),
                euler=(0, 180, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "cup support 2",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=cup_file,
                scale=1.0,
                pos=(0.64, 0.10, 0.8082),
                euler=(180, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "support cube 1",
            "morph": gs.morphs.Box(
                pos=(0.57, 0.0, 0.3), euler=(0, 0, 0), size=(0.02, 0.02, 0.03), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(friction=1),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        },
        {
            "name": "support cube 2",
            "morph": gs.morphs.Box(
                pos=(0.68, -0.32, 0.775), euler=(0, 0, 0), size=(0.02, 0.02, 0.03), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(friction=1),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.5),
        },
    ]


def _task_27():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "green target area",
            "morph": gs.morphs.Box(
                pos=(0.50, 0.0, 0.7625), euler=(0, 0, 0), size=(0.18, 0.18, 0.005), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.6),
        },
        {
            "name": "long rod",
            "morph": gs.morphs.Box(
                pos=(0.52, -0.15, 0.77), euler=(0, 90, 0), size=(0.02, 0.02, 0.40), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=(0.8, 0.7, 0.5), double_sided=False),
        },
        {
            "name": "hollow roll",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("a0c77eb5-d5b7-4754-8122-3badaf242b7e/obj.glb"),
                scale=1.2,
                pos=(0.88, 0.0, 0.76 + 0.0695 * 1.2),
                euler=(0, 90, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=10.0, friction=1.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


def _task_28():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    slope_file = str(_DEMO / "repaired" / "slope.glb")
    return [
        {
            "name": "cube 1",
            "morph": gs.morphs.Box(
                pos=(0.40, -0.10, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=0.01),
            "surface": gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.5),
        },
        {
            "name": "cube 2",
            "morph": gs.morphs.Box(
                pos=(0.44, 0.00, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=0.01),
            "surface": gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.5),
        },
        {
            "name": "cube 3",
            "morph": gs.morphs.Box(
                pos=(0.48, 0.10, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=0.01),
            "surface": gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.5),
        },
        {
            "name": "goal line",
            "morph": gs.morphs.Box(
                pos=(0.9, 0.0, 0.761), euler=(0, 0, 0), size=(0.01, 1.30, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(friction=0.5),
            "surface": gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.3),
        },
        {
            "name": "smooth slope",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=slope_file,
                scale=0.64,
                pos=(0.66, -0.16, 0.7934),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.01),
            "surface": gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True),
        },
        {
            "name": "rough slope",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=slope_file,
                scale=0.64,
                pos=(0.66, 0.16, 0.7934),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.5),
            "surface": gs.surfaces.Rough(color=(0.5, 0.4, 0.3), double_sided=True),
        },
    ]


def _task_29():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "small wall",
            "morph": gs.morphs.Box(
                pos=(0.31, 0.0, 0.82), euler=(0, 0, 0), size=(0.02, 0.30, 0.12), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(rho=400.0),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.6),
        },
        {
            "name": "board",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=str(_DEMO / "repaired" / "cutting_board.glb"),
                scale=0.8,
                pos=(0.535, 0.0, 0.8009),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=100),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "support cube",
            "morph": gs.morphs.Box(
                pos=(0.515, 0.0, 0.78), euler=(0, 0, 0), size=(0.04, 0.04, 0.04), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(friction=1),
            "surface": gs.surfaces.Default(color=(0.95, 0.95, 0.95), roughness=0.5),
        },
    ]


# ---------------------------------------------------------------------------
# Tasks 01/05/07/12/23/26 — recovered from `mcap_converted_0301/<NN>_SUC_raw/`.
# These MCAPs use the `we.v2` schema (newer than the lattice.v1 set used by the
# 16 tasks above), so `convert_mcap_to_npz.py` auto-detects the format.  The
# scene definitions below are ported from gs-core-robowits source files at
# `gs_env_sim/envs/robowits/{01,05,07,12,23,26}_*.py` so geometry matches what
# the trajectory was recorded against.  The quat-correction logic in
# apply_frame handles any pure-yaw difference between the random initial pose
# of the source env and the natural euler set on the morph.
# ---------------------------------------------------------------------------


def _task_01():
    """01_ruler_align_cubes — push 3 colored cubes against a long rigid ruler."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "long rigid ruler",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb"),
                scale=1.2,
                pos=(0.43, 0.0, 0.76 + 0.0097 * 1.2),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "first target cube",
            "morph": gs.morphs.Box(
                pos=(0.56, -0.05, 0.785),
                euler=(0, 0, 0),
                size=(0.05, 0.05, 0.05),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        },
        {
            "name": "second target cube",
            "morph": gs.morphs.Box(
                pos=(0.62, 0.12, 0.785),
                euler=(0, 0, 0),
                size=(0.05, 0.05, 0.05),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        },
        {
            "name": "third target cube",
            "morph": gs.morphs.Box(
                pos=(0.48, 0.03, 0.785),
                euler=(0, 0, 0),
                size=(0.05, 0.05, 0.05),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.7, 0.9), roughness=0.5),
        },
    ]


def _task_05():
    """05_box_into_basket — place a box into a high basket while a long board leans."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "long board",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("08893dc4-bfb1-49ca-9e47-4b3958a21e4b/obj.glb"),
                scale=(0.1567, 0.151, 0.214),
                pos=(0.50, -0.15, 0.77),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "high basket",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("c6009731-c1d9-48f9-9486-1d5754c336d9/obj.glb"),
                scale=0.611,
                pos=(0.595, 0.04, 0.806),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "box",
            "morph": gs.morphs.Box(
                pos=(0.395, 0.105, 0.81),
                euler=(0, 0, 0),
                size=(0.1, 0.1, 0.1),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(friction=1),
            "surface": gs.surfaces.Smooth(color=(1.0, 0.5, 0.0), double_sided=True),
        },
    ]


def _task_07():
    """07_assemble_pages_with_bar — align 2 eyelet pages and bind with a stabilizing bar."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "nope",
            "morph": gs.morphs.Box(
                pos=(0.0, 0.0, 0.2),
                size=(0.01, 0.01, 0.01),
                fixed=True,
                collision=False,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(),
        },
        {
            "name": "page A with eyelets",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/pageA.glb"),
                scale=0.6,
                pos=(0.40, -0.07, 0.796),
                euler=(0.0, 90.0, 180.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        },
        {
            "name": "page B with eyelets",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/pageB.glb"),
                scale=0.6,
                pos=(0.4, 0.07, 0.796),
                euler=(0.0, 90.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.7, 0.7, 0.7), double_sided=True),
        },
        {
            "name": "stabilizing bar",
            "morph": gs.morphs.Box(
                pos=(0.4, 0.0, 0.796),
                euler=(0.0, 0.0, 0.0),
                size=(0.2, 0.015, 0.015),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=500.0, friction=0.8),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.2, ior=1.5),
        },
    ]


def _task_12():
    """12_pour_ball_through_funnel — guide a small ball through a tilted funnel into a container."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "red bottle",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_bk("ffb3fbe7-1355-465f-8750-475210d8c949/obj.glb"),
                scale=(1.4, 1.4, 1.0),
                pos=(0.50, 0.00, 0.88026),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
                coacd_options=gs.options.CoacdOptions(
                    threshold=0.01, preprocess_resolution=150, max_convex_hull=50, decimate=True
                ),
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Glass(color=(0.8, 0.2, 0.2), ior=1.5, double_sided=True),
        },
        {
            "name": "funnel",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/funnel.glb"),
                scale=(0.7, 0.7, 1.1),
                pos=(0.64, -0.08, 0.825),
                euler=(0.0, -90.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "ball container",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb"),
                scale=1.1,
                pos=(0.50, 0.14, 0.76 + 0.0459 * 1.1),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Plastic(color=(0.8, 0.9, 1.0), opacity=0.2, double_sided=True),
        },
        {
            "name": "small ball",
            "morph": gs.morphs.Sphere(
                pos=(0.50, 0.14, 0.85),
                radius=0.005,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.2, 0.6, 0.9), double_sided=True),
        },
    ]


def _task_23():
    """23_place_book_on_shelf — slot a book into a 5-slot bookshelf."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    book_glb = _bk("ee74f10e-acc7-4f5a-80e2-1b2999c6743e/obj.glb")
    shelf_color = (27 / 255, 103 / 255, 235 / 255)
    return [
        {
            "name": "shelf left",
            "morph": gs.morphs.Box(
                pos=(0.58, 0.15, 0.86),
                euler=(0, 0, 0),
                size=(0.16, 0.02, 0.2),
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=450.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=shelf_color, double_sided=True),
        },
        {
            "name": "shelf right",
            "morph": gs.morphs.Box(
                pos=(0.58, -0.15, 0.86),
                euler=(0, 0, 0),
                size=(0.16, 0.02, 0.2),
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=450.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=shelf_color, double_sided=True),
        },
        {
            "name": "shelf back",
            "morph": gs.morphs.Box(
                pos=(0.67, 0.0, 0.86),
                euler=(0, 0, 0),
                size=(0.02, 0.32, 0.2),
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=450.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=shelf_color, double_sided=True, opacity=0.2),
        },
        {
            "name": "shelf top",
            "morph": gs.morphs.Box(
                pos=(0.62, 0.0, 0.96),
                euler=(0, 0, 0),
                size=(0.1, 0.32, 0.02),
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=450.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=shelf_color, double_sided=True),
        },
        {
            "name": "book on shelf 1",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.6, -0.077, 0.83),
                euler=(-43.5, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "book on shelf 2",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.6, -0.03, 0.83),
                euler=(-43.3, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "book on shelf 3",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.6, 0.038, 0.83),
                euler=(-15.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "book on shelf 4",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.6, 0.07, 0.83),
                euler=(-15.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "book on shelf 5",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.6, 0.123, 0.83),
                euler=(-1.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "book",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=book_glb,
                scale=(0.4, 0.4, 0.4),
                pos=(0.4, 0.0, 0.83),
                euler=(0.0, 0.0, 0.0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=0.5),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "board",
            "morph": gs.morphs.Box(
                pos=(0.2, 0.2, 0.96),
                euler=(0, 0, 0),
                size=(0.2, 0.03, 0.03),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=(0.3, 0.3, 0.3), double_sided=True),
        },
    ]


def _task_26():
    """26_align_chopsticks_with_ruler — line up 6 chopsticks against a ruler on a board."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    chopstick_glb = _rw("toolbench/chopstick.glb")
    dis = 0.2
    dis2 = 0.2
    return [
        {
            "name": "board",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb"),
                scale=1.0,
                pos=(0.54 + dis2, 0.0, 0.76 + 0.01),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "ruler",
            "morph": gs.morphs.Box(
                pos=(0.56, 0.33, 0.8),
                euler=(0, 0, 0),
                size=(0.2, 0.03, 0.03),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=50.0, friction=1.0),
            "surface": gs.surfaces.Smooth(color=(0.3, 0.3, 0.3), double_sided=True),
        },
        {
            "name": "first chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, -0.02, 0.85),
                euler=(0, 0, 180),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "second chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, -0.02, 0.82),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "third chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, -0.00, 0.85),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "fourth chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, 0.00, 0.82),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "fifth chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, 0.02, 0.82),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "sixth chopstick",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=chopstick_glb,
                scale=(1.2, 2.8, 2.8),
                pos=(0.335 + dis, 0.02, 0.85),
                euler=(0, 0, 0),
                fixed=False,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


# ---------------------------------------------------------------------------
# Variant builders (suc_teleop _add*/_trap*/_pivot*)
#
# Each mirrors the gs-core-robowits per-variant env script
# (env/sim/src/gs_env_sim/envs/robowits/<NN>/<NN>_*_<Variant>.py), which
# fully overrides _add_custom_entities (no super call) — so the entity list
# below is the exact recording-time scene.
# ---------------------------------------------------------------------------


def _task_01_add1():
    """01_add1 — align cubes with paper cup + toy duck distractors."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "first target cube",
            "morph": gs.morphs.Box(
                pos=(0.54, 0.02, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.6, ior=1.5),
        },
        {
            "name": "second target cube",
            "morph": gs.morphs.Box(
                pos=(0.60, -0.10, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.6, ior=1.5),
        },
        {
            "name": "third target cube",
            "morph": gs.morphs.Box(
                pos=(0.48, 0.14, 0.785), euler=(0, 0, 0), size=(0.05, 0.05, 0.05), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.2, 0.95), roughness=0.6, ior=1.5),
        },
        {
            "name": "long rigid ruler",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb"),
                scale=1.0,
                pos=(0.38, 0.0, 0.7708),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "paper cup",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("69730348-4786-4ef3-b7a5-6d4572e43811/obj.glb"),
                scale=1.0,
                pos=(0.68, -0.05, 0.8216),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "toy duck",
            "morph": gs.morphs.Sphere(
                pos=(0.34, 0.12, 0.79), euler=(0, 0, 0), radius=0.03, fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.95, 0.9, 0.2), roughness=0.4, ior=1.5),
        },
    ]


def _task_01_trap2():
    """01_trap2 — align (smaller) cubes with ruler + dinner knife distractor."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "first target cube",
            "morph": gs.morphs.Box(
                pos=(0.45, 0.00, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(rho=500.0),
            "surface": gs.surfaces.Default(color=(0.9, 0.2, 0.2), roughness=0.6, ior=1.5),
        },
        {
            "name": "second target cube",
            "morph": gs.morphs.Box(
                pos=(0.60, -0.10, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(rho=500.0),
            "surface": gs.surfaces.Default(color=(0.2, 0.9, 0.2), roughness=0.6, ior=1.5),
        },
        {
            "name": "third target cube",
            "morph": gs.morphs.Box(
                pos=(0.35, 0.15, 0.7825), euler=(0, 0, 0), size=(0.045, 0.045, 0.045), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(rho=500.0),
            "surface": gs.surfaces.Default(color=(0.2, 0.2, 0.9), roughness=0.6, ior=1.5),
        },
        {
            "name": "long rigid ruler",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("33679be6-fc3f-40e0-ae2b-c6329d2d0ac8/obj.glb"),
                scale=1.0,
                pos=(0.52, -0.43, 0.7697),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=800.0, friction=0.8),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "dinner knife",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("07318397-04f4-42bb-8a25-b6d78f4a2a28/obj.glb"),
                scale=1.0,
                pos=(0.68, 0.35, 0.7639),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=780.0, friction=0.6),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


def _task_04_add1():
    """04_add1 — pinch_card with wooden pencil case + candle pot distractors."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "bank card",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("9b4245ca-361f-4940-a23d-090b1e547a52/obj.glb"),
                scale=0.8,
                pos=(0.46, 0.0, 0.843),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.4),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb"),
                scale=2,
                pos=(0.48, 0.08, 0.85),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "small table",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("17674e05-b713-4b2a-a04c-49aee4d4d401/obj.glb"),
                scale=0.5,
                pos=(0.46, 0, 0.76338),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "target cube",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.86), euler=(0, 0, 0), size=(0.15, 0.15, 0.2), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.4, ior=1.5),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.96), euler=(0, 0, 0), size=(0.15, 0.15, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        },
        {
            "name": "wooden pencil case",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("4de06b64-1660-4775-818b-4935e4cc8cf3/obj.glb"),
                scale=0.55,
                pos=(0.42, -0.38, 0.76 + 0.1031 * 0.55),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=400.0, friction=0.8),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "candle pot",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("d484ed47-44ea-4518-87ae-3397d3eb732f/obj.glb"),
                scale=1.538,
                pos=(0.341, 0.45, 0.76 + 0.0261 * 1.538),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=350.0, friction=0.7),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


def _task_04_trap2():
    """04_trap2 — pinch_card with soap bar trap."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "bank card",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("9b4245ca-361f-4940-a23d-090b1e547a52/obj.glb"),
                scale=0.8,
                pos=(0.46, 0.0, 0.843),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.4),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb"),
                scale=2,
                pos=(0.48, 0.08, 0.85),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "small table",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("17674e05-b713-4b2a-a04c-49aee4d4d401/obj.glb"),
                scale=0.5,
                pos=(0.46, 0, 0.76338),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "target cube",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.86), euler=(0, 0, 0), size=(0.15, 0.15, 0.2), fixed=True, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.7, 0.7, 0.7), roughness=0.4, ior=1.5),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.7, -0.3, 0.96), euler=(0, 0, 0), size=(0.15, 0.15, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.2), roughness=0.4, ior=1.5),
        },
        {
            "name": "soap bar",
            "morph": gs.morphs.Box(
                pos=(0.50, -0.07, 0.85), euler=(0, 0, 0), size=(0.08, 0.05, 0.015), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=0.3),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.92, 0.9), double_sided=True),
        },
    ]


def _task_06_add2():
    """06_add2 — dominos with vase / mug / screw / tennis ball distractors."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "white block 3",
            "morph": gs.morphs.Box(
                pos=(0.74, 0.00, 0.82), euler=(0, 0, 0), size=(0.02, 0.04, 0.12), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6, ior=1.4),
        },
        {
            "name": "white block 2",
            "morph": gs.morphs.Box(
                pos=(0.80, 0.00, 0.82), euler=(0, 0, 0), size=(0.02, 0.04, 0.12), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6, ior=1.4),
        },
        {
            "name": "white block 1",
            "morph": gs.morphs.Box(
                pos=(0.86, 0.00, 0.82), euler=(0, 0, 0), size=(0.02, 0.04, 0.12), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6, ior=1.4),
        },
        {
            "name": "red block",
            "morph": gs.morphs.Box(
                pos=(0.93, 0.00, 0.82), euler=(0, 0, 0), size=(0.02, 0.04, 0.12), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 0.0, 0.0), roughness=0.5, ior=1.4),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.99, 0.00, 0.761), euler=(0, 0, 0), size=(0.08, 0.08, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.1, 0.8, 0.2), double_sided=True),
        },
        {
            "name": "japanese vase",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("b6cde4f1-bd96-4bd4-84d5-228c86d9d8ff/obj.glb"),
                scale=0.753,
                pos=(0.55, -0.15, 0.76 + (0.239 * 0.753) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "coffee mug",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("47876be5-5857-400e-85c3-274f171d6a3d/obj.glb"),
                scale=1.276,
                pos=(0.35, 0.05, 0.76 + (0.0784 * 1.276) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "screw",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("1d373cbe-c73d-4929-bf32-e85a98dc4bca/obj.glb"),
                scale=2.703,
                pos=(0.40, 0.15, 0.76 + (0.0148 * 2.703) / 2),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "tennis ball",
            "morph": gs.morphs.Sphere(
                pos=(0.60, 0.00, 0.76 + 0.0335), euler=(0, 0, 0), radius=0.0335, fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(0.7, 0.9, 0.2), double_sided=True),
        },
    ]


def _task_06_pivot1():
    """06_pivot1 — dominos with thinner blocks + tennis-ball pivot (BK mesh)."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.960, 0.000, 0.761), euler=(0, 0, 0), size=(0.08, 0.08, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.0, 1.0, 0.0), roughness=0.8),
        },
        {
            "name": "white block 3",
            "morph": gs.morphs.Box(
                pos=(0.780, 0.0, 0.800), euler=(0, 0, 0), size=(0.02, 0.012, 0.08), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6),
        },
        {
            "name": "white block 2",
            "morph": gs.morphs.Box(
                pos=(0.815, 0.0, 0.800), euler=(0, 0, 0), size=(0.02, 0.012, 0.08), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6),
        },
        {
            "name": "white block 1",
            "morph": gs.morphs.Box(
                pos=(0.850, 0.0, 0.800), euler=(0, 0, 0), size=(0.02, 0.012, 0.08), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 1.0, 1.0), roughness=0.6),
        },
        {
            "name": "red block",
            "morph": gs.morphs.Box(
                pos=(0.885, 0.0, 0.800), euler=(0, 0, 0), size=(0.02, 0.012, 0.08), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(1.0, 0.0, 0.0), roughness=0.6),
        },
        {
            "name": "tennis ball",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb"),
                scale=0.433,
                pos=(0.600, 0.000, 0.7925),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


def _task_09_pivot1():
    """09_pivot1 — hold_cup with local repaired slope + eraser, target tilted -12°."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "slope",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=str(_DEMO / "repaired" / "slope.glb"),
                scale=(0.5, 1.5, 0.6),
                pos=(0.58, 0.0, 0.76 + 0.0586 * 0.5),
                euler=(0, 0, 90),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "cup",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb"),
                scale=1.1,
                pos=(0.40, -0.10, 0.76 + 0.0459 * 1.1),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.2, coup_friction=0.05),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb"),
                scale=(3, 4, 2),
                pos=(0.62, -0.12, 0.807),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.60, 0.0, 0.76 + 0.0586 * 0.8),
                euler=(-12.0, 0, 0),
                size=(0.13, 0.16, 0.002),
                fixed=True,
                collision=False,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.2),
        },
    ]


def _task_09_trap2():
    """09_trap2 — hold_cup with local repaired slope + coaster + eraser, target tilted -10.3°."""
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "slope",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=str(_DEMO / "repaired" / "slope.glb"),
                scale=(0.5, 1.5, 0.6),
                pos=(0.58, 0.0, 0.76 + 0.0586 * 0.5),
                euler=(0, 0, 90),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.60, 0.0, 0.79988), euler=(-10.3, 0, 0), size=(0.13, 0.16, 0.002), fixed=True, collision=False
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.2, 0.8, 0.3), roughness=0.2),
        },
        {
            "name": "cup",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("3d998505-6bbb-4cc2-8359-c147ac531430/obj.glb"),
                scale=1.1,
                pos=(0.66, -0.12, 0.76 + 0.0459 * 1.1),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.2, coup_friction=0.05),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "coaster",
            "morph": gs.morphs.Cylinder(
                radius=0.04, height=0.02, pos=(0.55, -0.195, 0.76 + 0.004), euler=(0, 0, 0), fixed=False, collision=True
            ),
            "material": gs.materials.Rigid(friction=1.5),
            "surface": gs.surfaces.Default(color=(0.6, 0.4, 0.2), roughness=0.7),
        },
        {
            "name": "eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("333832bb-68f0-4f72-a300-658c4fdccfdf/obj.glb"),
                scale=(3, 4, 2),
                pos=(0.46, -0.18, 0.807),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=1.2),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]


# ---------------------------------------------------------------------------
# Tasks with non-rigid entities in gs-core (MPM/SPH).  The replay driver only
# kinematically drives rigid bodies via NPZ rigid_<name> keys, so soft/fluid
# entities (dough/sand/water/foam balls) are converted to gs.materials.Rigid()
# here.  Pos/scale/euler/coacd/parse_glb_with_zup match gs-core exactly.
# ---------------------------------------------------------------------------


def _task_08():
    """08_round_dough_sheet — flatten a dough ball into a round sheet using a board + cutter.

    Non-rigid in gs-core: 'dough ball' (MPM.ElastoPlastic) — replayed as Rigid Sphere.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "dough ball",
            "morph": gs.morphs.Sphere(
                pos=(0.56, -0.05, 0.816),
                euler=(0, 0, 0),
                radius=0.04,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.95, 0.85, 0.65)),
        },
        {
            "name": "large flat board",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("e8afda3b-6dea-4bfc-859f-88a35bb623a0/obj.glb"),
                scale=0.8,
                pos=(0.56, -0.05, 0.768),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=500.0, friction=0.8),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "round cutter",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("662e6635-9e9f-4aed-991a-760c63592eb3/obj.glb"),
                scale=2.0,
                pos=(0.67, 0.165, 0.827),
                euler=(180, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=1200.0, friction=0.4),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
    ]


def _task_15():
    """15_separate_marbles_and_sand — colander-over-bowl gravity separator.

    Non-rigid in gs-core: 'sand' (MPM.Sand) — replayed as Rigid Cylinder.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    marble_glb = _bk("5cd459e5-fccb-44c5-a368-9249218e10ff/obj.glb")
    marble_positions = [
        (0.668, 0.08, 0.795),
        (0.692, 0.08, 0.795),
        (0.656, 0.062, 0.795),
        (0.68, 0.094, 0.815),
        (0.68, 0.066, 0.815),
    ]
    entities = [
        {
            "name": "colander",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("633e9459-f7ed-4507-ad97-ce2783b06a02/obj.glb"),
                scale=0.864,
                pos=(0.68, -0.12, 0.8038),
                euler=(180, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=300.0, friction=1.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "bowl",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("c652cf0f-d2eb-44bd-9e68-a2ceca698591/obj.glb"),
                scale=1.38,
                pos=(0.50, -0.12, 0.808576),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=300.0, friction=1.0),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "jar",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("f4777d21-c966-40dd-872c-8bf28e00d3ee/obj.glb"),
                scale=0.575,
                pos=(0.68, 0.08, 0.811118),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=300.0, friction=0.9),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
    ]
    for i, mpos in enumerate(marble_positions, start=1):
        entities.append(
            {
                "name": f"marble {i}",
                "morph": gs.morphs.Mesh(
                    align=False,
                    coacd_options=coacd,
                    file=marble_glb,
                    scale=0.12,
                    pos=mpos,
                    euler=(0, 0, 0),
                    fixed=False,
                    collision=True,
                    parse_glb_with_zup=True,
                ),
                "material": gs.materials.Rigid(rho=1200.0, friction=0.6),
                "surface": gs.surfaces.Smooth(double_sided=True),
            }
        )
    entities.append(
        {
            "name": "sand",
            "morph": gs.morphs.Cylinder(
                pos=(0.68, 0.08, 0.82),
                euler=(0, 0, 0),
                radius=0.022,
                height=0.04,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.9, 0.8, 0.3), double_sided=True),
        }
    )
    return entities


def _task_20():
    """20_ball_into_jar — drop a foam ball into a glass jar.

    Non-rigid in gs-core: 'foam ball' (MPM.Elastic) — replayed as Rigid Sphere.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "glass jar",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("f4777d21-c966-40dd-872c-8bf28e00d3ee/obj.glb"),
                scale=0.8,
                pos=(0.64, 0.0, 0.8350),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=250.0, friction=0.5),
            "surface": gs.surfaces.Glass(color=(0.85, 0.9, 0.95), double_sided=True),
        },
        {
            "name": "foam ball",
            "morph": gs.morphs.Sphere(
                pos=(0.46, -0.10, 0.8420),
                radius=0.06,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(color=(1.0, 0.9, 0.1), double_sided=True),
        },
    ]


def _task_21():
    """21_seal_colander — assemble colander+pitcher to retain water.

    Non-rigid in gs-core: 'water' (SPH.Liquid) — replayed as Rigid Cylinder.
    SPH containment blockers (1 cylinder + 12 boxes ringing the pitcher) are
    rendered as visible rigid entities so the replay scene stays geometrically
    identical to gs-core.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    _px, _py = 0.61, 0.0
    entities = [
        {
            "name": "perforated container",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/slotted_spoon.glb"),
                scale=0.8847,
                pos=(0.44, -0.06, 0.779),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(friction=0.6),
            "surface": gs.surfaces.Rough(double_sided=True),
        },
        {
            "name": "pitcher",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb"),
                scale=1.2476,
                pos=(_px, _py, 0.85),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "curved holder",
            "morph": gs.morphs.Sphere(
                radius=0.012,
                pos=(0.50, -0.15, 0.772),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(friction=0.6),
            "surface": gs.surfaces.Smooth(color=(0.85, 0.85, 0.85), double_sided=True),
        },
        {
            "name": "water",
            "morph": gs.morphs.Cylinder(
                radius=0.045,
                height=0.13,
                pos=(_px, _py, 0.86),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.6, 0.7, 1.0), double_sided=True),
        },
        # SPH containment ring — base plate
        {
            "name": "blocker_base",
            "morph": gs.morphs.Cylinder(
                pos=(_px, _py, 0.80),
                radius=0.06,
                height=0.04,
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(),
        },
    ]
    # SPH containment ring — 12 vertical staves
    for rot in range(0, 360, 30):
        sr, cr = float(np.sin(np.deg2rad(rot))), float(np.cos(np.deg2rad(rot)))
        entities.append(
            {
                "name": f"blocker_box_{rot}",
                "morph": gs.morphs.Box(
                    pos=(_px + 0.055 * cr, _py + 0.055 * sr, 0.875),
                    euler=(0.0, 0.0, float(rot)),
                    size=(0.01, 0.04, 0.15),
                    fixed=True,
                    collision=True,
                ),
                "material": gs.materials.Rigid(),
                "surface": gs.surfaces.Default(),
            }
        )
    return entities


def _task_22():
    """22_stabilize_bottle — sand collar stabilizes a leaning tube in a bowl.

    Non-rigid in gs-core: 'dry_sand' (MPM.Sand) — replayed as Rigid Cylinder.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "bowl",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("d8dd7f3f-103d-4daf-b579-188178dc4d9e/obj.glb"),
                scale=1.76,
                pos=(0.52, 0.0, 0.76 + 0.024 * 1.76 + 0.001),
                euler=(0, 0, 0),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "tube",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("e620c6a5-0a69-4a70-a9a7-62c91931715e/obj.glb"),
                scale=2,
                pos=(0.4, 0.25, 0.8),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=50.0),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "sand_container",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb"),
                scale=1.2,
                pos=(0.35, -0.16, 0.8202),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Smooth(double_sided=True),
        },
        {
            "name": "dry_sand",
            "morph": gs.morphs.Cylinder(
                radius=0.022,
                height=0.06 + 0.06,
                pos=(0.35, -0.14, 0.825 + 0.03),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(),
        },
    ]


def _task_25():
    """25_water_into_mug — drop heavy object into pitcher to displace water into mug.

    Non-rigid in gs-core: 'water' (SPH.Liquid) — replayed as Rigid Cylinder.
    SPH containment blockers (1 cylinder + 11 standard boxes + 1 short box at
    rot=90, ringing the pitcher) are rendered as visible rigid entities so the
    replay scene stays geometrically identical to gs-core.

    Mug pos uses the gs-core helper ``mug_pos_relative_to_pitcher`` evaluated
    at pitcher_yaw_deg=-90, body_dx=-0.18, body_dy=0:
    body→world rotation gives (wx, wy)=(0, +0.18), so mug=(0.50, 0.15, 0.8715).
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    _px, _py = 0.50, -0.03
    _pitcher_yaw_deg = -90.0
    entities = [
        {
            "name": "pitcher",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/pitcher.glb"),
                scale=1.525,
                pos=(_px, _py, 0.9515 - 0.08),
                euler=(0, 0, _pitcher_yaw_deg),
                fixed=True,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=800.0, sdf_min_res=128, sdf_max_res=256),
            "surface": gs.surfaces.Smooth(color=(0.8, 0.8, 0.85), double_sided=True),
        },
        {
            "name": "mug",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb"),
                scale=0.85,
                pos=(0.50, 0.15, 0.8715),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=1200.0),
            "surface": gs.surfaces.Smooth(color=(0.9, 0.75, 0.6), double_sided=True),
        },
        {
            "name": "heavy_large_object",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_rw("toolbench/rockman.glb"),
                scale=1.0,
                pos=(0.68, -0.32, 0.793),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=500.0, friction=0.8),
            "surface": gs.surfaces.Rough(color=(0.5, 0.5, 0.5), double_sided=True),
        },
        {
            "name": "water",
            "morph": gs.morphs.Cylinder(
                pos=(_px, _py + 0.01, 0.8715 + 0.2 - 0.08),
                euler=(0, 0, 0),
                radius=0.06,
                height=0.3,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Glass(color=(0.6, 0.85, 1.0), double_sided=True),
        },
        # SPH containment ring — base plate
        {
            "name": "blocker_base",
            "morph": gs.morphs.Cylinder(
                pos=(_px, _py + 0.02, 0.82),
                radius=0.11,
                height=0.05,
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(),
        },
    ]
    # SPH containment ring — 11 standard staves (rot=90 is omitted; replaced by short variant below)
    for rot in [0, 30, 60, 120, 150, 180, 210, 240, 270, 300, 330]:
        sin_rot = float(np.sin(np.deg2rad(rot + 180)))
        cos_rot = float(np.cos(np.deg2rad(rot + 180)))
        entities.append(
            {
                "name": f"blocker_box_{rot}",
                "morph": gs.morphs.Box(
                    pos=(_px + 0.09 * cos_rot, _py + 0.015 + 0.09 * sin_rot, 0.97 - 0.08),
                    euler=(0.0, 0.0, float(rot)),
                    size=(0.01, 0.05, 0.15),
                    fixed=True,
                    collision=True,
                ),
                "material": gs.materials.Rigid(),
                "surface": gs.surfaces.Default(),
            }
        )
    # SPH containment ring — short stave at rot=90 (gap for mug spout / different z)
    sin_rot = float(np.sin(np.deg2rad(90 + 180)))
    cos_rot = float(np.cos(np.deg2rad(90 + 180)))
    entities.append(
        {
            "name": "blocker_box_90",
            "morph": gs.morphs.Box(
                pos=(_px + 0.09 * cos_rot, _py + 0.015 + 0.09 * sin_rot, 0.95 - 0.08),
                euler=(0.0, 0.0, 90.0),
                size=(0.01, 0.05, 0.1),
                fixed=True,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(),
        }
    )
    return entities


def _task_30():
    """30_differentiate_cubes — sort wooden cube to target, keep metal cube in mug.

    Non-rigid in gs-core: 'water' (MPM.Liquid) — replayed as Rigid Cylinder.
    """
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "mug",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("94134b02-c73e-4f2c-ad1d-a00a78160d98/obj.glb"),
                scale=1.0,
                pos=(0.48, 0.0, 0.76 + 0.0477),
                euler=(0, 0, 0),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=1200.0),
            "surface": gs.surfaces.Smooth(color=(0.95, 0.8, 0.6), double_sided=True),
        },
        {
            "name": "wooden cube",
            "morph": gs.morphs.Box(
                pos=(0.48, 0.0, 0.8025),
                euler=(0, 0, 0),
                size=(0.02, 0.02, 0.02),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=300.0),
            "surface": gs.surfaces.Default(color=(1.0, 0.9, 0.1), roughness=0.6),
        },
        {
            "name": "metal cube",
            "morph": gs.morphs.Box(
                pos=(0.48, 0.0, 0.7825),
                euler=(0, 0, 0),
                size=(0.02, 0.02, 0.02),
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(rho=8000.0),
            "surface": gs.surfaces.Metal(color=(0.9, 0.1, 0.1), metal_type="iron"),
        },
        {
            "name": "water pitcher",
            "morph": gs.morphs.Mesh(
                align=False,
                coacd_options=coacd,
                file=_bk("c242567e-052d-4561-b2c0-2fed8a5e576b/obj.glb"),
                scale=1.0,
                pos=(0.68, -0.10, 0.76 + 0.0721),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                parse_glb_with_zup=True,
            ),
            "material": gs.materials.Rigid(rho=1500.0),
            "surface": gs.surfaces.Metal(color=(0.85, 0.85, 0.85), double_sided=True, metal_type="aluminium"),
        },
        {
            "name": "water",
            "morph": gs.morphs.Cylinder(
                pos=(0.68, -0.10, 0.805),
                euler=(0, 0, 0),
                radius=0.028,
                height=0.07,
                fixed=False,
                collision=True,
            ),
            "material": gs.materials.Rigid(),
            "surface": gs.surfaces.Default(color=(0.55, 0.75, 1.0), double_sided=True),
        },
        {
            "name": "target area",
            "morph": gs.morphs.Box(
                pos=(0.70, 0.35, 0.76 + 0.001),
                euler=(0, 0, 0),
                size=(0.12, 0.12, 0.002),
                fixed=True,
                collision=False,
            ),
            "material": gs.materials.Rigid(rho=200.0),
            "surface": gs.surfaces.Default(color=(0.1, 0.8, 0.2), roughness=0.9),
        },
    ]


# Task registry: task_id (str) -> builder function
TASK_REGISTRY: dict[str, Any] = {
    "01": _task_01,
    "03": _task_03,
    "04": _task_04,
    "05": _task_05,
    "06": _task_06,
    "08": _task_08,
    "09": _task_09,
    "10": _task_10,
    "11": _task_11,
    "12": _task_12,
    "13": _task_13,
    "14": _task_14,
    "15": _task_15,
    "16": _task_16,
    "17": _task_17,
    "18": _task_18,
    "19": _task_19,
    "20": _task_20,
    "21": _task_21,
    "22": _task_22,
    "23": _task_23,
    "24": _task_24,
    "25": _task_25,
    "26": _task_26,
    "27": _task_27,
    "28": _task_28,
    "29": _task_29,
    "30": _task_30,
    # _v0/_v1 share the base task scene; trajectory file differs per version
    "02_v0": _task_02,
    "02_v1": _task_02,
    "07_v0": _task_07,
    "07_v1": _task_07,
    # _add*/_trap*/_pivot* are distinct scenes (mirroring gs-core variant scripts)
    "01_add1": _task_01_add1,
    "01_trap2": _task_01_trap2,
    "04_add1": _task_04_add1,
    "04_trap2": _task_04_trap2,
    "06_add2": _task_06_add2,
    "06_pivot1": _task_06_pivot1,
    "09_pivot1": _task_09_pivot1,
    "09_trap2": _task_09_trap2,
}

TASK_NAMES: dict[str, str] = {
    "01": "ruler_align_cubes",
    "02": "retrieve_cube",
    "03": "gap_retrieve",
    "04": "pinch_card",
    "05": "box_into_basket",
    "06": "dominos",
    "07": "assemble_pages_with_bar",
    "08": "round_dough_sheet",
    "09": "hold_cup",
    "10": "collect_screws",
    "11": "place_tall_box",
    "12": "pour_ball_through_funnel",
    "13": "cover_with_lid",
    "14": "stack_cubes",
    "15": "separate_marbles_and_sand",
    "16": "stand_bulb",
    "17": "ball_onto_tower",
    "18": "cylinder_through_hole",
    "19": "stack_bowls",
    "20": "ball_into_jar",
    "21": "seal_colander",
    "22": "stabilize_bottle",
    "23": "place_book_on_shelf",
    "24": "raise_platform",
    "25": "water_into_mug",
    "26": "align_chopsticks_with_ruler",
    "27": "retrieve_roll",
    "28": "move_cube",
    "29": "balance_board",
    "30": "differentiate_cubes",
    "02_v0": "retrieve_cube_v0",
    "02_v1": "retrieve_cube_v1",
    "07_v0": "assemble_pages_with_bar_v0",
    "07_v1": "assemble_pages_with_bar_v1",
    "01_add1": "ruler_align_cubes_add1",
    "01_trap2": "ruler_align_cubes_trap2",
    "04_add1": "pinch_card_add1",
    "04_trap2": "pinch_card_trap2",
    "06_add2": "dominos_add2",
    "06_pivot1": "dominos_pivot1",
    "09_pivot1": "hold_cup_pivot1",
    "09_trap2": "hold_cup_trap2",
}

# HuggingFace trajectory download
_HF_TRAJ_REPO = "Genesis-Intelligence/internal_assets"
_HF_TRAJ_PREFIX = "ipc_trajectories/robowits"


def _resolve_traj(task_id: str, traj_arg: str | None) -> str:
    """Return path to trajectory NPZ, downloading from HF if needed."""
    if traj_arg:
        return traj_arg
    # Try local trajectories/ dir first
    local = _TRAJ_DIR / f"{task_id}.npz"
    if local.exists():
        return str(local)
    # Try HuggingFace
    try:
        return _hf_download(_HF_TRAJ_REPO, f"{_HF_TRAJ_PREFIX}/{task_id}.npz")
    except Exception:
        pass
    raise FileNotFoundError(
        f"No trajectory found for task {task_id}. "
        f"Place {task_id}.npz in {_TRAJ_DIR}/ or run convert_mcap_to_npz.py first."
    )


# ---------------------------------------------------------------------------
# Replay class
# ---------------------------------------------------------------------------


class RobowitsReplay(TrajectoryReplay):
    name = "robowits"
    cam_pos = (1.5122, -0.767, 1.8931)
    cam_lookat = (0.838, -0.3497, 1.2837)
    cam_fov = 40

    # ── Shared light rig ──────────────────────────────────────────────────────
    # Edit pos / color / intensity here; both renderers pick them up automatically.
    # Values are in Luisa units; Nyx multiplies intensity by
    # `_LUISA_TO_NYX_INTENSITY_SCALE` (defined module-top), nothing else.
    _LIGHTS: list[_LightDef] = [
        # Key light: above-left, warm, casting shadows across the scene
        {"pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 50.0},
        # Fill light: right side, cooler, large and soft
        {"pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 1.0},
        # Rim light: behind the scene, cool, hard — separates hands/shirt from dark background
        # {"pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 150.0},
        {"pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 150.0},
    ]

    def _light_intensity_factor(self) -> float:
        # Sphere lights are tuned to balance against the per-task env map. When
        # the env map is off (--no-use_env_map), the env contribution is zero,
        # so skip the dampening factor and run lights at their tuned full level.
        return _ENV_LIGHT_EFFECT if self.args.use_env_map else 1.0

    def _resolve_envmap(self) -> tuple[str, float, float]:
        """Return (local_path, yaw_deg, registry_multiplier) for the active envmap.

        In classic mode, all tasks use polyhaven small_empty_room_1; otherwise
        the per-task EXR from _TASK_ENVMAP is used.
        """
        if self.args.classic:
            yaw, mult = _CLASSIC_ENVMAP_REGISTRY
            path = _classic_envmap_path()
        else:
            env_filename = _TASK_ENVMAP[self.args.task]
            yaw, mult = _ENVMAP_REGISTRY[env_filename]
            path = _envmap(env_filename)
        if self.args.env_yaw is not None:
            yaw = float(self.args.env_yaw)
        return (path, yaw, mult)

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        env_path, env_yaw, env_registry_mult = self._resolve_envmap()
        env_multiplier = env_registry_mult * _OVERALL_INTENSITY if self.args.use_env_map else 0.0
        env_multiplier *= float(self.args.luisa_env_scale)
        # ImageTexture.image_color is clamped to [0, 1]; >1 multipliers would need
        # an exposure EV bump instead, but our 16-task table is all <= 1.
        assert env_multiplier <= 1.0, f"task {self.args.task}: envmap multiplier {env_multiplier} > 1 not supported"
        light_factor = self._light_intensity_factor()
        # Luisa rotates the env map opposite to Nyx around the up axis; negate so
        # the registry yaw means the same visual orientation in both backends.
        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_radius=100.0,
            env_euler=(0, 0, -env_yaw),
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path=env_path,
                    image_color=env_multiplier,
                    encoding="linear",
                ),
            ),
            lights=[
                SphereLight(
                    pos=l["pos"],
                    radius=l["radius"],
                    color=l["color"],
                    intensity=l["intensity"] * light_factor,
                )
                for l in self._LIGHTS
            ],
        )

    def nyx_lights(self):
        light_factor = self._light_intensity_factor()
        return [
            {
                "type": "point",
                "pos": l["pos"],
                "radius": float(l["radius"]),
                "color": l["color"],
                "intensity": float(l["intensity"]) * light_factor * _LUISA_TO_NYX_INTENSITY_SCALE,
            }
            for l in self._LIGHTS
        ]

    def nyx_env_map(self):
        # Match Luisa: same EXR, registry yaw, and registry-mult * dim so the
        # two renderers see identical sky. With --no-use_env_map, multiplier is
        # zero so the env contribution is pure black (texture is still loaded
        # but doesn't reach the scene).
        env_path, env_yaw, env_registry_mult = self._resolve_envmap()
        mult = env_registry_mult * _OVERALL_INTENSITY if self.args.use_env_map else 0.0
        return (env_path, env_yaw, mult)

    def nyx_light_field(self):
        # No 3DGS splat for Robowits — the per-task envmap is the visible
        # background, matching Luisa.
        return None

    def add_args(self, parser):
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            choices=sorted(TASK_REGISTRY.keys()),
            help="Task ID (e.g. 01, 02, 06, 23, 26, 29)",
        )
        parser.add_argument(
            "--classic",
            action="store_true",
            help="Classic mode: use the archived MarvinPika URDF (no new camera "
            "base), the polyhaven small_empty_room_1 envmap for every task, and "
            "a Y=0 side-on camera (looking from +X,+Z toward smaller X,Z). "
            "Output filename gains a _classic suffix.",
        )
        parser.add_argument(
            "--traj",
            type=str,
            default=None,
            help="Path to trajectory NPZ (default: auto-resolve from trajectories/)",
        )
        parser.add_argument(
            "--no-process",
            action="store_true",
            help="Disable CoACD convexify AND mesh decimation on all mesh entities. "
            "Replay is kinematic (poses are scripted), so convex collision shapes "
            "are unused — skipping mesh processing avoids the slow first-run "
            "preprocessing entirely.",
        )
        parser.add_argument(
            "--collision",
            action="store_true",
            help="Render everything (table, task entities, robot) in vis_mode='collision' "
            "to visualize the collision geometry instead of the visual meshes.",
        )
        parser.add_argument(
            "--use_env_map",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the per-task HDR env map (default: on). Pass --no-use_env_map "
            "for pure black env; sphere lights then run at full tuned intensity "
            "(no _ENV_LIGHT_EFFECT dampening) and the output filename gains a "
            "_no_envmap suffix to avoid clobbering env-map renders.",
        )
        parser.add_argument(
            "--env_yaw",
            type=float,
            default=None,
            help="Override the env-map yaw rotation (degrees about Z). When unset, "
            "uses the registry default (per-task yaw, or _CLASSIC_ENVMAP_REGISTRY "
            "for --classic).",
        )
        parser.add_argument(
            "--luisa-env-scale",
            type=float,
            default=0.6,
            help="Extra multiplier on the Luisa env-map brightness (Nyx unaffected). "
            "Use < 1 to darken the sky toward Nyx's look without touching the registry.",
        )
        parser.add_argument(
            "--free-sim",
            type=int,
            default=0,
            help="Run N free-physics steps with the robot pinned to frame-0 qpos, "
            "then render one PNG of the settled state. Switches the deformable/granular/"
            "fluid stand-in entities (08 dough, 15 sand, 20 foam ball, 21 water, 22 dry_sand, "
            "25 water, 30 water) from gs.materials.Rigid() to the matching MPM/SPH material "
            "from gs-core, enables physics + collisions, and uses the gs-core robowits "
            "scene options. 0 (default) keeps the kinematic-replay path.",
        )
        parser.add_argument(
            "--free-sim-output",
            type=str,
            default=None,
            help="Output PNG path for --free-sim. Defaults to "
            "data/ipc_demo/ipc_robowits/_settled_frames_classic_luisa/task<ID>.png",
        )

    def load_trajectory(self):
        # In classic mode, override the camera so both points lie on the Y=0
        # plane: camera at (+X, 0, +Z) looking back toward (X-X0, 0, Z-Z0).
        # XZ projection of the default cam_pos/lookat is preserved; we only
        # zero out the Y components so the view is purely side-on.
        if self.args.classic:
            cp = type(self).cam_pos
            cl = type(self).cam_lookat
            self.cam_pos = (cp[0], 0.0, cp[2])
            self.cam_lookat = (cl[0], 0.0, cl[2])

        task_id = self.args.task
        # Use the task id as the trajectory tag so render outputs are named
        # ipc_robowits_<task>_<renderer>_<datetime>.mp4 instead of "default".
        # With --no-use_env_map we add a _no_envmap suffix so those renders
        # land in a separate file and never clobber the env-map renders.
        # --classic adds a _classic suffix for the same reason.
        suffix = "" if self.args.use_env_map else "_no_envmap"
        if self.args.classic:
            suffix = f"{suffix}_classic"
        self.args.trajectory = f"task{task_id}{suffix}"
        traj_path = _resolve_traj(task_id, self.args.traj)
        traj = np.load(traj_path)

        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot qpos (18 DOF MARVIN_PIKA bimanual)
        if "robot_qpos" in traj:
            self._joint_qpos = traj["robot_qpos"].astype(np.float32)
            print(f"  Robot: {self._joint_qpos.shape[1]} DOF")
        else:
            self._joint_qpos = None
            print("  Robot: NO robot data (will show initial pose)")

        # Rigid object data: keys are "rigid_{entity_name}" with original names
        self._rigid_data = {}
        for key in traj.files:
            if key.startswith("rigid_"):
                entity_name = key[6:]
                self._rigid_data[entity_name] = traj[key].astype(np.float32)
                print(f"  {entity_name}: {traj[key].shape[0]} frames")

        # Compute FPS from timestamps
        if n_frames > 1:
            dt = self.sim_time[1] - self.sim_time[0]
            self.fps = min(int(1.0 / dt), 60) if dt > 0 else 60

        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        task_id = self.args.task
        vis_mode = "collision" if self.args.collision else "visual"
        # Robot URDF needs collision=True for collision geometry to be visible
        robot_collision = self.args.collision

        # Table (shared across all Robowits tasks — matches registry)
        scene.add_entity(
            gs.morphs.Mesh(
                align=False,
                file=TABLE_GLB,
                pos=(0.597, 0.0, 0.0),
                euler=(0, 0, 0),
                scale=(1.14, 1.0, 1.4377),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0),
            vis_mode=vis_mode,
        )

        # Task-specific entities
        entity_defs = TASK_REGISTRY[task_id]()
        for edef in entity_defs:
            morph = edef["morph"]
            if self.args.no_process and isinstance(morph, gs.morphs.Mesh):
                morph.convexify = False
                morph.decimate = False
            entity = scene.add_entity(
                morph=morph,
                material=edef.get("material", gs.materials.Rigid()),
                surface=edef.get("surface", gs.surfaces.Default()),
                vis_mode=vis_mode,
            )
            self._rigid_entities[edef["name"]] = entity

        # Robot (MARVIN_PIKA, 18 DOF, fixed base). --classic swaps to the
        # archived URDF that predates the new camera-base assembly.
        urdf_file = CLASSIC_MARVIN_PIKA_URDF if self.args.classic else MARVIN_PIKA_URDF
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_file,
                fixed=True,
                collision=robot_collision,
                pos=(0, 0, 1.08),
            ),
            vis_mode=vis_mode,
            surface={
                "paint_white_glossy": gs.surfaces.BSDF(color=(0.74, 0.74, 0.74), roughness=0.25, metallic=0.25),
                "plastic_black_rough": gs.surfaces.BSDF(
                    color=(0.02, 0.02, 0.03), roughness=0.35, metallic=0.0, ior=1.45
                ),
            }
            if self.args.classic
            else None,
        )

    def apply_frame(self, scene, frame_idx: int):
        import genesis.utils.geom as gu

        # Robot qpos
        if self._joint_qpos is not None and frame_idx < len(self._joint_qpos):
            self._robot.set_qpos(self._joint_qpos[frame_idx])

        # Rigid objects — NPZ stores pos(3) + euler(3, RPY radians)
        for name, entity in self._rigid_entities.items():
            if name in self._rigid_data and frame_idx < len(self._rigid_data[name]):
                pose = self._rigid_data[name][frame_idx]
                entity.set_pos(pose[:3])
                euler = torch.tensor(pose[3:], dtype=torch.float32)
                entity.set_quat(gu.xyz_to_quat(euler, rpy=True))


if __name__ == "__main__":
    RobowitsReplay().run()
