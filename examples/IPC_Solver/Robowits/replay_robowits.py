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
                pos=(0.7, -0.3, 0.96), euler=(0, 0, 0), size=(0.15, 0.15, 0.002), fixed=True, collision=False
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


def _task_11():
    import genesis as gs

    coacd = gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=80, max_convex_hull=20, decimate=True)
    return [
        {
            "name": "tall cardboard box",
            "morph": gs.morphs.Mesh(
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


# Task registry: task_id (str) -> builder function
TASK_REGISTRY: dict[str, Any] = {
    "02": _task_02,
    "03": _task_03,
    "04": _task_04,
    "06": _task_06,
    "09": _task_09,
    "11": _task_11,
    "13": _task_13,
    "14": _task_14,
    "16": _task_16,
    "17": _task_17,
    "18": _task_18,
    "19": _task_19,
    "24": _task_24,
    "27": _task_27,
    "28": _task_28,
    "29": _task_29,
}

TASK_NAMES: dict[str, str] = {
    "02": "retrieve_cube",
    "03": "gap_retrieve",
    "04": "pinch_card",
    "06": "dominos",
    "09": "hold_cup",
    "11": "place_tall_box",
    "13": "cover_with_lid",
    "14": "stack_cubes",
    "16": "stand_bulb",
    "17": "ball_onto_tower",
    "18": "cylinder_through_hole",
    "19": "stack_bowls",
    "24": "raise_platform",
    "27": "retrieve_roll",
    "28": "move_cube",
    "29": "balance_board",
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

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        env_filename = _TASK_ENVMAP[self.args.task]
        env_yaw, env_registry_mult = _ENVMAP_REGISTRY[env_filename]
        env_multiplier = env_registry_mult * _OVERALL_INTENSITY if self.args.use_env_map else 0.0
        env_path = _envmap(env_filename)
        # ImageTexture.image_color is clamped to [0, 1]; >1 multipliers would need
        # an exposure EV bump instead, but our 16-task table is all <= 1.
        assert env_multiplier <= 1.0, f"task {self.args.task}: envmap multiplier {env_multiplier} > 1 not supported"
        light_factor = self._light_intensity_factor()
        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_radius=100.0,
            env_euler=(0, 0, env_yaw),
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
        env_filename = _TASK_ENVMAP[self.args.task]
        env_yaw, env_registry_mult = _ENVMAP_REGISTRY[env_filename]
        mult = env_registry_mult * _OVERALL_INTENSITY if self.args.use_env_map else 0.0
        return (_envmap(env_filename), env_yaw, mult)

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
            help="Task ID (e.g. 02, 06, 29)",
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

    def load_trajectory(self):
        task_id = self.args.task
        # Use the task id as the trajectory tag so render outputs are named
        # ipc_robowits_<task>_<renderer>_<datetime>.mp4 instead of "default".
        # With --no-use_env_map we add a _no_envmap suffix so those renders
        # land in a separate file and never clobber the env-map renders.
        suffix = "" if self.args.use_env_map else "_no_envmap"
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
        # Per-entity quat correction, populated lazily on first apply_frame call.
        # Inner value is None when no correction is needed (recorded init differs
        # from natural by yaw only — random rotation, not a morph mismatch).
        self._quat_correction: "dict[str, torch.Tensor | None] | None" = None
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

        # Robot (MARVIN_PIKA, 18 DOF, fixed base)
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=MARVIN_PIKA_URDF,
                fixed=True,
                collision=robot_collision,
                pos=(0, 0, 1.08),
            ),
            vis_mode=vis_mode,
        )

    def apply_frame(self, scene, frame_idx: int):
        import genesis.utils.geom as gu

        # Robot qpos
        if self._joint_qpos is not None and frame_idx < len(self._joint_qpos):
            self._robot.set_qpos(self._joint_qpos[frame_idx])

        # Some MCAPs were recorded against a different morph euler than the
        # current source env (e.g. task 02: container morph changed from
        # euler=(90,0,90) lying-down → (0,0,90) upright). For those entities
        # we rebase the recorded *world* rotation so the new entity undergoes
        # the same world-frame rotation as the old:
        #   R_world(t) = Q_old(t) * inv(Q_old_init)
        #   Q_new(t)   = R_world(t) * Q_new_init = Q_old(t) * C
        # where C = inv(Q_old_init) * Q_new_init (right-multiplication).
        #
        # Only entities whose recorded_init differs from source_natural by
        # a non-yaw rotation (i.e. an actual morph orientation mismatch) get
        # corrected. Pure-yaw differences come from the source env's random
        # rotation augmentation and should be left alone — applying the
        # correction would inject any small recording noise into every frame,
        # producing a visibly tilted resting pose later in the trajectory.
        if self._quat_correction is None:
            self._quat_correction = {}
            # sin(half-angle) threshold around non-Z axes. ~0.1 ≈ 11° non-yaw.
            non_yaw_thresh = 0.1
            for name, entity in self._rigid_entities.items():
                if name not in self._rigid_data:
                    continue
                source_natural = entity.get_quat()
                if source_natural.dim() == 2:
                    source_natural = source_natural[0]
                source_natural = source_natural.detach().to("cpu", dtype=torch.float32)
                pose0 = self._rigid_data[name][0]
                euler0 = torch.tensor(pose0[3:], dtype=torch.float32)
                recorded_init = gu.xyz_to_quat(euler0, rpy=True).detach().cpu()
                w, x, y, z = recorded_init.tolist()
                inv_recorded = torch.tensor([w, -x, -y, -z], dtype=torch.float32)
                # transform_quat_by_quat(v, u) computes u * v, so passing
                # (source_natural, inv_recorded) yields inv_recorded * source_natural.
                C = gu.transform_quat_by_quat(source_natural, inv_recorded)
                # Decide whether to apply correction: check non-yaw axis component.
                cw, cx, cy, cz = C.tolist()
                non_yaw = (cx * cx + cy * cy) ** 0.5
                if non_yaw > non_yaw_thresh:
                    self._quat_correction[name] = C
                else:
                    self._quat_correction[name] = None  # no correction needed

        # Rigid objects — NPZ stores pos(3) + euler(3, RPY radians)
        for name, entity in self._rigid_entities.items():
            if name in self._rigid_data and frame_idx < len(self._rigid_data[name]):
                pose = self._rigid_data[name][frame_idx]
                entity.set_pos(pose[:3])
                euler = torch.tensor(pose[3:], dtype=torch.float32)
                recorded_quat = gu.xyz_to_quat(euler, rpy=True)
                correction = self._quat_correction[name]
                if correction is not None:
                    # Right-multiply: Q_new(t) = Q_old(t) * correction.
                    # transform_quat_by_quat(v, u) = u * v, so pass (correction, recorded_quat).
                    quat = gu.transform_quat_by_quat(correction, recorded_quat)
                else:
                    quat = recorded_quat
                entity.set_quat(quat)


if __name__ == "__main__":
    RobowitsReplay().run()
