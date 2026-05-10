"""Replay digital_twin teleop trajectories (Marvin GSS 54 DOF or Sharpa 58 DOF).

Replays NPZ trajectories produced by
``gs-core/scripts/mcap_to_npz.py`` against the corresponding
gs-core ``*_TWIN`` env: SORT_AND_PLACE_TWIN, POUR_COFFEE_BEANS_TWIN, and
PLACE_IN_DRAWER_TWIN. Default NPZs ship in ``trajectories/``:

    * ``sort_gss.npz``    — task=sort
    * ``pour_gss.npz``    — task=pour
    * ``drawer_gss.npz``  — task=drawer (drawer + primary have full qpos
                                         so the prismatic slide replays
                                         correctly)

The earlier Sharpa-recorded NPZs (``sort_objects_and_place.npz``,
``pour_coffee_beans.npz``, ``place_in_drawer.npz``) are still available via
``--traj <path>``. The robot URDF is auto-selected by the ``robot_qpos``
dimension in the NPZ (54 → GSS, 58 → Sharpa).

Usage
-----
    python replay_digital_twin.py --task sort
    python replay_digital_twin.py --task pour
    python replay_digital_twin.py --task drawer
    python replay_digital_twin.py --task drawer --render
    python replay_digital_twin.py --task pour --render --nyx

Asset resolution
----------------
Most digital_twin meshes/URDFs come from the HuggingFace dataset
``Genesis-Intelligence/Digital_twin_asset`` (commit pinned via
``_DIGITAL_TWIN_COMMIT`` to mirror gs-core's ``data_assets.py``). The drawer
URDF is loaded from gs-core's local override at
``data/local_assets/digital_twin/drawer/acrylic_drawer.urdf`` so the inertia
fix lands.

Rendering
---------
``--render`` (Luisa) and ``--nyx`` are wired up with a shared 3-light sphere
rig and a per-task HDR env map. The three tasks all live on the same
digital-twin table layout so one rig works; tweak ``_LIGHTS`` /
``_TASK_ENVMAP`` / ``_ENVMAP_REGISTRY`` below if a task wants a different look.
Output filenames are timestamp-free
(``data/ipc_demo/ipc_digital_twin/ipc_digital_twin_<task>_<renderer>.mp4``)
so re-renders overwrite cleanly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]  # Genesis-IPC root
_DEMO = _REPO / "DemoAssets"
_TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"
_GS_CORE = Path("/home/zhehuan/Desktop/hz/gs-core")

sys.path.insert(0, str(_REPO / "examples" / "IPC_Solver"))
from _replay_common import EgoCamera, TrajectoryReplay, marvin_urdf  # noqa: E402

# Marvin ego RealSense D455 calibration in robot base frame. Mirrors gs-core's
# `EgoRealsenseD455Calib` (schemas/gs_schemas/calib/registry.py) and the
# Genesis-side viewer values used by `get_ego_camera_params`
# (env/teleop/core/.../robot_visualizer.py:476). The viewer values
# (d=0.06, angle=41.5°) push the optical centre past the camera_rs housing
# mesh so the rendered ego view isn't occluded — same renderer caveat applies
# here, so we use the viewer-tuned values rather than the raw physical ones
# (d=0.01, angle=43.5°).
_EGO_CAM_BACK_CENTER = (0.083377, -0.0115, 0.325782)
_EGO_CAM_OPTICAL_DEPTH = 0.06
_EGO_CAM_ANGLE_RAD = 41.5 * np.pi / 180.0
_EGO_CAM_VFOV_DEG = 58.0  # f=fy=424, image_height=480 → vfov ≈ 58°
_MARVIN_BASE_WORLD = (0.0, 0.0, 1.08)  # see DigitalTwinReplay.build_scene


def _marvin_ego_camera() -> EgoCamera:
    """Build an EgoCamera using the gs-core marvin ego calibration."""
    sa = float(np.sin(_EGO_CAM_ANGLE_RAD))
    ca = float(np.cos(_EGO_CAM_ANGLE_RAD))
    fwd = np.array([sa, 0.0, -ca])
    back = np.array(_EGO_CAM_BACK_CENTER)
    pos_in_base = back + _EGO_CAM_OPTICAL_DEPTH * fwd
    lookat_in_base = back + 1.0 * fwd
    return EgoCamera(
        base_pos=_MARVIN_BASE_WORLD,
        offset=tuple(pos_in_base.tolist()),
        lookat_offset=tuple((lookat_in_base - pos_in_base).tolist()),
    )


# ---------------------------------------------------------------------------
# Shared assets
# ---------------------------------------------------------------------------

TABLE_GLB = str(_DEMO / "coat_hanger" / "work_table.glb")
MARVIN_SHARPA_URDF = marvin_urdf("marvin_sharpa")
MARVIN_GSS_URDF = marvin_urdf("marvin_gss")

# Robot URDF dispatch — picked from robot_qpos column count.
_ROBOT_URDF_BY_DOF = {
    54: ("gss", MARVIN_GSS_URDF),
    58: ("sharpa", MARVIN_SHARPA_URDF),
}

# Local override for drawer URDF (matches _DIGITAL_TWIN_LOCAL_OVERRIDES in
# gs-core/env/schemas/.../objects/registry.py — that one carries the inertial
# fix where the outer shell mass is 5.0 kg).
DRAWER_URDF = str(_DEMO / "repaired" / "drawer" / "acrylic_drawer.urdf")

# Mirror gs-core/env/schemas/.../data_assets.py
_DIGITAL_TWIN_REPO = "Genesis-Intelligence/Digital_twin_asset"
_DIGITAL_TWIN_COMMIT = "a9eb6ecbed37af1bf9c0553d8b013d9302e10508"
_DEMO_AUG_REPO = "Genesis-Intelligence/DemoAug_assets"
_DEMO_AUG_COMMIT = "4480ca6f204ae41612b6f0b01f7bcf6910161003"

# Local-only assets that don't live on HuggingFace (mirrors
# _DIGITAL_TWIN_LOCAL_OVERRIDES in gs-core/env/schemas/.../objects/registry.py).
_WOODEN_PLATEAU_GLB = str(_GS_CORE / "data/local_assets/digital_twin/wooden_plateau/wooden_plateau.glb")
_BIN_LID_GLB = str(_GS_CORE / "data/local_assets/digital_twin/bin_lid/bin_lid.glb")
# TB_V2 uses a modified jar URDF (no interior filler box that would eject beans).
# The URDF is local; the GLBs it references resolve via symlinks into the HF cache.
_TB_V2_COFFEE_JAR_URDF = str(
    _GS_CORE / "data/local_assets/digital_twin/table_bussing/coffee_jar/03_output/coffee_jar_modified.urdf"
)
# TB_V2 puddle mesh — gs-core reconstructs this on the fly via splashsurf and
# never persists it. Pre-baked once by ``scripts/generate_liquid_puddle_mesh.py``
# (mirrors LiquidManager particle box + 50% deactivation, splashsurf-1.5).
# Vertices are in world coords (centered on liquid_pos = (0.40, 0.21, 0.745));
# the recorded ``rigid_liquid_puddle`` pose is just a tiny z bias to anchor the
# AABB bottom on the table top, so leaving the mesh in world coords matches
# gs-core's "link at origin, geometry carries the offset" convention.
_TB_V2_PUDDLE_OBJ = str(_GS_CORE / "data/local_assets/digital_twin/table_bussing/liquid_puddle/puddle.obj")

# Per-task entity names that are fixed=True in gs-core (env-level override). For
# these we pop the rigid_<name> array from the NPZ, bake frame-0 pos into the
# morph, and skip set_pos/set_quat in apply_frame — matching gs-core, where the
# entity is welded to the world by the URDF's fixed root joint and only moved
# once (during _snap_objects_onto_table_at_build) before recording starts.
_TASK_FIXED_ENTITIES: dict[str, set[str]] = {
    "clear_table_v2": {
        "recycle_bin_green",
        "recycle_bin_grey",
        "white_dish_tray_body",
        "white_dish_tray_side",
    },
}


_HF_CACHE: dict[str, str] = {}


def _dt(rel_path: str) -> str:
    """Download a file from Digital_twin_asset at the pinned commit."""
    if rel_path not in _HF_CACHE:
        from huggingface_hub import hf_hub_download

        _HF_CACHE[rel_path] = hf_hub_download(
            repo_id=_DIGITAL_TWIN_REPO,
            repo_type="dataset",
            filename=rel_path,
            revision=_DIGITAL_TWIN_COMMIT,
        )
    return _HF_CACHE[rel_path]


def _da(rel_path: str) -> str:
    """Download a file from DemoAug_assets at the pinned commit."""
    key = f"demo_aug:{rel_path}"
    if key not in _HF_CACHE:
        from huggingface_hub import hf_hub_download

        _HF_CACHE[key] = hf_hub_download(
            repo_id=_DEMO_AUG_REPO,
            repo_type="dataset",
            filename=rel_path,
            revision=_DEMO_AUG_COMMIT,
        )
    return _HF_CACHE[key]


# EXRs not in the Digital_twin_asset HF dataset — fetched lazily from
# polyhaven (CC0) into the shared DemoAssets/_polyhaven_cache (same dir
# Robowits's --classic mode uses) so duplicate downloads are avoided.
_POLYHAVEN_URL_OVERRIDES: dict[str, str] = {
    "comfy_cafe_4k.exr": "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/comfy_cafe_4k.exr",
}


def _envmap(filename: str) -> str:
    """Resolve an envmap filename to a local path.

    First checks the polyhaven URL-override map (cached under
    DemoAssets/_polyhaven_cache/); falls back to the HF Digital_twin_asset
    repo for the curated 4K set.
    """
    url = _POLYHAVEN_URL_OVERRIDES.get(filename)
    if url is not None:
        cache_dir = _DEMO / "_polyhaven_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / filename
        if not target.exists():
            import urllib.request

            print(f"[envmap] downloading {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        return str(target)
    return _dt(f"envmap/4K/{filename}")


# Shared light rig for all three tasks — they live on the same digital-twin
# table layout so one rig works. Same shape as Robowits' rig but tuned for the
# brighter studio look the digital-twin captures were filmed under.
_LIGHTS = [
    # Key: above-left, warm, the dominant shadow source
    {"pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 50.0},
    # Fill: right-side soft, slightly cool
    {"pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.55, 0.6, 0.65), "intensity": 1.0},
    # Rim: behind to separate the robot/objects from the background
    {"pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 150.0},
]

# Per-EXR (rotation_deg, multiplier) — same shape as Robowits's registry. Keep
# this orthogonal to the task → EXR mapping so the values are easy to retune.
_ENVMAP_REGISTRY: dict[str, tuple[float, float]] = {
    "brown_photostudio_02_4k.exr": (0.0, 1.0),
    "bright_labratory.exr": (0.0, 1.0),
    "unfinished_office_night_4k.exr": (0.0, 1.0),
    "mirrored_hall_4k.exr": (0.0, 1.0),
    "burnt_warehouse_4k.exr": (0.0, 1.0),
    "fireplace_4k.exr": (0.0, 1.0),
    "neon_photostudio_4k.exr": (0.0, 1.0),
    "abandoned_greenhouse_4k.exr": (0.0, 1.0),
    "comfy_cafe_4k.exr": (0.0, 1.0),
}

# Task → EXR. Each of the 8 digital-twin tasks gets a distinct env map, and
# all 8 are disjoint from Robowits's _TASK_ENVMAP so the union of 16+8=24
# tasks across the two demos all have unique skies.
_TASK_ENVMAP: dict[str, str] = {
    "sort": "brown_photostudio_02_4k.exr",
    "pour": "bright_labratory.exr",
    "drawer": "unfinished_office_night_4k.exr",
    "pick_cup_tray": "mirrored_hall_4k.exr",
    "pick_corn_bin": "burnt_warehouse_4k.exr",
    "pick_potatoes": "fireplace_4k.exr",
    "ipc_pick_pen": "neon_photostudio_4k.exr",
    "pick_carrot_basket": "abandoned_greenhouse_4k.exr",
    "clear_table_v2": "comfy_cafe_4k.exr",
}

# Global dim on the registry envmap multiplier so the HDR sky doesn't
# overpower our key/fill/rim sphere lights under exposure=0.5/aces.
_OVERALL_INTENSITY = 0.5
# Dim sphere lights when env map is on, so HDR sky doesn't get drowned out.
# With --no-use_env_map the lights run at full tuned intensity.
_ENV_LIGHT_EFFECT = 0.1
# Luisa SphereLight intensity ↔ Nyx PointLight intensity scaling (same scale
# as Robowits — keeps the two renderers visually comparable).
_LUISA_TO_NYX_INTENSITY_SCALE = 0.3


# ---------------------------------------------------------------------------
# Per-task scene definitions
# ---------------------------------------------------------------------------
# Each builder returns a list of entity dicts.  Keys:
#   name       : entity name (must match NPZ key, i.e. rigid_{name})
#   morph      : callable returning gs.morphs.* (deferred so `gs` is imported lazily)
#   surface    : optional gs.surfaces.* for rendering
#
# Scale / euler values mirror MORPH_DIGITAL_TWIN_ARGS_MAP in gs-core
# objects/registry.py at the time of recording.


def _task_sort() -> list[dict]:
    import genesis as gs

    return [
        {
            "name": "fruits_primary_0",  # orange
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("orange/orange.glb"),
                scale=0.7,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "fruits_primary_1",  # grapes
            "morph": gs.morphs.Mesh(
                align=False,
                file=str(_DEMO / "repaired" / "grapes.glb"),
                scale=0.15,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "fruits_destination",  # grey_plate
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/plate/grey_plate.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "vegetables_primary_0",  # red_chili
            "morph": gs.morphs.Mesh(
                align=False,
                file=str(_DEMO / "repaired" / "red_chili.glb"),
                scale=0.16,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "vegetables_primary_1",  # green_pepper
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("green_pepper/green_pepper.glb"),
                scale=0.003,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "vegetables_destination",  # sage_plate
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/plate/sage_plate.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]


def _task_pour() -> list[dict]:
    import genesis as gs

    entries: list[dict] = [
        {
            "name": "src_container",  # yellow_plastic_bowl
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/yellow_plastic_bowl/yellow_plastic_bowl.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "dst_container",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/yellow_plastic_bowl/yellow_plastic_bowl.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]
    # 8 coffee beans (2x2x2 grid) — coffee_bean_simple.urdf, scale 0.1
    bean_urdf = _dt("coffee_bean/coffee_bean_simple.urdf")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                entries.append(
                    {
                        "name": f"content_{i}_{j}_{k}",
                        "morph": gs.morphs.URDF(
                            file=bean_urdf,
                            scale=0.1,
                            fixed=False,
                            collision=True,
                            convexify=True,
                            decimate=False,
                            merge_fixed_links=False,
                        ),
                    }
                )
    return entries


def _task_drawer() -> list[dict]:
    import genesis as gs

    return [
        {
            # Banana primary — uses default MESH_MORPH_DIGITAL_TWIN_DEFAULT
            # (banana/textured.obj); free-floating so qpos is a 7-vector
            # (pos + quat) and rigid_/qpos_ paths agree.
            "name": "primary",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("banana/textured.obj"),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            # Acrylic drawer — articulated (free base + 1 prismatic) so qpos
            # has 8 dims. URDF has the inertia fix (outer 5.0 kg) from the
            # gs-core local override.
            "name": "drawer",
            "morph": gs.morphs.URDF(
                file=DRAWER_URDF,
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
    ]


def _task_pick_cup_tray() -> list[dict]:
    """PICK_CUP_TRAY_TWIN — pick a cup onto a tray, then bimanually lift the
    tray off the under-tray plateau onto the destination plateau.

    NPZ layout: primary_0 = cup, destination = tray,
    distractor_0 = destination plateau, distractor_1 = under-tray plateau.
    """
    import genesis as gs

    return [
        {
            "name": "primary_0",  # cup
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("cup/cup.glb"),
                euler=(90, 0, 0),
                scale=0.125,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
                # Match gs-core PICK_CUP_TRAY_TWIN: skip COACD → single convex hull.
                decompose_object_error_threshold=float("inf"),
            ),
        },
        {
            "name": "destination",  # tray
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("tray/tray.glb"),
                euler=(90, 0, 0),
                scale=0.35,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "distractor_0",  # destination plateau (fixed in gs-core)
            "morph": gs.morphs.Mesh(
                align=False,
                file=_WOODEN_PLATEAU_GLB,
                fixed=True,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "distractor_1",  # under-tray plateau (fixed in gs-core)
            "morph": gs.morphs.Mesh(
                align=False,
                file=_WOODEN_PLATEAU_GLB,
                fixed=True,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]


def _task_pick_corn_bin() -> list[dict]:
    """PICK_CORN_BIN_TWIN — pick a corn cob and drop it into a bin (with the
    bin lid sitting next to the bin as a distractor).

    NPZ layout: primary_0 = corn, destination = bin, distractor_0 = bin_lid.
    """
    import genesis as gs

    return [
        {
            "name": "primary_0",  # corn
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/corn/corn.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "destination",  # bin
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("bin/bin.glb"),
                euler=(90, 0, 0),
                scale=0.27,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "distractor_0",  # bin_lid (local override — not on HF)
            "morph": gs.morphs.Mesh(
                align=False,
                file=_BIN_LID_GLB,
                euler=(90, 0, 0),
                scale=0.27,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]


def _task_pick_potatoes() -> list[dict]:
    """PICK_POTATOES_TWIN — pick two potatoes into a green plastic bowl.

    NPZ layout: primary_0/1 = potatoes, destination = green_plastic_bowl.
    """
    import genesis as gs

    def potato_morph():
        return gs.morphs.Mesh(
            align=False,
            file=_dt("verified/potato/potato.glb"),
            scale=1.50,
            fixed=False,
            collision=True,
            convexify=True,
            decimate=True,
        )

    return [
        {"name": "primary_0", "morph": potato_morph()},
        {"name": "primary_1", "morph": potato_morph()},
        {
            "name": "destination",  # green_plastic_bowl
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/green_plastic_bowl/green_plastic_bowl.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]


def _task_ipc_pick_pen() -> list[dict]:
    """IPC_PICK_PEN — pick a pen and place it in a pen holder, IPC variant.

    Entity names match IPCPickPenEnv: ``pen`` and ``pen_holder`` (the rigid
    PICK_PEN_TWIN variant uses ``primary_0`` / ``destination`` instead).
    Morph args mirror the digital_twin defaults — same as the rigid pick_pen.
    """
    import genesis as gs

    return [
        {
            "name": "pen",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("pen/pen.glb"),
                euler=(90, 0, 0),
                scale=0.15,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "pen_holder",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("pen_holder/pen_holder.glb"),
                euler=(90, 0, 0),
                scale=0.50,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
    ]


def _task_pick_carrot_basket() -> list[dict]:
    """PICK_CARROT_BASKET_TWIN — carrot into a basket.

    NPZ layout: primary_0 = carrot, destination = basket (basket_scan.glb).
    Morph args mirror gs-core MORPH_DIGITAL_TWIN_ARGS_MAP: carrot
    euler=(90,0,0); basket_scan euler=(90,180,0), scale=0.25. Both are
    Y-up GLBs, so file_meshes_are_zup=False lets Genesis convert to Z-up.
    """
    import genesis as gs

    return [
        {
            "name": "primary_0",  # carrot
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/carrot/carrot.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
                file_meshes_are_zup=False,
            ),
        },
        {
            "name": "destination",  # basket
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("basket/basket_scan.glb"),
                euler=(90, 180, 0),
                scale=0.25,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
                file_meshes_are_zup=False,
            ),
        },
    ]


def _task_clear_table_v2() -> list[dict]:
    """TABLE_BUSSING_V2 — clear a cluttered table into recycle bins / dish tray.

    Stitched from 6 continuation MCAPs by ``scripts/stitch_table_bussing_mcaps.py``.
    Entity names match TB_V2_OBJECT_ARGS in gs-core registry.py. All morph args
    mirror MORPH_DIGITAL_TWIN_ARGS_MAP (table_bussing/* paths) so the replay
    geometry is byte-identical to teleop.

    Notes:
        * The single coffee bean instance is ``coffee_bean_0_0_0`` (V2 default
          ``num_coffee_beans_*=1``).
        * ``coffee_jar`` uses the local-override modified URDF (no inner filler
          box) so beans don't get ejected.
        * ``recycle_bin_*`` and ``white_dish_tray_*`` are ``fixed=True`` per the
          TB_V2 env override (registry.py:1395-1436). They appear in
          ``_TASK_FIXED_ENTITIES["clear_table_v2"]`` so ``load_trajectory``
          pops their NPZ array and bakes the frame-0 position into morph.pos
          before scene build. With fixed=True the body link is welded to the
          world at (morph.pos, morph.euler), so no per-frame set_pos/set_quat
          is needed (and would mis-compose with the morph euler anyway —
          Genesis URDF with fixed root joint does not synthesise a free joint).
    """
    import genesis as gs

    return [
        {
            "name": "recycle_bin_green",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/recycle_bin/03_output/recycle_bin_green.urdf"),
                align=False,
                euler=(90, 0, 0),
                fixed=True,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
            # "surface": gs.surfaces.Default(opacity=0.3),
        },
        {
            "name": "recycle_bin_grey",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/recycle_bin/03_output/recycle_bin_grey.urdf"),
                align=False,
                euler=(90, 0, 0),
                fixed=True,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "white_dish_tray_body",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/dish_tray_rack/03_output/dish_tray_body.urdf"),
                align=False,
                euler=(90, 0, -90),
                fixed=True,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "white_dish_tray_side",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/dish_tray_rack/03_output/dish_tray_caddy_large.urdf"),
                align=False,
                euler=(90, 0, 90),
                fixed=True,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "plate_a",
            "morph": gs.morphs.Mesh(
                # align=False,
                file=_dt("table_bussing/plate/03_output/plate.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            "name": "yellow_plastic_bowl_a",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/yellow_plastic_bowl/03_output/yellow_plastic_bowl_modified.urdf"),
                # align=False,
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "yellow_plastic_bowl_b",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/yellow_plastic_bowl/03_output/yellow_plastic_bowl_modified.urdf"),
                # align=False,
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "yellow_plastic_bowl_c",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/yellow_plastic_bowl/03_output/yellow_plastic_bowl_modified.urdf"),
                # align=False,
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "coffee_jar",
            "morph": gs.morphs.URDF(
                file=_TB_V2_COFFEE_JAR_URDF,
                # align=None (default): URDF parser drops the empty "world"
                # root, makes coffee_jar a free-joint root, and _align_link
                # shifts the link frame to the GLB mesh COM. This matches the
                # pose semantics that gs-core's recorder used (gs-core's
                # URDFMorphArgs schema doesn't expose `align`, so gs.morphs.URDF
                # falls back to its default None → auto). Passing align=False
                # here would put the jar's *bottom* (mesh origin, jar base) at
                # the recorded z=0.81 instead of the COM, lifting the jar by
                # ~5.8 cm.
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "coffee_bean_0_0_0",
            "morph": gs.morphs.URDF(
                file=_dt("coffee_bean/coffee_bean_simple.urdf"),
                align=False,
                scale=0.1,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "fork",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/fork/03_output/fork_modified.urdf"),
                # align=False,
                # gs-core env override: registry.py _v2_fork.morph_args.euler.
                euler=(90, 0, 60),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "spoon",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/spoon/03_output/spoon_modified.urdf"),
                # align=False,
                # gs-core env override: registry.py _v2_spoon.morph_args.euler.
                euler=(90, 0, -35),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "tomato",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/tomato/03_output/tomato.urdf"),
                align=False,
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
            ),
        },
        {
            "name": "crumpled_paper",
            "morph": gs.morphs.URDF(
                file=_dt("table_bussing/crumpled_paper/03_output/crumpled_paper.urdf"),
                align=False,
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
                merge_fixed_links=False,
                # gs-core env override: registry.py _v2_crumpled_paper sets COACD_COARSE
                # (threshold=0.3, max_convex_hull=1) — single coarse hull, less mesh detail.
                coacd_options=gs.options.CoacdOptions(threshold=0.3, max_convex_hull=1),
            ),
        },
        {
            "name": "board_eraser",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_dt("verified/board_eraser/board_eraser.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=True,
            ),
        },
        {
            # Visual-only puddle, identical to gs-core's runtime LiquidManager
            # mesh (splashsurf reconstruction of a 2.5×2.5×0.5 cm particle box
            # at liquid_pos with 50% deactivation, single FIX_XY variant).
            # collision=False mirrors gs-core; fixed=False so the recorded
            # rigid_liquid_puddle pose can drive the small per-frame z anchor
            # bias and the late-trajectory hide (z = -10) when wiped.
            "name": "liquid_puddle",
            "morph": gs.morphs.Mesh(
                align=False,
                file=_TB_V2_PUDDLE_OBJ,
                fixed=False,
                collision=False,
                convexify=False,
                decimate=False,
            ),
            # Semi-transparent dark brown — LiquidManager._LIQUID_COLOR.
            "surface": gs.surfaces.Default(color=(0.25, 0.12, 0.05, 0.7)),
        },
    ]


TASK_REGISTRY: dict[str, Any] = {
    "sort": _task_sort,
    "pour": _task_pour,
    "drawer": _task_drawer,
    "pick_cup_tray": _task_pick_cup_tray,
    "pick_corn_bin": _task_pick_corn_bin,
    "pick_potatoes": _task_pick_potatoes,
    "ipc_pick_pen": _task_ipc_pick_pen,
    "pick_carrot_basket": _task_pick_carrot_basket,
    "clear_table_v2": _task_clear_table_v2,
}

TASK_NPZ: dict[str, str] = {
    "sort": "sort_gss.npz",
    "pour": "pour_gss.npz",
    "drawer": "drawer_gss.npz",
    "pick_cup_tray": "pick_cup_tray_gss.npz",
    "pick_corn_bin": "pick_corn_bin_gss.npz",
    "pick_potatoes": "pick_potatoes_gss.npz",
    "ipc_pick_pen": "ipc_pick_pen_gss.npz",
    "pick_carrot_basket": "pick_carrot_basket_gss.npz",
    "clear_table_v2": "clear_table_v2_gss.npz",
}


# ---------------------------------------------------------------------------
# Replay class
# ---------------------------------------------------------------------------


class DigitalTwinReplay(TrajectoryReplay):
    name = "digital_twin"
    cam_pos = (1.5122, -0.767, 1.8931)
    cam_lookat = (0.838, -0.3497, 1.2837)
    cam_fov = 40
    fps = 30  # gs-core digital_twin scene runs at dt = 1/30

    # ── Render hooks (Luisa + Nyx) ────────────────────────────────────────
    def _light_intensity_factor(self) -> float:
        # Sphere lights are tuned to balance against the per-task env map. When
        # the env map is off (--no-use_env_map), the env contribution is zero,
        # so skip the dampening factor and run lights at their tuned full level.
        return _ENV_LIGHT_EFFECT if self.args.use_env_map else 1.0

    def _env_map_settings(self) -> tuple[str, float, float]:
        env_filename = _TASK_ENVMAP[self.args.task]
        env_yaw, env_registry_mult = _ENVMAP_REGISTRY[env_filename]
        env_multiplier = env_registry_mult * _OVERALL_INTENSITY if self.args.use_env_map else 0.0
        return env_filename, env_yaw, env_multiplier

    def make_camera_traj(self, name: str):
        # "ego" = the calibrated marvin RealSense D455 ego view. See
        # gs-core's EgoRealsenseD455Calib (schemas/gs_schemas/calib/registry.py).
        if name == "ego":
            return _marvin_ego_camera()
        return super().make_camera_traj(name)

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        env_filename, env_yaw, env_multiplier = self._env_map_settings()
        # ImageTexture.image_color is clamped to [0, 1]; >1 multipliers would
        # need an exposure EV bump instead, but our 3-task table is all <= 1.
        assert env_multiplier <= 1.0, f"task {self.args.task}: envmap multiplier {env_multiplier} > 1 not supported"
        light_factor = self._light_intensity_factor()
        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_radius=100.0,
            env_euler=(0, 0, env_yaw),
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path=_envmap(env_filename),
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
                for l in _LIGHTS
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
            for l in _LIGHTS
        ]

    def nyx_env_map(self):
        env_filename, env_yaw, env_multiplier = self._env_map_settings()
        return (_envmap(env_filename), env_yaw, env_multiplier)

    def add_args(self, parser):
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            choices=sorted(TASK_REGISTRY.keys()),
            help="Task to replay (one of: sort, pour, drawer, pick_cup_tray, pick_corn_bin, pick_potatoes, ipc_pick_pen, pick_carrot_basket, clear_table_v2)",
        )
        parser.add_argument(
            "--traj",
            type=str,
            default=None,
            help="Override NPZ path (default: trajectories/<task>.npz)",
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
        traj_path = Path(self.args.traj) if self.args.traj else _TRAJ_DIR / TASK_NPZ[self.args.task]
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory NPZ not found: {traj_path}")
        traj = np.load(traj_path)
        env_suffix = "" if self.args.use_env_map else "_no_envmap"
        cam_suffix = "_ego" if self.args.camera_traj == "ego" else ""
        self.args.trajectory = f"{self.args.task}{env_suffix}{cam_suffix}"
        if self.args.camera_traj == "ego":
            # Match the gs-core ego RealSense D455 vertical FOV. cam_pos/lookat
            # are overridden per-frame by _camera_traj.get_pose(); cam_fov is
            # set once at scene build time, so we have to update it here.
            self.cam_fov = _EGO_CAM_VFOV_DEG
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        self._joint_qpos = traj["robot_qpos"].astype(np.float32)
        n_dof = self._joint_qpos.shape[1]
        if n_dof not in _ROBOT_URDF_BY_DOF:
            raise ValueError(f"Unsupported robot DOF count {n_dof}; expected one of {sorted(_ROBOT_URDF_BY_DOF)}")
        self._robot_kind, self._robot_urdf = _ROBOT_URDF_BY_DOF[n_dof]
        print(f"  Robot: {self._robot_kind} ({n_dof} DOF) — {self._robot_urdf}")

        # Per-entity rigid pose array (N, 7) = pos(3) + quat_wxyz(4).
        self._rigid_data = {}
        # Per-entity full qpos when recorded — drives set_qpos for articulated
        # entities (e.g. drawer prismatic slide).
        self._entity_qpos: dict[str, np.ndarray] = {}
        for key in traj.files:
            if key.startswith("rigid_"):
                self._rigid_data[key[6:]] = traj[key].astype(np.float32)
            elif key.startswith("qpos_"):
                self._entity_qpos[key[5:]] = traj[key].astype(np.float32)

        for name, arr in self._rigid_data.items():
            extra = f" (qpos: {self._entity_qpos[name].shape[1]})" if name in self._entity_qpos else ""
            print(f"  {name}: {arr.shape[0]} frames{extra}")

        # Table pose recorded in the NPZ — depends on the env's
        # `table_x_offset` at recording time (sort/pour/drawer were captured
        # with offset=0 → table at X=0.597; the 5 newer pick_* tasks were
        # captured with offset=0.203 → table at X=0.800). The table is fixed
        # in build_scene (not in _rigid_entities), so we read its pose once
        # here and bake it into the morph.
        # NOTE: only position is taken. The recorded quat is Genesis's
        # auto-applied y_up rotation for GLB+file_meshes_are_zup=True (see
        # genesis/options/morphs.py:_resolve_zup); passing it back into the
        # morph would compose with the same auto-rotation and tip the table
        # on its side. Leave quat=None and let Genesis apply y_up once.
        table_arr = self._rigid_data.pop("table", None)
        if table_arr is None or table_arr.shape[0] == 0:
            raise RuntimeError("NPZ missing rigid_table — cannot determine table pose")
        self._table_pos = tuple(float(x) for x in table_arr[0, :3])
        print(f"  table (fixed): pos={self._table_pos}")

        # Per-task entities that gs-core declares fixed=True (env override).
        # Pop their NPZ array, capture frame-0 position, and bake it into
        # the morph at scene-build time so the body link is welded to the
        # world at the on-table pose. This mirrors gs-core's flow, where
        # _snap_objects_onto_table_at_build moves the fixed entity onto
        # the table and the resulting state becomes the captured init state.
        self._static_poses: dict[str, tuple[float, float, float]] = {}
        for name in _TASK_FIXED_ENTITIES.get(self.args.task, set()):
            arr = self._rigid_data.pop(name, None)
            if arr is None or arr.shape[0] == 0:
                continue
            pos = tuple(float(x) for x in arr[0, :3])
            self._static_poses[name] = pos
            print(f"  {name} (fixed): pos={pos}")

        # FPS from timestamps when available
        if n_frames > 1:
            dt = float(self.sim_time[1] - self.sim_time[0])
            if dt > 0:
                self.fps = max(1, min(60, int(round(1.0 / dt))))

        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        # Table — same mesh as _table_digital_twin in gs-core. Pose is read
        # from the NPZ at frame 0 because `sim_digital_twin_env._apply_table_x_offset`
        # shifts the table by an env-specific offset before recording (older
        # sort/pour/drawer NPZs: offset=0 → X=0.597; newer pick_* NPZs:
        # offset=0.203 → X=0.800). Mesh params match gs-core MORPH_TABLE:
        # convexify left to Genesis auto (None), decimate=True (default),
        # coacd_options=COACD_HQ (preprocess_resolution=150).
        scene.add_entity(
            gs.morphs.Mesh(
                align=False,
                file=TABLE_GLB,
                pos=self._table_pos,
                scale=(1.14, 1.0, 1.445),
                fixed=True,
                file_meshes_are_zup=True,
                coacd_options=gs.options.CoacdOptions(preprocess_resolution=150),
            ),
            surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0),
        )

        # Task-specific entities
        for edef in TASK_REGISTRY[self.args.task]():
            morph = edef["morph"]
            # Bake on-table pose into fixed entities (see load_trajectory).
            if edef["name"] in self._static_poses:
                morph.pos = self._static_poses[edef["name"]]
            entity = scene.add_entity(
                morph=morph,
                surface=edef.get("surface", gs.surfaces.Default()),
            )
            self._rigid_entities[edef["name"]] = entity

        # Robot — URDF picked in load_trajectory() from robot_qpos DOF count
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=self._robot_urdf,
                fixed=True,
                pos=(0.0, 0.0, 1.08),
                merge_fixed_links=False,
            ),
        )

    def apply_frame(self, scene, frame_idx: int):
        # Robot qpos
        if frame_idx < len(self._joint_qpos):
            self._robot.set_qpos(self._joint_qpos[frame_idx])

        # Per-entity playback
        for name, entity in self._rigid_entities.items():
            # Prefer recorded full qpos when shape matches the entity's n_qs
            # (e.g. drawer prismatic slide). Falls back to base pose for the
            # plain mesh entities recorded before the qpos-recording fix.
            if name in self._entity_qpos:
                qpos = self._entity_qpos[name][frame_idx]
                if qpos.shape[0] == entity.n_qs:
                    entity.set_qpos(torch.from_numpy(qpos).float().unsqueeze(0))
                    continue
            if name in self._rigid_data and frame_idx < len(self._rigid_data[name]):
                pose = self._rigid_data[name][frame_idx]
                entity.set_pos(pose[:3])
                entity.set_quat(pose[3:])


if __name__ == "__main__":
    DigitalTwinReplay().run()
