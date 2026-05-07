"""Replay digital_twin teleop trajectories (Marvin GSS 54 DOF or Sharpa 58 DOF).

Replays NPZ trajectories produced by
``gs-core/scripts/mcap_to_digital_twin_npz.py`` against the corresponding
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
from _replay_common import TrajectoryReplay, marvin_urdf  # noqa: E402

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


def _envmap(filename: str) -> str:
    """Download a 4K HDR envmap from Digital_twin_asset (mirrors Robowits)."""
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


TASK_REGISTRY: dict[str, Any] = {
    "sort": _task_sort,
    "pour": _task_pour,
    "drawer": _task_drawer,
    "pick_cup_tray": _task_pick_cup_tray,
    "pick_corn_bin": _task_pick_corn_bin,
    "pick_potatoes": _task_pick_potatoes,
    "ipc_pick_pen": _task_ipc_pick_pen,
    "pick_carrot_basket": _task_pick_carrot_basket,
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
}


# ---------------------------------------------------------------------------
# Replay class
# ---------------------------------------------------------------------------


class DigitalTwinReplay(TrajectoryReplay):
    name = "digital_twin"
    cam_pos = (1.5, -0.9, 1.7)
    cam_lookat = (0.55, 0.0, 0.95)
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
            help="Task to replay (one of: sort, pour, drawer, pick_cup_tray, pick_corn_bin, pick_potatoes, ipc_pick_pen, pick_carrot_basket)",
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
        self.args.trajectory = f"{self.args.task}{env_suffix}"
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
            entity = scene.add_entity(
                morph=edef["morph"],
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
