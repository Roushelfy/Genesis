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
}

# Task → EXR. Each digital-twin task gets a distinct env map, and all three are
# disjoint from Robowits's _TASK_ENVMAP so the two demos don't share skies.
_TASK_ENVMAP: dict[str, str] = {
    "sort": "brown_photostudio_02_4k.exr",
    "pour": "bright_labratory.exr",
    "drawer": "unfinished_office_night_4k.exr",
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
                file=_dt("orange/orange.glb"),
                scale=0.7,
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "fruits_primary_1",  # grapes
            "morph": gs.morphs.Mesh(
                file=str(_DEMO / "repaired" / "grapes.glb"),
                scale=0.15,
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "fruits_destination",  # grey_plate
            "morph": gs.morphs.Mesh(
                file=_dt("verified/plate/grey_plate.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "vegetables_primary_0",  # red_chili
            "morph": gs.morphs.Mesh(
                file=str(_DEMO / "repaired" / "red_chili.glb"),
                scale=0.16,
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "vegetables_primary_1",  # green_pepper
            "morph": gs.morphs.Mesh(
                file=_dt("green_pepper/green_pepper.glb"),
                scale=0.003,
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "vegetables_destination",  # sage_plate
            "morph": gs.morphs.Mesh(
                file=_dt("verified/plate/sage_plate.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
    ]


def _task_pour() -> list[dict]:
    import genesis as gs

    entries: list[dict] = [
        {
            "name": "src_container",  # yellow_plastic_bowl
            "morph": gs.morphs.Mesh(
                file=_dt("verified/yellow_plastic_bowl/yellow_plastic_bowl.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
            ),
        },
        {
            "name": "dst_container",
            "morph": gs.morphs.Mesh(
                file=_dt("verified/yellow_plastic_bowl/yellow_plastic_bowl.glb"),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                convexify=False,
                decimate=False,
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
                file=_dt("banana/textured.obj"),
                fixed=False,
                collision=True,
                convexify=True,
                decimate=False,
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
                merge_fixed_links=False,
            ),
        },
    ]


TASK_REGISTRY: dict[str, Any] = {
    "sort": _task_sort,
    "pour": _task_pour,
    "drawer": _task_drawer,
}

TASK_NPZ: dict[str, str] = {
    "sort": "sort_gss.npz",
    "pour": "pour_gss.npz",
    "drawer": "drawer_gss.npz",
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
            help="Task to replay: sort, pour, or drawer",
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
        # Pin the render output to a stable, timestamp-free name so re-renders
        # overwrite cleanly. _replay_common appends a `_YYYYMMDD_HHMMSS` tag to
        # its default stem; setting args.output bypasses that for single
        # renders (kf_idx is always None for digital_twin).
        if getattr(self.args, "render", False):
            renderer_name = "nyx" if getattr(self.args, "nyx", False) else "luisa"
            self.args.output = (
                f"data/ipc_demo/ipc_{self.name}/ipc_{self.name}_{self.args.trajectory}_{renderer_name}.mp4"
            )
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

        # FPS from timestamps when available
        if n_frames > 1:
            dt = float(self.sim_time[1] - self.sim_time[0])
            if dt > 0:
                self.fps = max(1, min(60, int(round(1.0 / dt))))

        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        # Table — same mesh and pose as _table_digital_twin in gs-core
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.597, 0.0, 0.0),
                euler=(0, 0, 0),
                scale=(1.14, 1.0, 1.445),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
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
