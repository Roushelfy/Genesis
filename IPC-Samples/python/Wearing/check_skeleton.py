"""
Load kimono_v0 cloth pieces (one uipc mesh per OBJ, no rigid transform) with the G1 URDF
skeleton and capsule proxies. Use the GUI to reduce initial penetration, then Export rest/init
and Save Joint.

Each piece is a separate wearing target; exports are wearing_<name>_<i>_init-shape.obj /
rest-shape.obj. shrink_init_shape.py still expects a single wearing_cloth_0_* pair — use a
one-piece garment there or extend shrink to load multiple OBJs.

``joint_pose.json`` / ``inflation_scaling.json`` are loaded from the **first** existing path among:
``DemoAssets/kimono_v0/``, the run ``output`` dir (default ``Wearing/results/kimono_v0``), then
``Wearing/results/v1/``. Loading runs **after** ``setup_scene`` (defined in this file) and **before**
``world.init`` so the solver starts from the saved pose and inflation.

Those JSON files adjust **URDF pose and proxy inflation only** — they do **not** move cloth
vertices. The kimono still comes from ``kimono_v0/*.obj`` unless you add a separate path that
loads exported ``wearing_*_init-shape.obj`` meshes.

Run: python check_skeleton.py
Smoke test: python check_skeleton.py --no-gui

All **cloth** tunables (constitution + cloth–proxy contact) live in ``CLOTH`` (``KimonoClothParams``)
below — edit that object only; ``main`` passes it into ``setup_scene``. Other scene/solver keys stay
inside ``setup_scene`` unless you lift them the same way.

``setup_scene`` sets ``config["sanity_check"]["enable"] = 0`` so ``world.init`` stays permissive for
tight cloth/proxy setups. A **manual** check still runs via ``WearingRuntimeAPI.run_sanity_check()``
(GUI: once after init before the window loop; no-gui: after steps). Use the ImGui "Sanity Check"
button anytime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polyscope as ps

from uipc import Logger, Timer, builtin, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.core import Engine, Scene, World, SceneIO
from uipc.geometry import SimplicialComplexIO as GeometryIO
from uipc.geometry import label_surface, linemesh
from uipc.gui import SceneGUI

from wearing_core import WearingCore
from wearing_gui import WearingGUIController
from wearing_runtime import WearingRuntimeAPI
from wearing_urdf import UrdfForwardAdapter

_NO_GUI_STEPS = 10


@dataclass
class KimonoClothParams:
    """Kimono cloth: StrainLimitingBaraffWitkinShell + DiscreteShellBending + wearing cloth–proxy policy."""

    thickness: float = 1e-6
    young_modulus: float = 5.0e2
    poisson_ratio: float = 0.45
    mass_density: float = 200.0
    bending_stiffness: float = 100.0
    cloth_proxy_friction: float = 0.1
    cloth_proxy_restitution: float = 1e9


# Single place to tune cloth for this script (passed explicitly from ``main``).
CLOTH = KimonoClothParams()


def _default_asset_path(repo_root: Path, relative_name: str) -> Path:
    candidates = [repo_root / "DemoAssets" / relative_name]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_assets_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def _default_runtime_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "output" / "python" / "Wearing"


def _build_named_linemesh(vertices: np.ndarray, edges: np.ndarray, edge_names: list[str]) -> None:
    mesh = linemesh(vertices, edges)
    if len(edge_names) == int(edges.shape[0]):
        try:
            name_attr = mesh.edges().create("name", "")
            name_view = view(name_attr)
            for i, nm in enumerate(edge_names):
                name_view[i] = str(nm)
        except Exception:
            pass


def setup_scene(
    args: argparse.Namespace,
    cloth_targets: list[tuple[Any, str]],
    assets_output_dir: str | Path | None = None,
    runtime_workspace_dir: str | Path | None = None,
    cloth: KimonoClothParams | None = None,
) -> tuple[World, SceneGUI | None, WearingRuntimeAPI, UrdfForwardAdapter]:
    c = cloth if cloth is not None else KimonoClothParams()
    assets_dir = Path(assets_output_dir) if assets_output_dir is not None else _default_assets_output_dir()
    workspace_dir = Path(runtime_workspace_dir) if runtime_workspace_dir is not None else _default_runtime_workspace_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = Path(args.urdf) if args.urdf else None
    if urdf_path is None or (not urdf_path.exists()):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    Timer.disable_all()
    Logger.set_level(Logger.Level.Info)
    engine = Engine(args.backend, str(workspace_dir))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.001
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["newton"]["velocity_tol"] = 0.1
    config["newton"]["transrate_tol"] = 10.0
    config["contact"]["enable"] = True
    # config["contact"]["constitution"] = "al-ipc"
    # config["newton"]["max_iter"] = 16
    config["newton"]["semi_implicit"]["enable"] = 1
    config["newton"]["semi_implicit"]["beta_tol"] = 1e-3
    config["sanity_check"]["enable"] = 0
    config["linear_system"]["precond"]["mas"]["contact_aware"] = 1
    config["linear_system"]["solver"] = "linear_pcg"
    scene = Scene(config)
    scene.contact_tabular().default_model(0.05, 1e8)

    if not cloth_targets:
        raise ValueError("cloth_targets must contain at least one (mesh, name) pair.")

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(float(c.young_modulus), float(c.poisson_ratio))
    for cloth_mesh, _name in cloth_targets:
        label_surface(cloth_mesh)
        shell.apply_to(
            cloth_mesh,
            moduli=moduli,
            mass_density=float(c.mass_density),
            thickness=float(c.thickness),
        )
        bending.apply_to(cloth_mesh, bending_stiffness=float(c.bending_stiffness))
        cloth_is_dynamic = cloth_mesh.vertices().find(builtin.is_dynamic)
        view(cloth_is_dynamic)[:] = 0

    use_partition = not bool(getattr(args, "disable_mesh_partition", False))
    core = WearingCore(scene=scene, world=world, use_mesh_partition=True)
    for cloth_mesh, target_name in cloth_targets:
        core.add_target_mesh(cloth_mesh, target_name)

    urdf_adapter = UrdfForwardAdapter(scene, str(urdf_path), mesh_source=args.urdf_mesh_source)
    (
        skeleton_vertices,
        skeleton_edges,
        edge_joint_names,
        capsule_radii,
        edge_has_geometry,
        capsule_starts,
        capsule_ends,
        edge_group_keys,
    ) = urdf_adapter.line_skeleton()
    if skeleton_vertices.shape[0] == 0 or skeleton_edges.shape[0] == 0:
        raise RuntimeError(f"URDF skeleton is empty: {urdf_path}")
    _build_named_linemesh(skeleton_vertices, skeleton_edges, edge_joint_names)

    grouped_radius = np.asarray(capsule_radii, dtype=np.float64).copy()
    radius_groups: dict[str, list[float]] = {}
    for i in range(grouped_radius.shape[0]):
        if i < edge_has_geometry.shape[0] and not bool(edge_has_geometry[i]):
            continue
        key = edge_group_keys[i] if i < len(edge_group_keys) else f"group_{i}"
        radius_groups.setdefault(key, []).append(float(grouped_radius[i]))
    group_radius_value: dict[str, float] = {
        key: float(np.median(np.asarray(vals, dtype=np.float64))) for key, vals in radius_groups.items() if len(vals) > 0
    }
    for i in range(grouped_radius.shape[0]):
        key = edge_group_keys[i] if i < len(edge_group_keys) else f"group_{i}"
        if key in group_radius_value:
            grouped_radius[i] = group_radius_value[key]

    valid_radius_for_init: list[float] = []
    for i in range(grouped_radius.shape[0]):
        if i < edge_has_geometry.shape[0] and not bool(edge_has_geometry[i]):
            continue
        valid_radius_for_init.append(float(grouped_radius[i]))
    if len(valid_radius_for_init) > 0:
        global_initial_radius = float(max(0.002, np.median(np.asarray(valid_radius_for_init, dtype=np.float64)) * 0.1))
    else:
        global_initial_radius = 0.002

    proxy_edge_indices: list[int] = []
    for i, edge in enumerate(skeleton_edges):
        if i < edge_has_geometry.shape[0] and not bool(edge_has_geometry[i]):
            continue
        start = capsule_starts[i]
        end = capsule_ends[i]
        base_radius = global_initial_radius
        core.add_bone_proxy(start, end, radius=base_radius, name=f"bone_{i}")
        proxy_edge_indices.append(i)

    core.configure_collision_policy(
        cloth_proxy_friction=float(c.cloth_proxy_friction),
        cloth_proxy_restitution=float(c.cloth_proxy_restitution),
    )
    core.sync_proxy_targets()

    runtime = WearingRuntimeAPI(engine, core, world, scene, sgui=None, output_dir=assets_dir)
    runtime.attach_urdf_adapter(urdf_adapter, skeleton_edges, proxy_edge_indices, edge_joint_names)

    sgui = None
    if not args.no_gui:
        ps.init()
        ps.set_up_dir("z_up")
        sgui = SceneGUI(scene, "split")
        sgui.register()
        sgui.set_edge_width(1.0)
        urdf_adapter.register_polyscope_meshes(
            name_prefix=f"urdf_{args.urdf_mesh_source}",
            transparency=0.7,
            edge_width=0.0,
        )
        runtime.sgui = sgui

    return world, sgui, runtime, urdf_adapter, scene


def load_kimono_v0_cloth_targets(kimono_dir: Path, io: GeometryIO) -> list[tuple[Any, str]]:
    """Load each *.obj in ``kimono_dir`` (sorted by path) as its own cloth target. No transform."""
    paths = sorted(kimono_dir.glob("*.obj"))
    if not paths:
        raise FileNotFoundError(f"No .obj files found under {kimono_dir}")
    targets: list[tuple[Any, str]] = []
    for p in paths:
        mesh = io.read(str(p))
        if p.stem == "kimono_inner_lower":
            continue
        # if p.stem == "kimono_inner_upper":
        #     continue
        if p.stem == "kimono_belt":
            continue
        if p.stem == "kimono_outer":
            continue
        targets.append((mesh, p.stem))
    return targets


def _first_existing_file(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        if not d.is_dir():
            continue
        p = d / name
        if p.is_file():
            return p
    return None


def try_load_kimono_v0_saved_config(
    runtime: WearingRuntimeAPI,
    kimono_dir: Path,
    results_dir: Path,
) -> None:
    """
    Load wearing exports (GUI format) from the first directory that contains each file.

    Search order per file: ``kimono_dir`` → ``results_dir`` → ``Wearing/results/v1``.

    Safe to call **before** ``world.init`` (applies URDF + proxy geometry on the ``Scene``).
    """
    wearing_root = Path(__file__).resolve().parent
    candidates = [
        kimono_dir,
        results_dir,
        wearing_root / "results" / "v1",
    ]
    joint_path = _first_existing_file("joint_pose.json", candidates)
    inflation_path = _first_existing_file("inflation_scaling.json", candidates)
    if joint_path is not None:
        print(f"[check_skeleton] loading joint_pose: {joint_path}")
        runtime.load_joint_json(joint_path)
    else:
        print("[check_skeleton] joint_pose.json not found in:", ", ".join(str(d) for d in candidates))
    if inflation_path is not None:
        print(f"[check_skeleton] loading inflation_scaling: {inflation_path}")
        runtime.load_inflation_json(inflation_path)
    else:
        print("[check_skeleton] inflation_scaling.json not found in:", ", ".join(str(d) for d in candidates))


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_urdf = _default_asset_path(repo_root, "locomotion/assets/g1_29dof_rev_1_0.urdf")
    parser = argparse.ArgumentParser(description="Kimono + skeleton check (GUI or --no-gui smoke test).")
    parser.add_argument("--no-gui", action="store_true", help="Run a short headless test without Polyscope.")
    ns = parser.parse_args()
    return SimpleNamespace(
        backend="cuda",
        urdf=str(default_urdf),
        urdf_mesh_source="collision",
        no_gui=bool(ns.no_gui),
        disable_mesh_partition=False,
    )


def main(
    assets_output_dir: str | Path | None = None,
    runtime_workspace_dir: str | Path | None = None,
) -> None:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[3]
    kimono_dir = repo_root / "DemoAssets" / "kimono_v0"
    if not kimono_dir.is_dir():
        raise FileNotFoundError(f"Kimono directory not found: {kimono_dir}")

    io = GeometryIO()
    cloth_targets = load_kimono_v0_cloth_targets(kimono_dir, io)
    out_dir = Path(assets_output_dir) if assets_output_dir is not None else (Path(__file__).resolve().parent / "results" / "kimono_v0")
    out_dir.mkdir(parents=True, exist_ok=True)

    world, sgui, runtime, urdf_adapter, scene = setup_scene(
        args,
        cloth_targets,
        assets_output_dir=out_dir,
        runtime_workspace_dir=runtime_workspace_dir,
        cloth=CLOTH,
    )

    sio = SceneIO(scene)

    try_load_kimono_v0_saved_config(runtime, kimono_dir, out_dir)

    if args.no_gui:
        if not runtime._ensure_world_initialized():
            print("[no-gui] world init failed sanity checks; adjust cloth/proxy initialization.")
            print(runtime.latest_message)
            return
        for _ in range(_NO_GUI_STEPS):
            runtime.step_once()
        sanity_ok = runtime.run_sanity_check()
        print(f"[no-gui] frame={world.frame()} sanity_ok={sanity_ok}")
        print(runtime.latest_message)
        return

    if runtime._ensure_world_initialized():
        runtime.run_sanity_check()

    gui_controller = WearingGUIController(runtime, urdf_adapter)
    ps.set_user_callback(gui_controller.on_update)
    ps.show()

    sio.save("C:/Users/81946/Projects/LibuipcJointDesign/libuipc-samples/examples/39_test_mesh_partition_large_cloth/scene.json")


if __name__ == "__main__":
    main()
