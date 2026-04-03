import argparse
import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps

from uipc import Logger, Timer, builtin, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO as GeometryIO
from uipc.geometry import label_surface, linemesh
from uipc.gui import SceneGUI

from wearing_core import WearingCore
from wearing_gui import WearingGUIController
from wearing_runtime import WearingRuntimeAPI
from wearing_urdf import UrdfForwardAdapter

# Ignore capsule creation for links/joints matched by wildcard patterns.
# Example: ["pelvis*"]
IGNORE_LINK_PATTERNS: list[str] = []


def _asset_candidates(repo_root: Path, relative_name: str) -> list[Path]:
    return [
        repo_root / "DemoAssets" / relative_name,
    ]


def _default_asset_path(repo_root: Path, relative_name: str) -> Path:
    candidates = _asset_candidates(repo_root, relative_name)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_asset(repo_root: Path, relative_name: str) -> Path:
    candidates = _asset_candidates(repo_root, relative_name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Asset not found: {relative_name}. Tried: {', '.join(str(p) for p in candidates)}")


def _default_assets_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "v1"


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
    cloth_thickness: float = 0.0001,
    cloth_young: float = 5.0e1,
    cloth_poisson: float = 0.45,
    cloth_mass_density: float = 200.0,
    cloth_bending_stiffness: float = 10.0,
) -> tuple[World, SceneGUI | None, WearingRuntimeAPI, UrdfForwardAdapter | None]:
    assets_dir = Path(assets_output_dir) if assets_output_dir is not None else _default_assets_output_dir()
    workspace_dir = (
        Path(runtime_workspace_dir) if runtime_workspace_dir is not None else _default_runtime_workspace_dir()
    )
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
    config["gravity"] = [[0.0], [0.0], [-0.0]]
    config["newton"]["velocity_tol"] = 1.0
    config["newton"]["transrate_tol"] = 10.0
    config["newton"]["semi_implicit"]["enable"] = 1
    config["newton"]["semi_implicit"]["beta_tol"] = 1e-3
    config["sanity_check"]["enable"] = 0
    scene = Scene(config)
    scene.contact_tabular().default_model(0.05, 1e7)

    if not cloth_targets:
        raise ValueError("cloth_targets must contain at least one (mesh, name) pair.")

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(float(cloth_young), float(cloth_poisson))
    for cloth_mesh, _name in cloth_targets:
        label_surface(cloth_mesh)
        shell.apply_to(
            cloth_mesh,
            moduli=moduli,
            mass_density=float(cloth_mass_density),
            thickness=float(cloth_thickness),
        )
        bending.apply_to(cloth_mesh, bending_stiffness=float(cloth_bending_stiffness))
        cloth_is_dynamic = cloth_mesh.vertices().find(builtin.is_dynamic)
        view(cloth_is_dynamic)[:] = 0

    use_partition = not bool(getattr(args, "disable_mesh_partition", False))
    core = WearingCore(scene=scene, world=world, use_mesh_partition=use_partition)
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
        key: float(np.median(np.asarray(vals, dtype=np.float64)))
        for key, vals in radius_groups.items()
        if len(vals) > 0
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
    ignore_patterns = [str(p).strip() for p in IGNORE_LINK_PATTERNS if str(p).strip()]
    for i, edge in enumerate(skeleton_edges):
        if i < edge_has_geometry.shape[0] and not bool(edge_has_geometry[i]):
            continue
        link_name = edge_joint_names[i] if i < len(edge_joint_names) else f"bone_{i}"
        if any(fnmatch.fnmatch(str(link_name), pat) for pat in ignore_patterns):
            continue
        start = capsule_starts[i]
        end = capsule_ends[i]
        base_radius = global_initial_radius
        core.add_bone_proxy(start, end, radius=base_radius, name=f"bone_{i}")
        proxy_edge_indices.append(i)

    core.configure_collision_policy(cloth_proxy_friction=0.1, cloth_proxy_restitution=1e9)
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

    return world, sgui, runtime, urdf_adapter


def parse_arguments() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_urdf = _default_asset_path(repo_root, "g1_robot/assets/g1_29dof_rev_1_0.urdf")
    parser = argparse.ArgumentParser(description="Core-first wearing sample.")
    parser.add_argument("--backend", default="cuda", type=str, help="Backend name.")
    parser.add_argument(
        "--urdf",
        default=str(default_urdf),
        type=str,
        help="URDF file path used to build the skeleton (defaults to DemoAssets/g1_robot/assets/g1_29dof_rev_1_0.urdf).",
    )
    parser.add_argument(
        "--urdf-mesh-source",
        default="collision",
        choices=["collision", "visual"],
        help="Mesh source used to build URDF-derived skeleton and capsule estimates.",
    )
    parser.add_argument("--no-gui", action="store_true", help="Run smoke test without opening GUI.")
    parser.add_argument("--steps", default=10, type=int, help="Simulation steps used in --no-gui mode.")
    parser.add_argument(
        "--disable-mesh-partition", action="store_true", help="Disable mesh_partition for compatibility fallback."
    )
    return parser.parse_args()


def load_sweater_cloth_mesh(repo_root: Path, io: GeometryIO) -> Any:
    cloth_path = _resolve_asset(repo_root, "Sweater-Wearing.obj")
    return io.read(str(cloth_path))


def main(
    assets_output_dir: str | Path | None = None,
    runtime_workspace_dir: str | Path | None = None,
) -> None:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[3]
    io = GeometryIO()
    cloth_mesh = load_sweater_cloth_mesh(repo_root, io)
    world, sgui, runtime, urdf_adapter = setup_scene(
        args,
        [(cloth_mesh, "cloth")],
        assets_output_dir=assets_output_dir,
        runtime_workspace_dir=runtime_workspace_dir,
    )

    if args.no_gui:
        if not runtime._ensure_world_initialized():
            print("[no-gui] world init failed sanity checks; adjust cloth/proxy initialization.")
            print(runtime.latest_message)
            return
        step_count = max(1, int(args.steps))
        for _ in range(step_count):
            runtime.step_once()
        sanity_ok = runtime.run_sanity_check()
        print(f"[no-gui] frame={world.frame()} sanity_ok={sanity_ok}")
        print(runtime.latest_message)
        return

    gui_controller = WearingGUIController(runtime, urdf_adapter)
    ps.set_user_callback(gui_controller.on_update)
    ps.show()


if __name__ == "__main__":
    main()
