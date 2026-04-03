from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui
from yourdfpy import URDF

import re

from uipc import Logger, builtin, view
from uipc.constitution import AffineBodyShell, SoftTransformConstraint
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.constitution import SoftVertexTriangleStitch, ElasticModuli
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, label_surface, mesh_partition, trimesh
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

THIS_DIR = Path(__file__).resolve().parent


@dataclass
class RobotMeshNode:
    node_name: str
    local_vertices: np.ndarray
    faces: np.ndarray


@dataclass
class ClothPiece:
    name: str
    init_obj: Path
    rest_obj: Path
    fixed_out_obj: Path


def load_joint_pose_vector(json_path: str | Path, joint_names: list[str]) -> np.ndarray:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    raw = payload.get("joint_angles", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid joint pose json format: {json_path}")
    out = np.zeros((len(joint_names),), dtype=np.float64)
    for i, name in enumerate(joint_names):
        if name in raw:
            out[i] = float(raw[name])
    return out


def build_urdf_state(urdf_path: str | Path, joint_pose: np.ndarray, mesh_source: str = "collision"):
    if mesh_source not in ("collision", "visual"):
        raise ValueError("mesh_source must be either 'collision' or 'visual'.")
    load_collision = mesh_source == "collision"
    urdf = URDF.load(
        str(urdf_path),
        build_collision_scene_graph=load_collision,
        load_collision_meshes=load_collision,
        build_scene_graph=not load_collision,
        load_meshes=not load_collision,
    )
    mesh_scene = urdf.collision_scene if load_collision else urdf.scene
    if mesh_scene is None:
        raise RuntimeError(f"Failed to load {mesh_source} scene from URDF: {urdf_path}")

    joint_names = list(urdf.actuated_joint_names)
    if joint_pose.shape[0] != len(joint_names):
        raise ValueError(f"Joint pose size mismatch: {joint_pose.shape[0]} vs {len(joint_names)}")
    joint_state = {name: float(joint_pose[i]) for i, name in enumerate(joint_names)}
    urdf.update_cfg(joint_state)

    mesh_nodes: list[RobotMeshNode] = []
    node_world: dict[str, np.ndarray] = {}
    for node_name in mesh_scene.graph.nodes:
        node_tf, geom_name = mesh_scene.graph.get(node_name)
        if geom_name is None or geom_name not in mesh_scene.geometry:
            continue
        geom = mesh_scene.geometry[geom_name]
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        faces = np.asarray(geom.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            continue
        name = str(node_name)
        mesh_nodes.append(
            RobotMeshNode(
                node_name=name,
                local_vertices=np.array(vertices, copy=True),
                faces=np.array(faces, copy=True),
            )
        )
        node_world[name] = np.asarray(node_tf, dtype=np.float64)

    return mesh_nodes, node_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kimono shrink settle with static G1 warmup pose.")
    parser.add_argument("--no-gui", action="store_true", help="Run settling without GUI.")
    parser.add_argument("--steps", type=int, default=300, help="Settle steps in no-gui mode.")
    parser.add_argument(
        "--urdf-mesh-source",
        choices=["collision", "visual"],
        default="collision",
        help="Use URDF collision or visual meshes for static G1 bodies.",
    )
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = THIS_DIR / "results" / "kimono_v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = repo_root / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf"
    warmup_joint_json = output_dir / "joint_pose.json"
    return output_dir, urdf_path, warmup_joint_json


def discover_kimono_pieces(output_dir: Path) -> list[ClothPiece]:
    init_paths = sorted(output_dir.glob("wearing_*_init-shape.obj"))
    if not init_paths:
        raise FileNotFoundError(f"No kimono init shape found in {output_dir}. Expected wearing_*_init-shape.obj.")

    pieces: list[ClothPiece] = []
    for init_path in init_paths:
        stem = init_path.stem
        suffix = "_init-shape"
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        rest_path = output_dir / f"{base}_rest-shape.obj"
        if not rest_path.exists():
            raise FileNotFoundError(f"Missing rest-shape for {init_path.name}: {rest_path.name}")
        pieces.append(
            ClothPiece(
                name=base,
                init_obj=init_path,
                rest_obj=rest_path,
                fixed_out_obj=output_dir / f"{base}_init-shape_fixed.obj",
            )
        )

    if not pieces:
        raise RuntimeError(f"No valid kimono init/rest pairs discovered in {output_dir}.")
    return pieces


def build_scene(workspace: Path, pieces: list[ClothPiece]):
    Logger.set_level(Logger.Level.Info)
    engine = Engine("cuda", str(workspace))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.005
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["contact"]["friction"]["enable"] = True
    config["newton"]["velocity_tol"] = 0.1
    config["newton"]["semi_implicit"] = True
    config["sanity_check"]["enable"] = 0
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 16
    scene = Scene(config)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)
    cloth_contact = tabular.create("cloth_contact")

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(5.0e4, 0.499)
    cloth_io = SimplicialComplexIO()

    cloth_slots: list[tuple[ClothPiece, object, object]] = []
    for idx, piece in enumerate(pieces):
        cloth_mesh = cloth_io.read(str(piece.init_obj))
        rest_source = cloth_io.read(str(piece.rest_obj))
        label_surface(cloth_mesh)
        mesh_partition(cloth_mesh)

        shell.apply_to(cloth_mesh, moduli=moduli, mass_density=200.0, thickness=1e-6)
        bending.apply_to(cloth_mesh, bending_stiffness=100.0)
        cloth_contact.apply_to(cloth_mesh)

        rest_mesh = cloth_mesh.copy()
        rest_pos = view(rest_source.positions())
        if view(rest_mesh.positions()).shape != rest_pos.shape:
            raise ValueError(
                f"Rest shape mismatch for {piece.name}: init={view(rest_mesh.positions()).shape}, rest={rest_pos.shape}"
            )
        view(rest_mesh.positions())[:] = rest_pos

        cloth_object = scene.objects().create(f"cloth_{idx}_{piece.name}")
        cloth_geo_slot, cloth_rest_geo_slot = cloth_object.geometries().create(cloth_mesh, rest_mesh)
        cloth_slots.append((piece, cloth_geo_slot, cloth_rest_geo_slot))

    return engine, world, scene, cloth_contact, cloth_slots


def build_static_robot(scene: Scene, urdf_path: Path, warmup_joint_json: Path, mesh_source: str):
    load_collision = mesh_source == "collision"
    urdf_probe = URDF.load(
        str(urdf_path),
        build_collision_scene_graph=load_collision,
        load_collision_meshes=load_collision,
        build_scene_graph=not load_collision,
        load_meshes=not load_collision,
    )
    joint_names = list(urdf_probe.actuated_joint_names)
    joint_pose = load_joint_pose_vector(json_path=warmup_joint_json, joint_names=joint_names)
    mesh_nodes, node_world = build_urdf_state(urdf_path, joint_pose, mesh_source=mesh_source)

    robot_contact = scene.contact_tabular().create("g1_fix_contact")
    abs_constitution = AffineBodyShell()
    stc = SoftTransformConstraint()
    for i, node in enumerate(mesh_nodes):
        sc = trimesh(node.local_vertices, node.faces)
        label_surface(sc)
        abs_constitution.apply_to(sc, 80.0 * MPa, thickness=0.001)
        stc.apply_to(sc, np.array([5.0, 5.0], dtype=np.float64))
        robot_contact.apply_to(sc)

        obj = scene.objects().create(f"g1_fix_{i}_{node.node_name}")
        geo_slot, _ = obj.geometries().create(sc)
        geo = geo_slot.geometry()
        tf = node_world[node.node_name]
        view(geo.instances().find(builtin.is_constrained))[:] = 1
        view(geo.instances().find(builtin.aim_transform))[:] = tf
        view(geo.transforms())[:] = tf
        view(geo.instances().find(builtin.is_fixed))[:] = 1
        view(geo.instances().find(builtin.is_dynamic))[:] = 0
    return robot_contact


def _cloth_piece_original_name(piece_name: str) -> str:
    """``wearing_kimono_belt_0`` -> ``kimono_belt``."""
    s = re.sub(r"^wearing_", "", piece_name)
    s = re.sub(r"_\d+$", "", s)
    return s


def load_stitch_configs(output_dir: Path) -> list[dict]:
    """Discover and parse all ``stitch_*.json`` files under *output_dir*."""
    configs: list[dict] = []
    for p in sorted(output_dir.glob("stitch_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        configs.append(payload)
        print(f"[stitch] loaded {p.name}: {len(payload['pairs'])} pairs")
    return configs


def apply_stitches(
    scene: Scene,
    stitch_configs: list[dict],
    cloth_slots: list[tuple[ClothPiece, object, object]],
) -> None:
    slot_by_orig_name: dict[str, tuple[object, object]] = {}
    for piece, geo_slot, rest_geo_slot in cloth_slots:
        orig = _cloth_piece_original_name(piece.name)
        slot_by_orig_name[orig] = (geo_slot, rest_geo_slot)

    svts = SoftVertexTriangleStitch()
    stitch_obj = scene.objects().create("stitch")

    for cfg in stitch_configs:
        mesh_a = cfg["mesh_a"]
        mesh_b = cfg["mesh_b"]
        pairs = np.asarray(cfg["pairs"], dtype=np.int32)
        if pairs.size == 0:
            print(f"[stitch] {mesh_a} <-> {mesh_b}: 0 pairs, skipped")
            continue
        if mesh_a not in slot_by_orig_name:
            print(f"[stitch] WARNING: cloth '{mesh_a}' not found in scene, skipping")
            continue
        if mesh_b not in slot_by_orig_name:
            print(f"[stitch] WARNING: cloth '{mesh_b}' not found in scene, skipping")
            continue

        geo_a, rest_a = slot_by_orig_name[mesh_a]
        geo_b, rest_b = slot_by_orig_name[mesh_b]

        stitch_geo = svts.create_geometry(
            (geo_a, geo_b),
            (rest_a, rest_b),
            pairs,
            ElasticModuli.youngs_poisson(1.2e2, 0.49),
            min_separate_distance=0.001,
        )
        stitch_obj.geometries().create(stitch_geo)
        print(f"[stitch] applied {mesh_a} <-> {mesh_b}: {pairs.shape[0]} pairs")


def configure_contacts(scene: Scene, cloth_contact, robot_contact) -> None:
    tabular = scene.contact_tabular()
    tabular.insert(cloth_contact, cloth_contact, 0.5, 1.0 * GPa, True)
    tabular.insert(cloth_contact, robot_contact, 0.5, 1.0 * GPa, True)
    tabular.insert(robot_contact, robot_contact, 0.5, 1.0 * GPa, False)


def save_fixed_cloths(cloth_slots: list[tuple[ClothPiece, object, object]]) -> None:
    io = SimplicialComplexIO()
    for piece, cloth_geo_slot, _rest in cloth_slots:
        piece.fixed_out_obj.parent.mkdir(parents=True, exist_ok=True)
        io.write(str(piece.fixed_out_obj), cloth_geo_slot.geometry())
        print(f"[save] {piece.name}: {piece.fixed_out_obj}")


def run_no_gui(world: World, cloth_slots: list[tuple[ClothPiece, object, object]], steps: int = 300) -> None:
    for i in range(steps):
        world.advance()
        world.retrieve()
        if (i + 1) % 50 == 0 or (i + 1) == steps:
            print(f"[no-gui] settle_step={i + 1}/{steps}")

    save_fixed_cloths(cloth_slots)


def _build_stitch_vis(
    stitch_configs: list[dict],
    cloth_slots: list[tuple[ClothPiece, object, object]],
) -> list[tuple[str, np.ndarray, object, object, np.ndarray]]:
    """Build per-config data needed to draw stitch lines.

    Returns list of ``(net_name, pairs, geo_slot_a, geo_slot_b, faces_b_from_rest)``.
    ``faces_b_from_rest`` is the triangle index array (Mx3 int32) read once from rest geometry.
    """
    slot_by_orig: dict[str, tuple[object, object]] = {}
    for piece, gs, rgs in cloth_slots:
        slot_by_orig[_cloth_piece_original_name(piece.name)] = (gs, rgs)

    vis_items: list[tuple[str, np.ndarray, object, object, np.ndarray]] = []
    for cfg in stitch_configs:
        ma, mb = cfg["mesh_a"], cfg["mesh_b"]
        pairs = np.asarray(cfg["pairs"], dtype=np.int32)
        if pairs.size == 0 or ma not in slot_by_orig or mb not in slot_by_orig:
            continue
        gs_a = slot_by_orig[ma][0]
        gs_b = slot_by_orig[mb][0]
        faces_b = np.asarray(view(gs_b.geometry().triangles().topo()), copy=True).reshape(-1, 3)
        vis_items.append((f"stitch_{ma}_{mb}", pairs, gs_a, gs_b, faces_b))
    return vis_items


def _register_stitch_nets(
    vis_items: list[tuple[str, np.ndarray, object, object, np.ndarray]],
) -> dict[str, object]:
    nets: dict[str, object] = {}
    for net_name, pairs, gs_a, gs_b, faces_b in vis_items:
        nodes, edges = _stitch_nodes_edges(pairs, gs_a, gs_b, faces_b)
        if nodes.shape[0] == 0:
            continue
        nets[net_name] = ps.register_curve_network(net_name, nodes, edges, radius=0.0005)
    return nets


def _stitch_nodes_edges(
    pairs: np.ndarray,
    gs_a: object,
    gs_b: object,
    faces_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = pairs.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    pos_a = np.asarray(view(gs_a.geometry().positions()), copy=False).reshape(-1, 3)
    pos_b = np.asarray(view(gs_b.geometry().positions()), copy=False).reshape(-1, 3)

    nodes = np.zeros((2 * n, 3), dtype=np.float64)
    edges = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        vid = pairs[i, 0]
        tid = pairs[i, 1]
        nodes[2 * i] = pos_a[vid]
        tri = faces_b[tid]
        nodes[2 * i + 1] = (pos_b[tri[0]] + pos_b[tri[1]] + pos_b[tri[2]]) / 3.0
        edges[i] = [2 * i, 2 * i + 1]
    return nodes, edges


def run_gui(
    world: World,
    scene: Scene,
    cloth_slots: list[tuple[ClothPiece, object, object]],
    stitch_configs: list[dict],
) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1.0)

    for piece, _gs, _rgs in cloth_slots:
        try:
            sm = ps.get_surface_mesh(f"cloth_{cloth_slots.index((piece, _gs, _rgs))}_{piece.name}")
            sm.set_transparency(0.5)
        except Exception:
            pass

    vis_items = _build_stitch_vis(stitch_configs, cloth_slots)
    stitch_nets = _register_stitch_nets(vis_items)

    state = {"run": False}

    def on_update() -> None:
        nonlocal state
        if imgui.Button("Run / Pause"):
            state["run"] = not state["run"]
        imgui.SameLine()
        if imgui.Button("Step Once"):
            world.advance()
            world.retrieve()
            sgui.update()
            _update_stitch_nets()

        if imgui.Button("Save All Fixed Init Shape"):
            save_fixed_cloths(cloth_slots)

        imgui.Text(f"Frame: {world.frame()}")

        if state["run"]:
            world.advance()
            world.retrieve()
            sgui.update()
            _update_stitch_nets()

    def _update_stitch_nets() -> None:
        for net_name, pairs, gs_a, gs_b, faces_b in vis_items:
            if net_name not in stitch_nets:
                continue
            nodes, _ = _stitch_nodes_edges(pairs, gs_a, gs_b, faces_b)
            stitch_nets[net_name].update_node_positions(nodes)

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    args = parse_args()
    output_dir, urdf_path, warmup_joint_json = default_paths()
    pieces = discover_kimono_pieces(output_dir)

    engine, world, scene, cloth_contact, cloth_slots = build_scene(output_dir, pieces)
    robot_contact = build_static_robot(scene, urdf_path, warmup_joint_json, mesh_source="visual")
    configure_contacts(scene, cloth_contact, robot_contact)

    stitch_configs = load_stitch_configs(output_dir)
    if stitch_configs:
        apply_stitches(scene, stitch_configs, cloth_slots)

    world.init(scene)
    world.retrieve()

    if args.no_gui:
        run_no_gui(world, cloth_slots, steps=max(1, int(args.steps)))
        return

    _ = engine
    run_gui(world, scene, cloth_slots, stitch_configs)


if __name__ == "__main__":
    main()
