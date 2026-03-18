from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui
from yourdfpy import URDF

from uipc import Logger, Vector3, builtin, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.constitution import AffineBodyShell, SoftTransformConstraint
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


def build_urdf_state(urdf_path: str | Path, joint_pose: np.ndarray):
    urdf = URDF.load(
        str(urdf_path),
        build_collision_scene_graph=True,
        load_collision_meshes=True,
        build_scene_graph=False,
        load_meshes=False,
    )
    mesh_scene = urdf.collision_scene
    if mesh_scene is None:
        raise RuntimeError(f"Failed to load collision scene from URDF: {urdf_path}")

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
    parser = argparse.ArgumentParser(description="Fix wearing init shape with static warmup pose.")
    parser.add_argument("--no-gui", action="store_true", help="Run settling without GUI.")
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = THIS_DIR / "results" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = repo_root / "DemoAssets" / "locomotion" / "assets" / "g1_29dof_rev_1_0.urdf"
    warmup_joint_json = THIS_DIR / "results" / "v1" / "joint_pose.json"
    cloth_obj = THIS_DIR / "results" / "v1" / "wearing_cloth_0_init-shape.obj"
    rest_cloth_obj = THIS_DIR / "results" / "v1" / "wearing_cloth_0_rest-shape.obj"
    return output_dir, urdf_path, warmup_joint_json, cloth_obj, rest_cloth_obj


def build_scene(workspace: Path, cloth_obj: Path, rest_cloth_obj: Path):
    Logger.set_level(Logger.Level.Warn)
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
    scene = Scene(config)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)
    cloth_contact = tabular.create("cloth_contact")

    cloth_io = SimplicialComplexIO()
    cloth_mesh = cloth_io.read(str(cloth_obj))
    rest_source = cloth_io.read(str(rest_cloth_obj))
    label_surface(cloth_mesh)
    mesh_partition(cloth_mesh)

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(5.0e4, 0.499)
    shell.apply_to(cloth_mesh, moduli=moduli, mass_density=200.0, thickness=0.0001)
    bending.apply_to(cloth_mesh, bending_stiffness=10.0)
    cloth_contact.apply_to(cloth_mesh)

    rest_mesh = cloth_mesh.copy()
    rest_pos = view(rest_source.positions())
    if view(rest_mesh.positions()).shape != rest_pos.shape:
        raise ValueError(
            f"Rest shape vertex layout mismatch: init={view(rest_mesh.positions()).shape}, rest={rest_pos.shape}"
        )
    view(rest_mesh.positions())[:] = rest_pos

    cloth_object = scene.objects().create("cloth")
    cloth_geo_slot, cloth_rest_geo_slot = cloth_object.geometries().create(cloth_mesh, rest_mesh)
    return engine, world, scene, cloth_contact, cloth_geo_slot, cloth_rest_geo_slot


def build_static_robot(scene: Scene, urdf_path: Path, warmup_joint_json: Path):
    urdf_probe = URDF.load(str(urdf_path), build_collision_scene_graph=False, load_collision_meshes=False)
    joint_names = list(urdf_probe.actuated_joint_names)
    joint_pose = load_joint_pose_vector(json_path=warmup_joint_json, joint_names=joint_names)
    mesh_nodes, node_world = build_urdf_state(urdf_path, joint_pose)

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


def configure_contacts(scene: Scene, cloth_contact, robot_contact) -> None:
    tabular = scene.contact_tabular()
    tabular.insert(cloth_contact, cloth_contact, 0.5, 1.0 * GPa, True)
    tabular.insert(cloth_contact, robot_contact, 0.5, 1.0 * GPa, True)
    tabular.insert(robot_contact, robot_contact, 0.5, 1.0 * GPa, False)


def save_fixed_cloth(cloth_geo_slot, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    io = SimplicialComplexIO()
    io.write(str(save_path), cloth_geo_slot.geometry())


def run_no_gui(world: World, cloth_geo_slot, save_fixed_path: Path, steps: int = 300) -> None:
    for i in range(steps):
        world.advance()
        world.retrieve()
        if (i + 1) % 50 == 0 or (i + 1) == steps:
            print(f"[no-gui] settle_step={i + 1}/{steps}")

    save_fixed_cloth(cloth_geo_slot, save_fixed_path)
    print(f"[no-gui] saved fixed surface: {save_fixed_path}")


def run_gui(world: World, scene: Scene, cloth_geo_slot, save_fixed_path: Path) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1.0)
    state = {}
    state["run"] = False
    state["save_path_text"] = str(save_fixed_path)

    def on_update() -> None:
        nonlocal state
        if imgui.Button("Run / Pause"):
            state["run"] = not state["run"]
        imgui.SameLine()
        if imgui.Button("Step Once"):
            world.advance()
            world.retrieve()
            sgui.update()

        if imgui.Button("Save Fixed Init Shape"):
            out_path = Path(state["save_path_text"])
            save_fixed_cloth(cloth_geo_slot, out_path)
            print(f"[gui] saved fixed surface: {out_path}")

        changed, new_path = imgui.InputText("Save Path", state["save_path_text"])
        if changed:
            state["save_path_text"] = str(new_path)

        imgui.Text(f"Frame: {world.frame()}")
        
        if state["run"]:
            world.advance()
            world.retrieve()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    args = parse_args()
    output_dir, urdf_path, warmup_joint_json, cloth_obj, rest_cloth_obj = default_paths()
    save_fixed_path = output_dir / "wearing_cloth_0_init-shape_fixed.obj"

    engine, world, scene, cloth_contact, cloth_geo_slot, cloth_rest_geo_slot = build_scene(
        output_dir, cloth_obj, rest_cloth_obj
    )
    robot_contact = build_static_robot(scene, urdf_path, warmup_joint_json)
    configure_contacts(scene, cloth_contact, robot_contact)

    world.init(scene)
    world.retrieve()

    if args.no_gui:
        run_no_gui(world, cloth_geo_slot, save_fixed_path)
        return

    _ = engine
    run_gui(world, scene, cloth_geo_slot, save_fixed_path)


if __name__ == "__main__":
    main()
