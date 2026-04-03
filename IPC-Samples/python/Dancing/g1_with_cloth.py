from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc.builtin as builtin
from uipc import Logger, SceneIO, Vector3, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, ground, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc import Timer
from asset_dir import AssetDir
from uipc.unit import GPa

from motion_replay import create_motion_replay_player

IGNORE_LINK_PATTERNS: list[str] = [
    # "head_link",
    # "left_shoulder_yaw_link*",
    # "right_shoulder_yaw_link*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay G1 with cloth in pure pyuipc.")
    parser.add_argument("--no-gui", action="store_true", help="Run replay without opening GUI.")
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = AssetDir.output_path(__file__)
    urdf_path = repo_root / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf"
    npz_path = repo_root / "DemoAssets" / "g1_robot" / "dataset.npz"
    warmup_joint_json = repo_root / "IPC-Samples" / "python" / "Wearing" / "results" / "v1" / "joint_pose.json"
    cloth_obj = (
        repo_root / "IPC-Samples" / "python" / "Wearing" / "results" / "v1" / "wearing_cloth_0_init-shape_fixed.obj"
    )
    rest_cloth_obj = (
        repo_root / "IPC-Samples" / "python" / "Wearing" / "results" / "v1" / "wearing_cloth_0_rest-shape.obj"
    )
    return output_dir, urdf_path, npz_path, warmup_joint_json, cloth_obj, rest_cloth_obj


def build_scene(output_dir: Path, cloth_obj: Path, rest_cloth_obj: Path):
    Logger.set_level(Logger.Level.Warn)
    Timer.enable_all()
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["contact"]["friction"]["enable"] = True
    config["newton"]["velocity_tol"] = 0.1
    config["newton"]["semi_implicit"] = True
    # config["sanity_check"]["enable"] = True
    scene = Scene(config)
    tabular = scene.contact_tabular()
    tabular.default_model(0.1, 1e7)
    cloth_contact = tabular.create("cloth_contact")

    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0, Vector3.UnitZ()))

    cloth_io = SimplicialComplexIO()
    cloth_mesh = cloth_io.read(str(cloth_obj))
    rest_source_mesh = cloth_io.read(str(rest_cloth_obj))
    label_surface(cloth_mesh)
    mesh_partition(cloth_mesh)

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(5.0e4, 0.46)
    shell.apply_to(cloth_mesh, moduli=moduli, mass_density=200.0, thickness=0.0001)
    bending.apply_to(cloth_mesh, bending_stiffness=10.0)
    is_dynamic = cloth_mesh.vertices().find(builtin.is_dynamic)
    # view(is_dynamic)[:] = 0
    cloth_contact.apply_to(cloth_mesh)

    # Build rest container from init mesh (same topology/attributes as simulated cloth),
    # then overwrite positions using rest-shape OBJ coordinates.
    rest_cloth_mesh = cloth_mesh.copy()
    rest_positions = view(rest_source_mesh.positions())
    if view(rest_cloth_mesh.positions()).shape != rest_positions.shape:
        raise ValueError(
            f"Rest shape vertex layout mismatch: init={view(rest_cloth_mesh.positions()).shape}, "
            f"rest={rest_positions.shape}"
        )
    view(rest_cloth_mesh.positions())[:] = rest_positions

    cloth_object = scene.objects().create("cloth")
    cloth_geo_slot, cloth_rest_geo_slot = cloth_object.geometries().create(cloth_mesh, rest_cloth_mesh)

    return engine, world, scene, cloth_contact, cloth_geo_slot, cloth_rest_geo_slot


def build_player(scene: Scene, urdf_path: Path, npz_path: Path, warmup_joint_json: Path, cloth_contact):
    player = create_motion_replay_player(
        scene=scene,
        urdf_path=urdf_path,
        npz_path=npz_path,
        episode="demo_0",
        warmup_joint_json_path=warmup_joint_json,
        warmup_frames=60,
        loop=True,
        object_prefix="g1_link",
        stc_strength=np.array([5.0, 5.0], dtype=np.float64),
        enable_robot_robot_contact=False,
        enable_default_robot_contact=False,
        ignore_link_patterns=IGNORE_LINK_PATTERNS,
        shell_thickness=0.0001,
    )
    tabular = scene.contact_tabular()
    robot_contact = player.driver.robot_contact
    # Cloth self-collision: enabled.
    tabular.insert(cloth_contact, cloth_contact, 0.05, 1e7, True)
    # Cloth-robot collision: enabled.
    tabular.insert(cloth_contact, robot_contact, 0.5, 1e8, True)
    # Robot-robot collision: disabled explicitly.
    tabular.insert(robot_contact, robot_contact, 0, 0, False)
    return player


def run_no_gui(world: World, player) -> None:
    total = player.num_frames + player.warmup_frames
    for step_idx in range(total):
        world.advance()
        world.retrieve()
        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == total:
            print(f"[no-gui] step={step_idx + 1}/{total}")
    status = player.last_status
    print(f"[no-gui] done world_frame={status.world_frame} replay_frame={status.frame_index}")


def run_gui(world: World, scene: Scene, player, workspace: Path) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    player.driver.register_ignored_polyscope_meshes(
        name_prefix="ignored_link",
        transparency=0.45,
        edge_width=0.0,
    )
    sgui = SceneGUI(scene, "split")
    sio = SceneIO(scene)
    sgui.register()
    sgui.set_edge_width(1.0)

    state = {}
    state["run"] = False
    state["steps_per_tick"] = 1
    state["CurrentFrame"] = 0

    sio.write_surface(f"{str(workspace)}/surface_{world.frame()}.obj")

    def on_update() -> None:
        nonlocal state
        if imgui.Button("Play / Pause"):
            state["run"] = not state["run"]
        imgui.SameLine()
        if imgui.Button("Step Once"):
            world.advance()
            world.retrieve()
            state["CurrentFrame"] = world.frame()
            sgui.update()

        changed, steps_per_tick = imgui.SliderInt("Speed (steps/tick)", state["steps_per_tick"], 1, 8)
        if changed:
            state["steps_per_tick"] = int(max(1, steps_per_tick))
        changed, loop_enabled = imgui.Checkbox("Loop Replay", bool(player.loop))
        if changed:
            player.loop = bool(loop_enabled)

        imgui.Text(f"Replay frame: {player.last_status.frame_index + 1}/{player.num_frames}")
        imgui.Text(f"Warmup frames: {player.warmup_frames}")
        imgui.Text(f"Current frame: {state['CurrentFrame']}")

        if state["run"]:
            for _ in range(state["steps_per_tick"]):
                world.advance()
                world.retrieve()
                state["CurrentFrame"] = world.frame()
                sio.write_surface(f"{str(workspace)}/surface_{world.frame()}.obj")
                Timer.report()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    args = parse_args()
    output_dir, urdf_path, npz_path, warmup_joint_json, cloth_obj, rest_cloth_obj = default_paths()

    engine, world, scene, cloth_contact, cloth_geo_slot, cloth_rest_geo_slot = build_scene(
        output_dir, cloth_obj, rest_cloth_obj
    )
    player = build_player(scene, urdf_path, npz_path, warmup_joint_json, cloth_contact)

    # Cloth mesh and robot were authored in the same local space.
    # Align cloth to the replay root transform before simulation starts.
    view(cloth_geo_slot.geometry().transforms())[0] = player.current_root_transform
    view(cloth_rest_geo_slot.geometry().transforms())[0] = player.current_root_transform

    world.init(scene)
    world.retrieve()

    if args.no_gui:
        run_no_gui(world, player)
        return

    _ = engine
    run_gui(world, scene, player, output_dir)


if __name__ == "__main__":
    main()
