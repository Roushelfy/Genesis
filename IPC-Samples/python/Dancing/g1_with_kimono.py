from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

import uipc.builtin as builtin
from uipc import Logger, SceneIO, Timer, Vector3, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli,
    ElasticModuli2D,
    SoftVertexTriangleStitch,
    StrainLimitingBaraffWitkinShell,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, ground, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import GPa

from asset_dir import AssetDir
from motion_replay import create_motion_replay_player

IGNORE_LINK_PATTERNS: list[str] = []

BELT_REST_SCALE = 0.95


@dataclass
class ClothPiece:
    name: str
    init_obj: Path
    rest_obj: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay G1 with kimono in pure pyuipc.")
    parser.add_argument("--no-gui", action="store_true", help="Run replay without opening GUI.")
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(AssetDir.output_path(__file__))
    urdf_path = repo_root / "DemoAssets" / "locomotion" / "assets" / "g1_29dof_rev_1_0.urdf"
    npz_path = repo_root / "DemoAssets" / "locomotion" / "dataset.npz"
    kimono_dir = repo_root / "IPC-Samples" / "python" / "Wearing" / "results" / "kimono_v0"
    warmup_joint_json = kimono_dir / "joint_pose.json"
    return output_dir, urdf_path, npz_path, warmup_joint_json, kimono_dir


def discover_kimono_pieces(kimono_dir: Path) -> list[ClothPiece]:
    init_paths = sorted(kimono_dir.glob("wearing_*_init-shape_fixed.obj"))
    if not init_paths:
        raise FileNotFoundError(f"No fixed kimono init shapes found in {kimono_dir}")

    pieces: list[ClothPiece] = []
    for init_path in init_paths:
        stem = init_path.stem
        suffix = "_init-shape_fixed"
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        rest_path = kimono_dir / f"{base}_rest-shape.obj"
        if not rest_path.exists():
            raise FileNotFoundError(f"Missing rest-shape for {init_path.name}: {rest_path.name}")
        pieces.append(ClothPiece(name=base, init_obj=init_path, rest_obj=rest_path))

    if not pieces:
        raise RuntimeError(f"No valid kimono pieces discovered in {kimono_dir}.")
    return pieces


def _cloth_piece_original_name(piece_name: str) -> str:
    """``wearing_kimono_belt_0`` -> ``kimono_belt``."""
    s = re.sub(r"^wearing_", "", piece_name)
    s = re.sub(r"_\d+$", "", s)
    return s


def _is_belt_piece(piece: ClothPiece) -> bool:
    return "belt" in piece.name.lower()


def build_scene(output_dir: Path, pieces: list[ClothPiece]):
    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["contact"]["friction"]["enable"] = True
    config["newton"]["velocity_tol"] = 0.7
    # config["newton"]["semi_implicit"] = True
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    scene = Scene(config)
    scene.animator().substep(10)

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)
    cloth_contact = tabular.create("cloth_contact")
    ground_contact = tabular.create("ground_contact")

    # ground_geo = ground(-0.7, Vector3.UnitZ())
    # ground_contact.apply_to(ground_geo)
    # ground_obj = scene.objects().create("ground")
    # ground_obj.geometries().create(ground_geo)

    shell = StrainLimitingBaraffWitkinShell()
    bending = DiscreteShellBending()
    moduli = ElasticModuli2D.youngs_poisson(1.0e4, 0.499)
    cloth_io = SimplicialComplexIO()

    cloth_slots: list[tuple[ClothPiece, object, object]] = []
    for idx, piece in enumerate(pieces):
        cloth_mesh = cloth_io.read(str(piece.init_obj))
        rest_source = cloth_io.read(str(piece.rest_obj))
        label_surface(cloth_mesh)
        mesh_partition(cloth_mesh)

        shell.apply_to(cloth_mesh, moduli=moduli, mass_density=200.0, thickness=1e-6)
        bending.apply_to(cloth_mesh, bending_stiffness=300.0)
        cloth_contact.apply_to(cloth_mesh)

        rest_mesh = cloth_mesh.copy()
        rest_pos = np.asarray(view(rest_source.positions()), copy=True)
        if view(rest_mesh.positions()).shape != rest_pos.shape:
            raise ValueError(
                f"Rest shape mismatch for {piece.name}: "
                f"init={view(rest_mesh.positions()).shape}, rest={rest_pos.shape}"
            )

        if _is_belt_piece(piece):
            centroid = rest_pos.mean(axis=0)
            view(rest_mesh.positions())[:] = centroid + (rest_pos - centroid) * BELT_REST_SCALE
            print(f"[belt] scaled rest shape to {BELT_REST_SCALE} for {piece.name}")
        else:
            view(rest_mesh.positions())[:] = rest_pos

        cloth_object = scene.objects().create(f"cloth_{idx}_{piece.name}")
        cloth_geo_slot, cloth_rest_geo_slot = cloth_object.geometries().create(cloth_mesh, rest_mesh)
        cloth_slots.append((piece, cloth_geo_slot, cloth_rest_geo_slot))

    return engine, world, scene, cloth_contact, ground_contact, cloth_slots


def load_stitch_configs(kimono_dir: Path) -> list[dict]:
    configs: list[dict] = []
    for p in sorted(kimono_dir.glob("stitch_*.json")):
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


def build_player(
    scene: Scene,
    urdf_path: Path,
    npz_path: Path,
    warmup_joint_json: Path,
    cloth_contact,
    ground_contact,
):
    player = create_motion_replay_player(
        scene=scene,
        urdf_path=urdf_path,
        npz_path=npz_path,
        episode="demo_0",
        warmup_joint_json_path=warmup_joint_json,
        warmup_frames=60,
        loop=True,
        object_prefix="g1_link",
        stc_strength=np.array([10.0, 10.0], dtype=np.float64),
        enable_robot_robot_contact=False,
        enable_default_robot_contact=False,
        ignore_link_patterns=IGNORE_LINK_PATTERNS,
        shell_thickness=0.0001,
        mesh_source="visual",
    )
    player.sequence.qpos_joint_order = player.sequence.qpos_joint_order[:298]

    tabular = scene.contact_tabular()
    robot_contact = player.driver.robot_contact
    tabular.insert(cloth_contact, cloth_contact, 0.5, 1e7, True)
    tabular.insert(cloth_contact, robot_contact, 0.2, 1e8, True)
    tabular.insert(robot_contact, robot_contact, 0, 0, False)
    tabular.insert(ground_contact, robot_contact, 0, 0, False)
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

    state = {"run": False, "steps_per_tick": 1, "CurrentFrame": 0}

    sio.write_surface(f"{str(workspace)}/surface_{world.frame()}.obj")

    def on_update() -> None:
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
                world.dump()
                state["CurrentFrame"] = world.frame()
                sio.write_surface(f"{str(workspace)}/surface_{world.frame()}.obj")
                Timer.report()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    args = parse_args()
    output_dir, urdf_path, npz_path, warmup_joint_json, kimono_dir = default_paths()

    pieces = discover_kimono_pieces(kimono_dir)
    print(f"[kimono] discovered {len(pieces)} pieces: {[p.name for p in pieces]}")

    engine, world, scene, cloth_contact, ground_contact, cloth_slots = build_scene(output_dir, pieces)

    stitch_configs = load_stitch_configs(kimono_dir)
    if stitch_configs:
        apply_stitches(scene, stitch_configs, cloth_slots)

    player = build_player(scene, urdf_path, npz_path, warmup_joint_json, cloth_contact, ground_contact)

    root_tf = player.current_root_transform
    for _piece, cloth_geo_slot, cloth_rest_geo_slot in cloth_slots:
        view(cloth_geo_slot.geometry().transforms())[0] = root_tf
        view(cloth_rest_geo_slot.geometry().transforms())[0] = root_tf

    world.init(scene)
    world.retrieve()
    world.recover(298)

    if args.no_gui:
        run_no_gui(world, player)
        return

    _ = engine
    run_gui(world, scene, player, output_dir)


if __name__ == "__main__":
    main()
