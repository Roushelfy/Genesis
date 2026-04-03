"""Batch-capable G1 kimono simulation and USD export.

Parameterised by dataset index (0-5) and lower-body clothing toggle.

Examples::

    # Simulate dataset 0 with lower clothing, no GUI, dump every frame
    uv run python g1_kimono_batch.py --dataset 0 --no-gui

    # Simulate dataset 2 without lower clothing
    uv run python g1_kimono_batch.py --dataset 2 --no-lower --no-gui

    # Export all dumped frames to USD
    uv run python g1_kimono_batch.py --dataset 0 --recover

    # Export first 100 frames
    uv run python g1_kimono_batch.py --dataset 0 --recover 100
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Logger, SceneIO, Timer, Vector3, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli,
    ElasticModuli2D,
    SoftVertexTriangleStitch,
    StrainLimitingBaraffWitkinShell,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import GPa

from motion_replay import create_motion_replay_player

IGNORE_LINK_PATTERNS: list[str] = []

BELT_REST_SCALE = 0.8

DATASET_FILES = [
    "dataset.npz",
    "dataset1.npz",
    "dataset2.npz",
    "dataset3.npz",
    "dataset4.npz",
    "dataset5.npz",
]


@dataclass
class ClothPiece:
    name: str
    init_obj: Path
    rest_obj: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch G1 kimono simulation / USD export.")
    parser.add_argument("--dataset", type=int, required=True, choices=range(6), help="Dataset index 0-5.")
    parser.add_argument("--no-lower", action="store_true", help="Skip lower-body clothing piece.")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI.")
    parser.add_argument(
        "--recover",
        type=int,
        nargs="?",
        const=-1,
        default=None,
        metavar="MAX_FRAME",
        help="Recover mode: load dumps and export USD. Optionally limit to MAX_FRAME frames (default: all).",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        metavar="FRAME",
        help="Resume simulation from a previously dumped frame.",
    )
    parser.add_argument(
        "--substep",
        type=int,
        default=0,
        metavar="N",
        help="Override animator substep (default: use script constant).",
    )
    parser.add_argument(
        "--settle",
        type=int,
        default=0,
        metavar="N",
        help="Settle mode: recover to last dumped frame, then run N extra frames "
        "with robot frozen to let cloth reach equilibrium.",
    )
    return parser.parse_args()


def variant_name(dataset_idx: int, no_lower: bool) -> str:
    return f"d{dataset_idx}_{'no_lower' if no_lower else 'lower'}"


def get_paths(dataset_idx: int, no_lower: bool) -> tuple[Path, Path, Path, Path, Path, str]:
    repo_root = Path(__file__).resolve().parents[3]
    vname = variant_name(dataset_idx, no_lower)
    output_dir = repo_root / "DemoAssets" / "costume" / "output" / "g1_kimono_batch" / vname
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = repo_root / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf"
    npz_path = repo_root / "DemoAssets" / "g1_robot" / DATASET_FILES[dataset_idx]
    kimono_dir = repo_root / "DemoAssets" / "kimono"
    warmup_joint_json = kimono_dir / "joint_pose.json"
    return output_dir, urdf_path, npz_path, warmup_joint_json, kimono_dir, vname


def _cloth_piece_original_name(piece_name: str) -> str:
    """``wearing_kimono_belt_0`` -> ``kimono_belt``."""
    s = re.sub(r"^wearing_", "", piece_name)
    s = re.sub(r"_\d+$", "", s)
    return s


def _is_belt_piece(piece: ClothPiece) -> bool:
    return "belt" in piece.name.lower()


def discover_kimono_pieces(kimono_dir: Path, skip_pieces: set[str]) -> list[ClothPiece]:
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
        orig_name = _cloth_piece_original_name(base)
        if orig_name in skip_pieces:
            print(f"[kimono] skipping {base} (in skip list)")
            continue
        rest_path = kimono_dir / f"{base}_rest-shape.obj"
        if not rest_path.exists():
            raise FileNotFoundError(f"Missing rest-shape for {init_path.name}: {rest_path.name}")
        pieces.append(ClothPiece(name=base, init_obj=init_path, rest_obj=rest_path))

    if not pieces:
        raise RuntimeError(f"No valid kimono pieces discovered in {kimono_dir}.")
    return pieces


def build_scene(output_dir: Path, pieces: list[ClothPiece], substep_override: int = 0):
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
    config["newton"]["velocity_tol"] = 0.5
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    scene = Scene(config)
    actual_substep = substep_override if substep_override > 0 else 20
    scene.animator().substep(actual_substep)
    print(f"[scene] animator substep = {actual_substep}")

    tabular = scene.contact_tabular()
    tabular.default_model(0.5, 1.0 * GPa)
    cloth_contact = tabular.create("cloth_contact")
    ground_contact = tabular.create("ground_contact")

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
                f"Rest shape mismatch for {piece.name}: init={view(rest_mesh.positions()).shape}, rest={rest_pos.shape}"
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
        if mesh_a not in slot_by_orig_name or mesh_b not in slot_by_orig_name:
            skipped = [m for m in (mesh_a, mesh_b) if m not in slot_by_orig_name]
            print(f"[stitch] {mesh_a} <-> {mesh_b}: skipped (missing: {skipped})")
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
        loop=False,
        object_prefix="g1_link",
        stc_strength=np.array([10.0, 10.0], dtype=np.float64),
        enable_robot_robot_contact=False,
        enable_default_robot_contact=False,
        ignore_link_patterns=IGNORE_LINK_PATTERNS,
        shell_thickness=0.0001,
        mesh_source="visual",
    )
    total_qpos = len(player.sequence.qpos_joint_order)
    player.sequence.qpos_joint_order = player.sequence.qpos_joint_order[: total_qpos - 1]
    print(f"[player] using {total_qpos - 1}/{total_qpos} replay frames from {npz_path.name}")

    tabular = scene.contact_tabular()
    robot_contact = player.driver.robot_contact
    tabular.insert(cloth_contact, cloth_contact, 0.5, 1e7, True)
    tabular.insert(cloth_contact, robot_contact, 0.2, 1e8, True)
    tabular.insert(robot_contact, robot_contact, 0, 0, False)
    tabular.insert(ground_contact, robot_contact, 0, 0, False)
    return player


def run_no_gui(world: World, player, resume_from: int = 0) -> None:
    total = player.num_frames + player.warmup_frames
    start = resume_from if resume_from > 0 else 0
    if start > 0:
        print(f"[no-gui] resuming from step {start}")
    for step_idx in range(start, total):
        world.advance()
        world.retrieve()
        world.dump()
        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == total:
            print(f"[no-gui] step={step_idx + 1}/{total}")
    status = player.last_status
    print(f"[no-gui] done world_frame={status.world_frame} replay_frame={status.frame_index}")


def run_settle(world: World, workspace: Path, extra_frames: int) -> None:
    """Recover to last dumped frame, then run extra_frames more to let cloth settle."""
    last = _find_max_dump_frame(workspace)
    if last == 0:
        print("[settle] no dump files found, nothing to do")
        return
    print(f"[settle] recovering to frame {last}, then running {extra_frames} extra frames ...")
    world.recover(last)
    for i in range(1, extra_frames + 1):
        world.advance()
        world.retrieve()
        world.dump()
        frame_num = last + i
        if i % 10 == 0 or i == extra_frames:
            print(f"[settle] extra frame {i}/{extra_frames} (total frame {frame_num})")
    print(f"[settle] done. total frames now: {_find_max_dump_frame(workspace)}")


def _find_max_dump_frame(workspace: Path) -> int:
    dump_dir = workspace / "dump" / "common" / "sim_engine.cpp"
    if not dump_dir.exists():
        return 0
    max_frame = 0
    for p in dump_dir.glob("state.*.json"):
        try:
            n = int(p.stem.split(".")[1])
            max_frame = max(max_frame, n)
        except (IndexError, ValueError):
            continue
    return max_frame


def run_recover(
    world: World,
    scene: Scene,
    workspace: Path,
    cloth_slots: list[tuple[ClothPiece, object, object]],
    player,
    vname: str,
    max_frame: int = -1,
) -> None:
    from usd_exporter import UsdExporter, read_obj_faces

    available = _find_max_dump_frame(workspace)
    if available == 0:
        print(f"[recover] no dump files found in {workspace}")
        return
    total = available if max_frame <= 0 else min(max_frame, available)
    print(f"[recover] found {available} dump frames, will export {total}")

    dt = 0.01
    fps = 1.0 / dt
    usd_path = workspace / "animation.usdc"
    exporter = UsdExporter(str(usd_path), fps=fps, root_prim=vname)

    for piece, geo_slot, _rest in cloth_slots:
        faces = read_obj_faces(str(piece.init_obj))
        exporter.add_deformable(piece.name, geo_slot, faces)

    for binding in player.driver.bindings:
        node = next(n for n in player.urdf_kinematics.mesh_nodes if n.node_name == binding.node_name)
        exporter.add_rigid(binding.node_name, binding.geo_slot, node.local_vertices, node.faces)

    for frame in range(1, total + 1):
        world.recover(frame)
        world.retrieve()
        exporter.capture_frame()
        if frame % 50 == 0:
            print(f"[recover] exported frame {frame}/{total}")

    exporter.close()
    print(f"[recover] done, exported {total} frames to {usd_path}")


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
    skip_pieces: set[str] = {"kimono_inner_lower"} if args.no_lower else set()
    output_dir, urdf_path, npz_path, warmup_joint_json, kimono_dir, vname = get_paths(args.dataset, args.no_lower)

    print(f"[batch] variant={vname}, dataset={DATASET_FILES[args.dataset]}, skip={skip_pieces or 'none'}")

    pieces = discover_kimono_pieces(kimono_dir, skip_pieces)
    print(f"[kimono] discovered {len(pieces)} pieces: {[p.name for p in pieces]}")

    engine, world, scene, cloth_contact, ground_contact, cloth_slots = build_scene(
        output_dir, pieces, substep_override=args.substep
    )

    stitch_configs = load_stitch_configs(kimono_dir)
    if stitch_configs:
        apply_stitches(scene, stitch_configs, cloth_slots)

    player = build_player(scene, urdf_path, npz_path, warmup_joint_json, cloth_contact, ground_contact)

    root_tf = player.current_root_transform  # 4x4 column-vector convention
    rot = root_tf[:3, :3]
    trans = root_tf[:3, 3].flatten()
    for _piece, cloth_geo_slot, cloth_rest_geo_slot in cloth_slots:
        for slot in (cloth_geo_slot, cloth_rest_geo_slot):
            pos = view(slot.geometry().positions())
            pos_squeezed = np.asarray(pos).squeeze()  # (N, 3)
            transformed = (rot @ pos_squeezed.T).T + trans
            pos[:] = transformed.reshape(pos.shape)

    world.init(scene)
    world.retrieve()

    if args.settle > 0:
        run_settle(world, output_dir, args.settle)
        run_recover(world, scene, output_dir, cloth_slots, player, vname)
        return

    if args.resume_from > 0:
        print(f"[batch] recovering to frame {args.resume_from} before simulation ...")
        world.recover(args.resume_from)

    if args.recover is not None:
        run_recover(world, scene, output_dir, cloth_slots, player, vname, max_frame=args.recover)
        return

    if args.no_gui:
        run_no_gui(world, player, resume_from=args.resume_from)
        return

    _ = engine
    run_gui(world, scene, player, output_dir)


if __name__ == "__main__":
    main()
