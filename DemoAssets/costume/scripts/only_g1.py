from __future__ import annotations

import argparse
from pathlib import Path

import polyscope as ps
from polyscope import imgui

from uipc import Logger
from uipc.core import Engine, Scene, World
from uipc import Vector3
from uipc.geometry import ground
from uipc.gui import SceneGUI
from uipc.unit import GPa
from motion_replay import create_motion_replay_player

IGNORE_LINK_PATTERNS: list[str] = [
    # "head_link*",
    # "left_shoulder_yaw_link*",
    # "right_shoulder_yaw_link*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay G1 locomotion NPZ in pure pyuipc.")
    parser.add_argument("--no-gui", action="store_true", help="Run replay without opening GUI.")
    return parser.parse_args()


def default_paths() -> tuple[Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = repo_root / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf"
    npz_path = repo_root / "DemoAssets" / "g1_robot" / "dataset.npz"
    wearing_joint_json = repo_root / "DemoAssets" / "kimono" / "joint_pose.json"
    return output_dir, urdf_path, npz_path, wearing_joint_json


def build_scene(output_dir: Path) -> tuple[Engine, World, Scene]:
    Logger.set_level(Logger.Level.Warn)
    engine = Engine("cuda", str(output_dir))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.02
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = False
    config["contact"]["d_hat"] = 0.001
    config["contact"]["friction"]["enable"] = True
    config["newton"]["semi_implicit"] = True
    config["sanity_check"]["enable"] = 0
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1.0 * GPa)

    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0, Vector3.UnitZ()))
    return engine, world, scene


def build_player(scene: Scene, urdf_path: Path, npz_path: Path, wearing_joint_json: Path | None):
    return create_motion_replay_player(
        scene=scene,
        urdf_path=urdf_path,
        npz_path=npz_path,
        episode="demo_0",
        warmup_joint_json_path=wearing_joint_json,
        warmup_frames=60,
        loop=True,
        object_prefix="g1_link",
        ignore_link_patterns=IGNORE_LINK_PATTERNS,
    )


def run_no_gui(world: World, player) -> None:
    total = player.num_frames + player.warmup_frames
    for step_idx in range(total):
        world.advance()
        world.retrieve()
        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == total:
            print(f"[no-gui] step={step_idx + 1}/{total}")
    status = player.last_status
    print(f"[no-gui] done world_frame={status.world_frame} replay_frame={status.frame_index}")


def run_gui(world: World, scene: Scene, player) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    player.driver.register_ignored_polyscope_meshes(
        name_prefix="ignored_link",
        transparency=0.45,
        edge_width=0.0,
    )
    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1.0)

    run = False
    steps_per_tick = 1

    def on_update() -> None:
        nonlocal run, steps_per_tick
        if imgui.Button("Play / Pause"):
            run = not run
        imgui.SameLine()
        if imgui.Button("Step Once"):
            world.advance()
            world.retrieve()
            sgui.update()

        changed, steps_per_tick = imgui.SliderInt("Speed (steps/tick)", steps_per_tick, 1, 8)
        if changed:
            steps_per_tick = int(max(1, steps_per_tick))
        changed, loop_enabled = imgui.Checkbox("Loop Replay", bool(player.loop))
        if changed:
            player.loop = bool(loop_enabled)

        imgui.Text(f"Replay frame: {player.last_status.frame_index + 1}/{player.num_frames}")
        imgui.Text(f"Warmup frames: {player.warmup_frames}")
        imgui.Text(f"World frame: {world.frame()}")
        if run:
            for _ in range(steps_per_tick):
                world.advance()
                world.retrieve()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


def main() -> None:
    args = parse_args()
    output_dir, urdf_path, npz_path, wearing_joint_json = default_paths()

    engine, world, scene = build_scene(output_dir)
    player = build_player(
        scene,
        urdf_path,
        npz_path,
        wearing_joint_json=wearing_joint_json,
    )

    world.init(scene)
    if args.no_gui:
        run_no_gui(world, player)
        return

    _ = engine  # Keep engine referenced for entire GUI lifetime.
    run_gui(world, scene, player)


if __name__ == "__main__":
    main()
