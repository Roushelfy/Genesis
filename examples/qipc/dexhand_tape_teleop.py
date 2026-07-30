"""Keyboard teleoperation for the reusable Marvin Wuji QIPC tape world.

Keyboard Controls:
    1 / 2 / 3   - control the right hand / left hand / both
    Arrow keys  - move the selected palm target in XY
    J/K         - move down/up
    N/M         - yaw left/right
    U/O         - pitch up/down
    L/;         - roll left/right
    Space       - hold to close the selected thumb/index, release to open
    Backslash   - reset targets to the current palm poses
    Esc         - quit
"""

import argparse
import time

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

DELTA_POS = 0.003
DELTA_ROT = 0.02


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=None, help="marvin_wuji_capsule_scaled.urdf")
    parser.add_argument("--mode", choices=["bond", "soft"], default="bond", help="tape adhesion mode")
    parser.add_argument(
        "--sticky-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="adhesive contact between the hands and tape",
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="run N steps without the viewer and exit",
    )
    parser.add_argument(
        "--probe-close-step",
        type=int,
        default=None,
        help="headless step at which the right thumb/index start closing",
    )
    parser.add_argument(
        "--probe-release-step",
        type=int,
        default=None,
        help="headless step at which the right thumb/index are released",
    )
    parser.add_argument("--newton-tol", type=float, default=0.01)
    parser.add_argument("--linear-tol", type=float, default=None)
    parser.add_argument("--linear-max-iter", type=int, default=None)
    return parser.parse_args()


def _run_headless(world, args: argparse.Namespace) -> None:
    initial_palm = None
    step_ms = []
    newton_iterations = []
    pcg_iterations = []

    for step in range(args.headless_steps):
        if step == args.probe_close_step:
            world.set_grip("right", True)
            initial_palm = world.palm_position("right")
            print("[probe] closing right thumb/index", flush=True)
        if step == args.probe_release_step:
            world.set_grip("right", False)
            print("[probe] opening right thumb/index", flush=True)

        start = time.perf_counter()
        stats = world.step()
        step_ms.append((time.perf_counter() - start) * 1e3)
        newton_iterations.append(stats.newton_iters)
        pcg_iterations.append(stats.max_pcg_iters)

        if step in {
            args.probe_close_step,
            args.probe_release_step,
            args.headless_steps - 1,
        }:
            hand = world.hand_dofs_position("right")
            torque = world.hand_dofs_applied_force("right")
            print(
                f"[probe] step={step:03d} palm={np.round(world.palm_position('right'), 4)} "
                f"hand=[{hand.min():.3f},{hand.max():.3f}] "
                f"max_tau={np.abs(torque).max():.3f} "
                f"newton={stats.newton_iters} pcg={stats.max_pcg_iters} "
                f"{step_ms[-1]:.0f}ms",
                flush=True,
            )

    tape_positions = world.tape_positions()
    drift = float(np.linalg.norm(world.palm_position("right") - initial_palm)) if initial_palm is not None else 0.0
    linear_cap = int(world.coupler_options["solver_linear_max_iter"])
    print(
        f"[headless] step ms median={np.median(step_ms):.0f} "
        f"mean={np.mean(step_ms):.0f} max={max(step_ms):.0f} | "
        f"newton max={max(newton_iterations)} | "
        f"pcg median={np.median(pcg_iterations):.0f} max={max(pcg_iterations)} "
        f"at_cap={sum(value >= linear_cap for value in pcg_iterations)}/{len(pcg_iterations)} "
        f"(tol_rate={world.coupler_options['solver_linear_tol_rate']})",
        flush=True,
    )
    print(
        f"[headless] tape z=[{tape_positions[:, 2].min():.4f},"
        f"{tape_positions[:, 2].max():.4f}] "
        f"finite={np.isfinite(tape_positions).all()} palm_drift={drift:.4f}m "
        f"right_palm={np.round(world.palm_position('right'), 3)}",
        flush=True,
    )


def _run_interactive(world) -> None:
    if world.scene.viewer is None:
        gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
        return
    world.scene.viewer.update(force=True)

    active = {"right"}
    gizmos = {
        side: world.scene.draw_debug_frame(
            T=gu.trans_quat_to_T(*world.palm_target(side)),
            axis_length=0.12,
            origin_size=0.008,
            axis_radius=0.005,
        )
        for side in ("right", "left")
    }
    running = True
    gripping = set()

    def select(sides) -> None:
        active.clear()
        active.update(sides)
        gs.logger.info(f"Controlling: {'+'.join(sorted(active))}")

    def move(delta) -> None:
        for side in active:
            world.move_palm_target(side, delta)

    def rotate(axis: int, amount: float) -> None:
        delta = np.zeros(3, dtype=gs.np_float)
        delta[axis] = amount
        for side in active:
            world.rotate_palm_target(side, delta)

    def set_grip(closed: bool) -> None:
        if closed:
            gripping.clear()
            gripping.update(active)
            for side in gripping:
                world.set_grip(side, True)
        else:
            for side in gripping:
                world.set_grip(side, False)
            gripping.clear()
        gs.logger.info(
            f"Grip: right={'closed' if world.grip_is_closed('right') else 'open'} "
            f"left={'closed' if world.grip_is_closed('left') else 'open'}"
        )

    def stop() -> None:
        nonlocal running
        running = False

    world.scene.viewer.register_keybinds(
        Keybind("select_right", Key._1, KeyAction.RELEASE, callback=select, args=({"right"},)),
        Keybind("select_left", Key._2, KeyAction.RELEASE, callback=select, args=({"left"},)),
        Keybind("select_both", Key._3, KeyAction.RELEASE, callback=select, args=({"right", "left"},)),
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_targets", Key.BACKSLASH, KeyAction.RELEASE, callback=world.reset_targets),
        Keybind("close_grip", Key.SPACE, KeyAction.PRESS, callback=set_grip, args=(True,)),
        Keybind("open_grip", Key.SPACE, KeyAction.RELEASE, callback=set_grip, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        overwrite=True,
    )
    gs.logger.info(
        "Keys: 1/2/3 pick hand(s), arrows+J/K move, N/M U/O L/; rotate, "
        "hold Space to pinch, release to open, \\ reset targets, Esc quit"
    )

    try:
        while running and world.scene.viewer.is_alive():
            world.scene.update_debug_objects(
                tuple(gizmos[side] for side in ("right", "left")),
                tuple(gu.trans_quat_to_T(*world.palm_target(side)) for side in ("right", "left")),
            )
            world.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")


def main() -> None:
    args = _parse_args()
    gs.init(precision="64", logging_level="info")

    from genesis.engine.couplers.qipc_coupler.tape_world import (
        TapeWorldConfig,
        build_qipc_tape_world,
    )

    world = build_qipc_tape_world(
        TapeWorldConfig(
            mode=args.mode,
            sticky_hands=args.sticky_hands,
            newton_velocity_tol=args.newton_tol,
            linear_tol_rate=args.linear_tol,
            linear_max_iter=args.linear_max_iter,
            show_viewer=args.headless_steps == 0,
            urdf_path=args.urdf,
        )
    )
    if args.headless_steps > 0:
        _run_headless(world, args)
    else:
        _run_interactive(world)


if __name__ == "__main__":
    main()
