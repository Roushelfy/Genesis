"""
Minimal debug script: compare gear OBJ vs GLB collision meshes.

Loads sun_gear_handle, carrier, and ring_gear (OBJ + GLB each) at the same
positions as the replay scripts, using vis_mode="collision".
If each OBJ/GLB pair overlaps, that entity's transforms are correct.

Colors:
    green   = sun_gear_handle OBJ  (reference)
    red     = sun_gear_handle GLB  (--zup controls file_meshes_are_zup)
    cyan    = carrier OBJ          (reference)
    yellow  = carrier GLB          (file_meshes_are_zup=False)
    blue    = ring_gear OBJ        (reference)
    magenta = ring_gear GLB        (--zup controls file_meshes_are_zup)
    white   = planet_gear OBJ x3  (reference)
    orange  = planet_gear GLB x3  (--zup controls file_meshes_are_zup)

Usage
-----
    python examples/IPC_Solver/debug_gear_transform.py
    python examples/IPC_Solver/debug_gear_transform.py --zup
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_ASSETS = _REPO / "DemoAssets" / "planetary_gear" / "assets"

SUN_OBJ = str(_ASSETS / "sun_gear_handle.obj")
SUN_GLB = str(_ASSETS / "sun_gear_handle_debug.glb")
CARRIER_OBJ = str(_ASSETS / "carrier.obj")
CARRIER_GLB = str(_ASSETS / "carrier.glb")
RING_OBJ = str(_ASSETS / "ring_gear.obj")
RING_GLB = str(_ASSETS / "ring_gear.glb")
PLANET_OBJ = str(_ASSETS / "planet_gear.obj")
PLANET_GLB = str(_ASSETS / "planet_gear.glb")

# Same transforms as replay_gear_traj.py
CX, CY, CZ = 0.5, 0.0, 0.79   # mechanism center — matches replay script
EULER = (0, 0, 270)
SCALE = 0.0012
GEAR_WIDTH_MM = 12.0
CARRIER_TZ = -(GEAR_WIDTH_MM / 2) * SCALE  # matches carrier offset in replay scripts

MODUL = 3
SUN_TEETH = 12
PLANET_TEETH = 9
NUM_PLANETS = 3
ORBIT_R = MODUL * (SUN_TEETH + PLANET_TEETH) / 2 * SCALE
GEAR_ROTATION_DEG = 270.0


def _planet_position(index: int) -> tuple[float, float, float]:
    orbit_angle = math.radians(index * 360.0 / NUM_PLANETS + GEAR_ROTATION_DEG)
    self_rot_deg = index * 360.0 * SUN_TEETH / PLANET_TEETH + GEAR_ROTATION_DEG
    tx = ORBIT_R * math.cos(orbit_angle)
    ty = ORBIT_R * math.sin(orbit_angle)
    return tx, ty, self_rot_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OBJ vs GLB gear collision meshes.")
    parser.add_argument(
        "--zup",
        action="store_true",
        help="Set file_meshes_are_zup=True on the sun gear GLB (default: False)",
    )
    parser.add_argument(
        "--traj",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to trajectory .npz — enables LEFT/RIGHT arrow frame stepping",
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Skip adding GLB entities — show OBJ meshes only",
    )
    args = parser.parse_args()
    show_glb = not args.no_glb

    if not Path(SUN_GLB).exists():
        sys.exit(f"GLB not found: {SUN_GLB}\nExport sun_gear_handle from Blender and save it there.")

    import genesis as gs

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=True,
        vis_options=gs.options.VisOptions(show_world_frame=True),
    )

    rigid = gs.materials.Rigid(rho=1000.0)
    sun_pos = (CX, CY, CZ)
    carrier_pos = (CX, CY, CZ + CARRIER_TZ)

    # Dynamic entity refs for trajectory stepping: name -> [obj_ent, glb_ent]
    rigid_entities: dict[str, list] = {}

    # Sun gear — OBJ reference
    _e = scene.add_entity(
        gs.morphs.Mesh(file=SUN_OBJ, pos=sun_pos, euler=EULER, scale=SCALE, fixed=False, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.7, 0.2, 1.0)),  # green
        vis_mode="collision",
    )
    rigid_entities["sun_gear"] = [_e]

    # Sun gear — GLB test (--zup controls file_meshes_are_zup)
    if show_glb:
        _e = scene.add_entity(
            gs.morphs.Mesh(file=SUN_GLB, pos=sun_pos, euler=EULER, scale=SCALE, fixed=False, convexify=False,
                           file_meshes_are_zup=args.zup),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.8, 0.2, 0.2, 1.0)),  # red
            vis_mode="collision",
        )
        # rigid_entities["sun_gear"].append(_e)  # GLB — not driven by trajectory

    # Carrier — OBJ reference
    _e = scene.add_entity(
        gs.morphs.Mesh(file=CARRIER_OBJ, pos=carrier_pos, euler=EULER, scale=SCALE, fixed=False, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.7, 0.7, 1.0)),  # cyan
        vis_mode="collision",
    )
    rigid_entities["carrier"] = [_e]

    # Carrier — GLB
    if show_glb:
        _e = scene.add_entity(
            gs.morphs.Mesh(file=CARRIER_GLB, pos=carrier_pos, euler=EULER, scale=SCALE, fixed=False, convexify=False,
                           file_meshes_are_zup=False),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.8, 0.7, 0.1, 1.0)),  # yellow
            vis_mode="collision",
        )
        # rigid_entities["carrier"].append(_e)  # GLB — not driven by trajectory

    # Ring gear — OBJ reference (fixed, not driven by trajectory)
    scene.add_entity(
        gs.morphs.Mesh(file=RING_OBJ, pos=sun_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.3, 0.9, 1.0)),  # blue
        vis_mode="collision",
    )

    # Ring gear — GLB (fixed, not driven by trajectory)
    if show_glb:
        scene.add_entity(
            gs.morphs.Mesh(file=RING_GLB, pos=sun_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False,
                           file_meshes_are_zup=True),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.9, 0.2, 0.8, 1.0)),  # magenta
            vis_mode="collision",
        )

    # Planet gears — 3 each
    for i in range(NUM_PLANETS):
        tx, ty, self_rot_deg = _planet_position(i)
        planet_pos = (CX + tx, CY + ty, CZ)
        planet_euler = (0, 0, self_rot_deg)

        _e = scene.add_entity(
            gs.morphs.Mesh(file=PLANET_OBJ, pos=planet_pos, euler=planet_euler, scale=SCALE,
                           fixed=False, convexify=False),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.9, 0.9, 0.9, 1.0)),  # white
            vis_mode="collision",
        )
        rigid_entities[f"planet_gear_{i}"] = [_e]

        if show_glb:
            _e = scene.add_entity(
                gs.morphs.Mesh(file=PLANET_GLB, pos=planet_pos, euler=planet_euler, scale=SCALE,
                               fixed=False, convexify=False, file_meshes_are_zup=args.zup),
                material=rigid,
                surface=gs.surfaces.Metal(color=(0.9, 0.5, 0.1, 1.0)),  # orange
                vis_mode="collision",
            )
            # rigid_entities[f"planet_gear_{i}"].append(_e)  # GLB — not driven by trajectory

    print(f"\nSun gear   OBJ (green)   : {SUN_OBJ}")
    print(f"Sun gear   GLB (red)     : {SUN_GLB}  file_meshes_are_zup={args.zup}")
    print(f"Carrier    OBJ (cyan)    : {CARRIER_OBJ}")
    print(f"Carrier    GLB (yellow)  : {CARRIER_GLB}  file_meshes_are_zup=False")
    print(f"Ring gear  OBJ (blue)    : {RING_OBJ}")
    print(f"Ring gear  GLB (magenta) : {RING_GLB}  file_meshes_are_zup=True")
    print(f"Planets    OBJ (white)   : {PLANET_OBJ}  x{NUM_PLANETS}")
    print(f"Planets    GLB (orange)  : {PLANET_GLB}  file_meshes_are_zup={args.zup}  x{NUM_PLANETS}")
    print(f"\npos=({CX},{CY},{CZ})  euler={EULER}  scale={SCALE}  carrier_tz={CARRIER_TZ:.5f}")
    print("Each OBJ+GLB pair should overlap.\n")

    scene.build()

    # ── Optional trajectory stepping ─────────────────────────────────────────
    if args.traj is None:
        print("\n[traj] No --traj file given — static view only. Pass --traj PATH to enable frame stepping.\n")
    if args.traj:
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        print(f"[traj] Loading {args.traj} …")
        traj = np.load(args.traj)
        rigid_data = {
            "sun_gear":      traj["rigid_sun_gear"],
            "carrier":       traj["rigid_carrier"],
            "planet_gear_0": traj["rigid_planet_gear_0"],
            "planet_gear_1": traj["rigid_planet_gear_1"],
            "planet_gear_2": traj["rigid_planet_gear_2"],
        }
        n_frames = len(traj["sim_time"])
        _frame = [0]

        def _apply_frame(frame_idx: int) -> None:
            for name, ents in rigid_entities.items():
                if name in rigid_data and frame_idx < len(rigid_data[name]):
                    pose = rigid_data[name][frame_idx]
                    pos  = pose[:3]
                    quat = pose[3:]
                    print(f"  {name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    for ent in ents:
                        ent.set_pos(pos)
                        ent.set_quat(quat)
            scene._visualizer.update_visual_states(force_render=True)
            scene._visualizer.update(force=True)

        def _on_next_frame() -> None:
            _frame[0] = min(_frame[0] + 1, n_frames - 1)
            _apply_frame(_frame[0])
            print(f"[traj] frame {_frame[0]}/{n_frames - 1}")

        def _on_prev_frame() -> None:
            _frame[0] = max(_frame[0] - 1, 0)
            _apply_frame(_frame[0])
            print(f"[traj] frame {_frame[0]}/{n_frames - 1}")

        scene.viewer.register_keybinds(
            Keybind("next_frame", Key.RIGHT, KeyAction.RELEASE, callback=_on_next_frame),
            Keybind("prev_frame", Key.LEFT,  KeyAction.RELEASE, callback=_on_prev_frame),
        )
        print(f"\nLoaded {n_frames} frames from {args.traj}")
        print("LEFT / RIGHT arrows to step one frame at a time.\n")

    while True:
        scene._visualizer.update(force=True)


if __name__ == "__main__":
    main()
