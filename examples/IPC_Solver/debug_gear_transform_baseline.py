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
    args = parser.parse_args()

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

    # Sun gear — OBJ reference
    scene.add_entity(
        gs.morphs.Mesh(file=SUN_OBJ, pos=sun_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.7, 0.2, 1.0)),  # green
        vis_mode="collision",
    )

    # Sun gear — GLB test (--zup controls file_meshes_are_zup)
    scene.add_entity(
        gs.morphs.Mesh(file=SUN_GLB, pos=sun_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False,
                       file_meshes_are_zup=args.zup),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.8, 0.2, 0.2, 1.0)),  # red
        vis_mode="collision",
    )

    # Carrier — OBJ reference
    scene.add_entity(
        gs.morphs.Mesh(file=CARRIER_OBJ, pos=carrier_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.7, 0.7, 1.0)),  # cyan
        vis_mode="collision",
    )

    # Carrier — GLB
    scene.add_entity(
        gs.morphs.Mesh(file=CARRIER_GLB, pos=carrier_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False,
                       file_meshes_are_zup=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.8, 0.7, 0.1, 1.0)),  # yellow
        vis_mode="collision",
    )

    # Ring gear — OBJ reference
    scene.add_entity(
        gs.morphs.Mesh(file=RING_OBJ, pos=sun_pos, euler=EULER, scale=SCALE, fixed=True, convexify=False),
        material=rigid,
        surface=gs.surfaces.Metal(color=(0.2, 0.3, 0.9, 1.0)),  # blue
        vis_mode="collision",
    )

    # Ring gear — GLB (--zup controls file_meshes_are_zup)
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

        scene.add_entity(
            gs.morphs.Mesh(file=PLANET_OBJ, pos=planet_pos, euler=planet_euler, scale=SCALE,
                           fixed=True, convexify=False),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.9, 0.9, 0.9, 1.0)),  # white
            vis_mode="collision",
        )
        scene.add_entity(
            gs.morphs.Mesh(file=PLANET_GLB, pos=planet_pos, euler=planet_euler, scale=SCALE,
                           fixed=True, convexify=False, file_meshes_are_zup=args.zup),
            material=rigid,
            surface=gs.surfaces.Metal(color=(0.9, 0.5, 0.1, 1.0)),  # orange
            vis_mode="collision",
        )

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

    while True:
        scene.viewer.update()


if __name__ == "__main__":
    main()
