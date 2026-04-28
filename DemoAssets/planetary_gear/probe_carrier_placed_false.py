"""Genesis settle with placed=False (planet_0 on table, matching gs_core)."""
import math
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
MS = 0.0012
CX, CY, CZ = 0.5, 0.0, 0.79

import genesis as gs
gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
    coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=2e-4 * 1.2),
)
rigid_set_mat = gs.materials.Rigid(rho=1000.0, coup_friction=0.001)
rigid_gear_mat = gs.materials.Rigid(rho=70.0, coup_friction=0.2)

scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"ring_gear.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MS, fixed=True,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

carrier_tz = -(12.0/2) * MS
carrier = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"carrier.obj"), pos=(CX,CY,CZ+carrier_tz),
                   euler=(0,0,270), scale=MS, fixed=False,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

pin_tz = -12.0 * MS
scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"support_pin.obj"), pos=(CX,CY,CZ+pin_tz),
                   scale=MS, fixed=True, convexify=False, decimate=False),
    material=rigid_set_mat)

SUN_TEETH, PLANET_TEETH, NUM_PLANETS = 12, 9, 3
orbit_r = 3 * (SUN_TEETH + PLANET_TEETH) / 2 * MS

for i in range(NUM_PLANETS):
    orbit_angle = math.radians(i * 360.0 / NUM_PLANETS + 270.0)
    self_rot = i * 360.0 * SUN_TEETH / PLANET_TEETH + 270.0
    tx = orbit_r * math.cos(orbit_angle)
    ty = orbit_r * math.sin(orbit_angle)

    if i == 0:
        # placed=False: planet_0 on table at Y=-0.20
        px, py, pz = CX, CY - 0.20, CZ
        mat = rigid_gear_mat
    else:
        px, py, pz = CX + tx, CY + ty, CZ
        mat = rigid_set_mat

    scene.add_entity(
        gs.morphs.Mesh(file=str(ASSETS/"planet_gear_v2.obj"),
                       pos=(px, py, pz), euler=(0,0,self_rot),
                       scale=MS, fixed=False, convexify=False, decimate=False),
        material=mat)

sun = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"sun_gear_handle_v2.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MS, fixed=False,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

scene.add_entity(gs.morphs.Box(size=(2,2,0.02), pos=(0,0,-0.01), fixed=True))

scene.build(n_envs=0)

c0 = float(np.array(carrier.get_pos())[2])
print(f"placed=False settle (planet_0 on table):")
print(f"  frame  0: carrier COM Z = {c0:.6f}")
for step in range(20):
    scene.step()
    cz = float(np.array(carrier.get_pos())[2])
    print(f"  frame {step+1:2d}: carrier COM Z = {cz:.6f}")
print(f"  drop = {c0 - cz:.6f} m = {(c0-cz)*1000:.3f} mm")
