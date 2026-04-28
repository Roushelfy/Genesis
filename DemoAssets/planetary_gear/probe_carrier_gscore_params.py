"""Genesis settle with EXACT gs_core IPCCouplerOptions for ipc_gear."""
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
    coupler_options=gs.options.IPCCouplerOptions(
        contact_d_hat=0.001,
        contact_resistance=1e7,
        newton_max_iterations=32,
        newton_tolerance=0.1,
        newton_semi_implicit_enable=True,
        linear_system_tolerance=1e-6,
        n_linesearch_iterations=10,
    ),
)

rigid_set_mat = gs.materials.Rigid(rho=1000.0, coup_friction=0.001)

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
    oa = math.radians(i * 360.0 / NUM_PLANETS + 270.0)
    sr = i * 360.0 * SUN_TEETH / PLANET_TEETH + 270.0
    tx, ty = orbit_r * math.cos(oa), orbit_r * math.sin(oa)
    if i == 0:
        px, py = CX, CY - 0.20
    else:
        px, py = CX + tx, CY + ty
    scene.add_entity(
        gs.morphs.Mesh(file=str(ASSETS/"planet_gear_v2.obj"),
                       pos=(px, py, CZ), euler=(0,0,sr), scale=MS,
                       fixed=False, convexify=False, decimate=False),
        material=rigid_set_mat)

sun = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"sun_gear_handle_v2.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MS, fixed=False,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

scene.add_entity(
    gs.morphs.Plane(normal=(0,0,1), visualization=False),
    material=gs.materials.Rigid(needs_coup=False))
scene.build(n_envs=0)

c0 = float(np.array(carrier.get_pos())[2])
print(f"gs_core params (resistance=1e7, semi_implicit=True, newton_tol=0.1):")
print(f"  frame  0: carrier COM Z = {c0:.6f}")
for step in range(20):
    scene.step()
    cz = float(np.array(carrier.get_pos())[2])
    print(f"  frame {step+1:2d}: carrier COM Z = {cz:.6f}")
print(f"  drop = {(c0 - cz)*1000:.3f} mm")
