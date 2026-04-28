"""Run Genesis gear scene using IPCGearEnvConfig setup and print carrier Z."""
import sys, math
import numpy as np
from pathlib import Path

# Add gs_core to path
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "gs_core_IPC_demo" / "env" / "sim" / "src"))
sys.path.insert(0, str(_REPO / "gs_core_IPC_demo" / "env" / "schemas" / "src"))

ASSETS = Path(__file__).resolve().parent / "assets"
MS = 0.0012
CX, CY, CZ = 0.5, 0.0, 0.79

import genesis as gs
gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
    coupler_options=gs.options.IPCCouplerOptions(
        contact_d_hat=2e-4 * 1.2,
    ),
)

# Match IPCGearEnvConfig exactly
rigid_set_mat = gs.materials.Rigid(rho=1000.0, coup_friction=0.001)

# Ring gear (fixed)
scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"ring_gear.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MS, fixed=True,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

# Carrier (free)
carrier_tz = -(12.0/2) * MS
carrier = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"carrier.obj"), pos=(CX,CY,CZ+carrier_tz),
                   euler=(0,0,270), scale=MS, fixed=False,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

# Support pin (fixed)
pin_tz = -12.0 * MS
scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"support_pin.obj"), pos=(CX,CY,CZ+pin_tz),
                   scale=MS, fixed=True, convexify=False, decimate=False),
    material=rigid_set_mat)

# Planet gears (pre-placed, free)
SUN_TEETH, PLANET_TEETH, NUM_PLANETS = 12, 9, 3
orbit_r = 3 * (SUN_TEETH + PLANET_TEETH) / 2 * MS
planets = []
for i in range(NUM_PLANETS):
    orbit_angle = math.radians(i * 360.0 / NUM_PLANETS + 270.0)
    self_rot = i * 360.0 * SUN_TEETH / PLANET_TEETH + 270.0
    tx = orbit_r * math.cos(orbit_angle)
    ty = orbit_r * math.sin(orbit_angle)
    mat = gs.materials.Rigid(rho=1000.0, coup_friction=0.001) if i > 0 else \
          gs.materials.Rigid(rho=1000.0, coup_friction=0.001)
    p = scene.add_entity(
        gs.morphs.Mesh(file=str(ASSETS/"planet_gear_v2.obj"),
                       pos=(CX+tx, CY+ty, CZ), euler=(0,0,self_rot),
                       scale=MS, fixed=False, convexify=False, decimate=False),
        material=mat)
    planets.append(p)

# Sun gear (free)
sun = scene.add_entity(
    gs.morphs.Mesh(file=str(ASSETS/"sun_gear_handle_v2.obj"), pos=(CX,CY,CZ),
                   euler=(0,0,270), scale=MS, fixed=False,
                   convexify=False, decimate=False),
    material=rigid_set_mat)

# Ground
scene.add_entity(gs.morphs.Box(size=(2,2,0.02), pos=(0,0,-0.01), fixed=True))

scene.build(n_envs=0)

print("Genesis carrier settle (full gear assembly, OBJ meshes, d_hat=2.4e-4):")
c_pos = np.array(carrier.get_pos())
s_pos = np.array(sun.get_pos())
print(f"  frame  0: carrier Z={c_pos[2]:.6f}  sun Z={s_pos[2]:.6f} (initial/COM)")

for step in range(20):
    scene.step()
    c_pos = np.array(carrier.get_pos())
    s_pos = np.array(sun.get_pos())
    print(f"  frame {step+1:2d}: carrier Z={c_pos[2]:.6f}  sun Z={s_pos[2]:.6f}")
