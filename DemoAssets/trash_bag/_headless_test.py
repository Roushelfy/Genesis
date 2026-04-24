"""Headless test for trash_bag demo — runs N frames without polyscope."""
import sys, os, math, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uipc import Logger, Engine, World, Scene, SceneIO, Animation, view
from uipc.geometry import SimplicialComplex, trimesh, ground, label_surface
from uipc.constitution import (
    NeoHookeanShell,
    StrainPlasticDiscreteShellBending,
    StrainPlasticDiscreteShellBendingModifier,
    ElasticModuli2D,
    SoftPositionConstraint,
)
from uipc import builtin

BAG_RADIUS      = 0.20
BAG_HEIGHT      = 0.40
BAG_CIRC_SEGS   = 24
BAG_HEIGHT_SEGS = 16
BAG_THICKNESS   = 0.0015
SHELL_DENSITY   = 150.0
SHELL_YOUNG     = 5.0e4
SHELL_POISSON   = 0.35
BENDING_STIFFNESS        = 2.0e2
YIELD_THRESHOLD          = 0.08
HARDENING_MODULUS        = 0.0
FROZEN_BENDING_STIFFNESS = 8.0e3
RIM_STRENGTH    = 200.0
CRUMPLE_FRAMES  = 200
FREEZE_FRAME    = 200
STIFFEN_FRAMES  = 80
RELEASE_FRAMES  = 120
TOTAL_FRAMES    = CRUMPLE_FRAMES + STIFFEN_FRAMES + RELEASE_FRAMES
GROUND_Y        = -0.50

def make_bag_mesh(radius, height, circ, vstacks):
    vertices, triangles = [], []
    for j in range(vstacks + 1):
        y = -j * height / vstacks
        for i in range(circ):
            theta = 2.0 * math.pi * i / circ
            vertices.append([radius * math.cos(theta), y, radius * math.sin(theta)])
    for j in range(vstacks):
        for i in range(circ):
            i_next = (i + 1) % circ
            v00, v10 = j*circ+i, j*circ+i_next
            v01, v11 = (j+1)*circ+i, (j+1)*circ+i_next
            triangles += [[v00, v11, v10], [v00, v01, v11]]
    bottom_center_idx = len(vertices)
    vertices.append([0.0, -height, 0.0])
    bottom_ring_start = vstacks * circ
    for i in range(circ):
        i_next = (i + 1) % circ
        triangles.append([bottom_center_idx, bottom_ring_start+i_next, bottom_ring_start+i])
    sc = trimesh(np.array(vertices, dtype=np.float64), np.array(triangles, dtype=np.int32))
    label_surface(sc)
    return sc

Logger.set_level(Logger.Level.Warn)
workspace = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(workspace, exist_ok=True)

engine = Engine("cuda", workspace)
world  = World(engine)

cfg = Scene.default_config()
cfg["dt"]                           = 0.01
cfg["gravity"]                      = [[0.0], [-9.8], [0.0]]
cfg["contact"]["enable"]            = True
cfg["contact"]["friction"]["enable"] = False
cfg["line_search"]["max_iter"]      = 12
cfg["linear_system"]["tol_rate"]    = 1.0e-3
scene = Scene(cfg)

scene.contact_tabular().default_model(0.3, 1.0e9)
default_contact = scene.contact_tabular().default_element()

shell           = NeoHookeanShell()
plastic_bending = StrainPlasticDiscreteShellBending()
freeze_modifier = StrainPlasticDiscreteShellBendingModifier()
spc             = SoftPositionConstraint()

bag_mesh = make_bag_mesh(BAG_RADIUS, BAG_HEIGHT, BAG_CIRC_SEGS, BAG_HEIGHT_SEGS)
moduli = ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON)
shell.apply_to(bag_mesh, moduli, SHELL_DENSITY, BAG_THICKNESS)
plastic_bending.apply_to(bag_mesh, BENDING_STIFFNESS, YIELD_THRESHOLD, HARDENING_MODULUS)
freeze_modifier.apply_to(bag_mesh, new_bending_stiffness=FROZEN_BENDING_STIFFNESS)
spc.apply_to(bag_mesh, RIM_STRENGTH)
default_contact.apply_to(bag_mesh)

bag_object   = scene.objects().create("trash_bag")
bag_geo_slot = bag_object.geometries().create(bag_mesh)[0]

ground_object = scene.objects().create("ground")
ground_object.geometries().create(ground(GROUND_Y))

rim_vertices = list(range(BAG_CIRC_SEGS))
rim_rest_pos = np.array(view(bag_mesh.positions()), copy=True).reshape(-1, 3)[rim_vertices]

def animate_bag(info: Animation.UpdateInfo):
    frame = info.frame()
    geo   = info.geo_slots()[0].geometry()
    if frame == FREEZE_FRAME:
        fa = geo.meta().find("cancel_plastic")
        if fa is not None:
            view(fa)[0] = 1
    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_pos        = view(geo.vertices().find(builtin.aim_position))
    is_constrained[:] = 0
    if frame < FREEZE_FRAME + STIFFEN_FRAMES:
        for k, v in enumerate(rim_vertices):
            is_constrained[v] = 1
            aim_pos[v]        = rim_rest_pos[k].reshape(3, 1)

scene.animator().insert(bag_object, animate_bag)
world.init(scene)
print("[headless] init OK")

N = TOTAL_FRAMES
for i in range(1, N + 1):
    world.advance()
    if not world.is_valid():
        print(f"[headless] world INVALID at frame {i}")
        sys.exit(1)
    world.retrieve()
    if i % 25 == 0 or i == FREEZE_FRAME or i == FREEZE_FRAME + 1:
        print(f"[headless] frame {i}/{N} OK  phase={'crumpling' if i < FREEZE_FRAME else ('frozen' if i < FREEZE_FRAME+STIFFEN_FRAMES else 'released')}")

print("[headless] ALL FRAMES PASSED")
