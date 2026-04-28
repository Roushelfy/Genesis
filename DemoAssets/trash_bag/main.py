"""
Trash-bag plasticity-freeze demo.

Timeline  (dt = 0.01 s/frame):
  0 – 1 s  (frames   0 – 100): bag falls freely, hits ground, accumulates plastic folds.
  1 – 3 s  (frames 100 – 300): SoftPositionConstraint smoothly lifts the rim back up.
      3 s  (frame  300       ): freeze plasticity — theta_bar reset to rest angle,
                                bending stiffness raised to FROZEN_BENDING_STIFFNESS.
  3 – 4 s  (frames 300 – 400): rim still pinned; bag elastically springs back toward
                                its original shape under the raised bending stiffness.
  4 – 5 s  (frames 400 – 500): all pins released; bag falls with purely elastic bending.

Usage
-----
    python main.py
"""

from __future__ import annotations

import math
import os

import numpy as np

import polyscope as ps
from polyscope import imgui

from uipc import Logger, Engine, World, Scene, SceneIO, Animation, view
from uipc.geometry import SimplicialComplex, trimesh, ground, label_surface, mesh_partition
from uipc.constitution import (
    NeoHookeanShell,
    StrainPlasticDiscreteShellBending,
    StrainPlasticDiscreteShellBendingModifier,
    ElasticModuli2D,
    SoftPositionConstraint,
)
from uipc import builtin

# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------
BAG_RADIUS      = 0.20         # m
BAG_HEIGHT      = 0.40         # m
BAG_CIRC_SEGS   = 24           # circumference segments
BAG_HEIGHT_SEGS = 16           # height segments
BAG_THICKNESS   = 0.0015       # m

SHELL_DENSITY  = 150.0         # kg/m^3
SHELL_YOUNG    = 5.0e4         # Pa
SHELL_POISSON  = 0.35

# Yield threshold: angle (rad) beyond which plastic flow occurs.
# 0.0 = perfect plasticity (any bend leaves a crease).
# Large value = elastic only.
BENDING_STIFFNESS  = 5.0e1     # Pa*m  — soft, crumples easily
YIELD_THRESHOLD    = 0.5       # rad   — yields at ~23 deg bend
HARDENING_MODULUS  = 0.0       # perfect plasticity (no hardening)

# After freeze: theta_bar is reset to the original rest angle and
# bending stiffness is raised to this value.
NEW_BENDING_STIFFNESS = 8.0e3  # Pa*m

RIM_STRENGTH = 300.0           # SoftPositionConstraint spring strength

# ---------------------------------------------------------------------------
# Simulation schedule  (1 frame = dt = 0.01 s)
# ---------------------------------------------------------------------------
FALL_FRAMES   = 100   # frames   0 – 100  (0 – 1 s)  : free-fall, no pins
LIFT_START    = 100   # frame  100        (1 s)       : start smooth lift
FREEZE_FRAME  = 300   # frame  300        (3 s)       : freeze plasticity + stiffen
RELEASE_FRAME = 400   # frame  400        (4 s)       : release all pins
TOTAL_FRAMES  = 500   # frame  500        (5 s)       : end of demo

GROUND_Y    = -0.60   # ground plane y
BAG_START_Y =  0.0    # top rim starts at y=0; bottom at y = -BAG_HEIGHT


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------
def make_bag_mesh(radius: float, height: float,
                  circ: int, vstacks: int) -> SimplicialComplex:
    """Cylindrical bag: open top ring at BAG_START_Y, closed bottom cap."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for j in range(vstacks + 1):
        y = BAG_START_Y - j * height / vstacks
        for i in range(circ):
            theta = 2.0 * math.pi * i / circ
            vertices.append([radius * math.cos(theta), y, radius * math.sin(theta)])
    for j in range(vstacks):
        for i in range(circ):
            i_next = (i + 1) % circ
            v00 = j * circ + i;        v10 = j * circ + i_next
            v01 = (j + 1) * circ + i;  v11 = (j + 1) * circ + i_next
            triangles += [[v00, v11, v10], [v00, v01, v11]]
    bottom_center_idx = len(vertices)
    vertices.append([0.0, BAG_START_Y - height, 0.0])
    base = vstacks * circ
    for i in range(circ):
        i_next = (i + 1) % circ
        triangles.append([bottom_center_idx, base + i_next, base + i])
    sc = trimesh(np.array(vertices, dtype=np.float64),
                 np.array(triangles, dtype=np.int32))
    label_surface(sc)
    return sc


# ---------------------------------------------------------------------------
# Scene  (module-level — all objects kept alive to avoid GC-related crashes)
# ---------------------------------------------------------------------------
Logger.set_level(Logger.Level.Warn)

workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(workspace, exist_ok=True)

engine = Engine("cuda", workspace)
world  = World(engine)

cfg = Scene.default_config()
cfg["dt"]                            = 0.01
cfg["gravity"]                       = [[0.0], [-9.8], [0.0]]
cfg["contact"]["enable"]             = True
cfg["contact"]["friction"]["enable"] = False
cfg["line_search"]["max_iter"]       = 16
cfg["linear_system"]["tol_rate"]     = 1.0e-3
scene = Scene(cfg)

scene.contact_tabular().default_model(0.3, 1.0e9)
default_contact = scene.contact_tabular().default_element()

shell           = NeoHookeanShell()
plastic_bending = StrainPlasticDiscreteShellBending()
freeze_modifier = StrainPlasticDiscreteShellBendingModifier()  # zero-energy modifier
spc             = SoftPositionConstraint()

bag_mesh = make_bag_mesh(BAG_RADIUS, BAG_HEIGHT, BAG_CIRC_SEGS, BAG_HEIGHT_SEGS)
mesh_partition(bag_mesh)

moduli = ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON)
shell.apply_to(bag_mesh, moduli, SHELL_DENSITY, BAG_THICKNESS)
plastic_bending.apply_to(bag_mesh, BENDING_STIFFNESS, YIELD_THRESHOLD, HARDENING_MODULUS)
# Writes meta attrs "cancel_plastic"=0 and "target_bending_stiffness"=0.
# Animation sets these at frame FREEZE_FRAME to trigger the transition.
freeze_modifier.apply_to(bag_mesh)
spc.apply_to(bag_mesh, RIM_STRENGTH)
default_contact.apply_to(bag_mesh)

bag_object   = scene.objects().create("trash_bag")
bag_geo_slot = bag_object.geometries().create(bag_mesh)[0]

ground_object = scene.objects().create("ground")
ground_object.geometries().create(ground(GROUND_Y))

scene_io = SceneIO(scene)

# Rim vertex indices (ring j=0, the top opening of the bag)
rim_vertices = list(range(BAG_CIRC_SEGS))
rim_rest_pos = np.array(view(bag_mesh.positions()), copy=True).reshape(-1, 3)[rim_vertices]


# ---------------------------------------------------------------------------
# Animation callback
# ---------------------------------------------------------------------------
# Rim positions captured at the start of the lift (frame LIFT_START).
# Filled lazily on the first call with LIFT_START <= frame < FREEZE_FRAME.
lift_start_rim_pos = None


def animate_bag(info: Animation.UpdateInfo):
    global lift_start_rim_pos
    frame = info.frame()
    geo   = info.geo_slots()[0].geometry()

    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_pos        = view(geo.vertices().find(builtin.aim_position))
    is_constrained[:] = 0   # default: no pins this frame

    # 1 – 3 s: smoothly lift rim from landed position back to rest position.
    # Capture actual rim positions on the very first lift frame to avoid a
    # sudden gap between where the rim is and where we aim it.
    if LIFT_START <= frame < FREEZE_FRAME:
        if lift_start_rim_pos is None:
            all_pos = np.array(view(geo.positions()), copy=True).reshape(-1, 3)
            lift_start_rim_pos = all_pos[rim_vertices].copy()

        t = (frame - LIFT_START) / max(FREEZE_FRAME - LIFT_START - 1, 1)
        t = float(np.clip(t, 0.0, 1.0))
        t_s = t * t * (3.0 - 2.0 * t)   # smoothstep

        for k, v in enumerate(rim_vertices):
            is_constrained[v] = 1
            interp = (1.0 - t_s) * lift_start_rim_pos[k] + t_s * rim_rest_pos[k]
            aim_pos[v] = interp.reshape(3, 1)

    # 3 s: freeze plasticity.
    # Backend resets theta_bar to initial rest angle and raises bending stiffness.
    # Write per-edge attrs: set all edges to freeze + target stiffness.
    if frame == FREEZE_FRAME:
        cancel_plastic = geo.edges().find("cancel_plastic")
        target_bending_stiffness = geo.edges().find("target_bending_stiffness")
        if cancel_plastic is not None:
            view(cancel_plastic)[:] = 1
        if target_bending_stiffness is not None:
            view(target_bending_stiffness)[:] = NEW_BENDING_STIFFNESS

    # 3 – 4 s: rim still pinned at rest positions so the bag can elastically
    # spring back toward its original shape under the raised bending stiffness.
    if FREEZE_FRAME <= frame < RELEASE_FRAME:
        for k, v in enumerate(rim_vertices):
            is_constrained[v] = 1
            aim_pos[v] = rim_rest_pos[k].reshape(3, 1)

    # 4 – 5 s: all pins released; bag falls freely with purely elastic bending.


scene.animator().insert(bag_object, animate_bag)
world.init(scene)


# ---------------------------------------------------------------------------
# Polyscope visualisation
# ---------------------------------------------------------------------------
def phase_label(frame: int) -> str:
    if frame < LIFT_START:           # 0 – 1 s
        return "0-1s  falling + crumpling  [plastic]"
    if frame < FREEZE_FRAME:         # 1 – 3 s
        return "1-3s  lifting              [plastic]"
    if frame < RELEASE_FRAME:        # 3 – 4 s
        return "3-4s  elastic spring-back  [frozen]"
    return     "4-5s  free fall            [frozen / elastic]"


ps.init()
ps.set_ground_plane_mode("shadow_only")

surface = scene_io.simplicial_surface()
ps_mesh = ps.register_surface_mesh(
    "trash_bag",
    surface.positions().view().reshape(-1, 3),
    surface.triangles().topo().view().reshape(-1, 3),
)
ps_mesh.set_smooth_shade(True)

run              = False
freeze_triggered = False


def on_update():
    global run, freeze_triggered

    if imgui.Button("Run / Pause"):
        run = not run
    imgui.SameLine()
    if imgui.Button("Step"):
        do_advance()
    imgui.SameLine()
    # Manual freeze overrides the auto-trigger at frame FREEZE_FRAME
    if imgui.Button("Freeze now"):
        geo = bag_geo_slot.geometry()
        fa  = geo.edges().find("cancel_plastic")
        ts  = geo.edges().find("target_bending_stiffness")
        if fa is not None:
            view(fa)[:] = 1
        if ts is not None:
            view(ts)[:] = NEW_BENDING_STIFFNESS
        freeze_triggered = True

    if run:
        do_advance()

    frame = world.frame()
    imgui.Separator()
    imgui.Text("Trash-bag plasticity-freeze demo")
    imgui.Text(f"Frame : {frame} / {TOTAL_FRAMES}  ({frame * 0.01:.2f} s)")
    imgui.Text(f"Phase : {phase_label(min(frame, TOTAL_FRAMES))}")
    imgui.Separator()
    imgui.Text(f"Bending stiffness (plastic) : {BENDING_STIFFNESS:.0f} Pa*m")
    imgui.Text(f"Bending stiffness (frozen)  : {NEW_BENDING_STIFFNESS:.0f} Pa*m")
    imgui.Text(f"Yield threshold             : {YIELD_THRESHOLD:.3f} rad  (~{math.degrees(YIELD_THRESHOLD):.1f} deg)")
    imgui.Text(f"Auto-freeze at frame        : {FREEZE_FRAME}  ({FREEZE_FRAME * 0.01:.1f} s)")
    if frame >= FREEZE_FRAME or freeze_triggered:
        imgui.Text(">> Plasticity FROZEN — purely elastic bending <<")


def do_advance():
    if world.frame() >= TOTAL_FRAMES:
        return
    world.advance()
    if not world.is_valid():
        return
    world.retrieve()
    merged = scene_io.simplicial_surface()
    ps_mesh.update_vertex_positions(merged.positions().view().reshape(-1, 3))


ps.set_user_callback(on_update)
ps.show()
