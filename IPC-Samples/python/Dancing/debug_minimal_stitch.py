"""Minimal stitch reproduction: 1 vertex (mesh A) + 1 triangle (mesh B) + 1 stitch.

Extracted from stitch pair: vertex_a=64 in kimono_inner_lower, triangle_b=5020 in kimono_inner_upper.

Usage::

    uv run python debug_minimal_stitch.py
"""

from __future__ import annotations

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Logger, Timer, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli,
    ElasticModuli2D,
    SoftVertexTriangleStitch,
    Empty,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplex, label_surface, mesh_partition, trimesh
from uipc.gui import SceneGUI
from uipc.unit import GPa

if True:
    # Mesh A: single vertex (from kimono_inner_lower vertex 64)
    VERT_A = np.array([0.0704928667088553, -0.10671846963521708, -0.10867820298907753])

    # Mesh B: single triangle (from kimono_inner_upper triangle 5020, vertices 1314/1313/2399)
    TRI_V0 = np.array([0.06665994189065337, -0.10524519402358158, -0.10070570154561433])
    TRI_V2 = np.array([0.06896152583818653, -0.11308983327700026, -0.10648632100044506])
    TRI_V1 = np.array([0.06977603661402008, -0.10662828364002827, -0.10969111719664178])
else:
    VERT_A = np.array([0.0, 1.0, 0.0])

    TRI_V0 = np.array([-1.0, 0.0, 0.0])
    TRI_V1 = np.array([1.0, 0.0, 0.0])
    TRI_V2 = np.array([0.0, 0.0, 1.0])

OUTPUT_DIR = "IPC-Samples/output/python/Dancing/debug_minimal_stitch"


def main() -> None:
    from pathlib import Path

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()
    engine = Engine("cuda", str(out))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 0.5
    config["linear_system"]["tol_rate"] = 1e-4
    config["newton"]["max_iter"] = 256
    config["extras"]["debug"]["dump_surface"] = True
    scene = Scene(config)

    # --- Mesh A: a tiny triangle patch around the stitch vertex ---
    # Create a small triangle with the stitch vertex at index 0
    offset = 0.02
    a_verts = np.array([
        VERT_A,
        VERT_A + [offset, 0, 0],
        VERT_A + [0, offset, 0],
    ], dtype=np.float64)
    a_faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh_a = trimesh(a_verts, a_faces)
    label_surface(mesh_a)
    empty = Empty()
    empty.apply_to(mesh_a)
    rest_a = mesh_a.copy()
    obj_a = scene.objects().create("mesh_a")
    slot_a, rest_slot_a = obj_a.geometries().create(mesh_a, rest_a)

    # --- Mesh B: the single triangle ---
    b_verts = np.array([TRI_V0, TRI_V1, TRI_V2], dtype=np.float64)
    b_faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh_b = trimesh(b_verts, b_faces)
    label_surface(mesh_b)
    empty = Empty()
    empty.apply_to(mesh_b)
    rest_b = mesh_b.copy()
    obj_b = scene.objects().create("mesh_b")
    slot_b, rest_slot_b = obj_b.geometries().create(mesh_b, rest_b)

    # --- Stitch: vertex 0 of mesh_a -> triangle 0 of mesh_b ---
    svts = SoftVertexTriangleStitch()
    stitch_obj = scene.objects().create("stitch")
    pairs = np.array([[0, 0]], dtype=np.int32)
    stitch_geo = svts.create_geometry(
        (slot_a, slot_b),
        (rest_slot_a, rest_slot_b),
        pairs,
        ElasticModuli.youngs_poisson(1e4, 0.498),
        min_separate_distance=0.0001,
    )
    stitch_obj.geometries().create(stitch_geo)
    print(f"[stitch] 1 pair: vertex 0 of mesh_a -> triangle 0 of mesh_b")
    print(f"[stitch] distance: {np.linalg.norm(VERT_A - (TRI_V0+TRI_V1+TRI_V2)/3):.6f}")

    world.init(scene)
    world.retrieve()

    # --- GUI ---
    ps.init()
    ps.set_up_dir("z_up")
    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1.0)

    def update_stitch_line():
        pa = np.asarray(view(slot_a.geometry().positions()), copy=True).squeeze()
        pb = np.asarray(view(slot_b.geometry().positions()), copy=True).squeeze()
        pt = pa[0]
        centroid = pb.mean(axis=0)
        nodes = np.array([pt, centroid])
        edges = np.array([[0, 1]], dtype=np.int32)
        ps.register_curve_network("stitch_line", nodes, edges, radius=0.0005)

    update_stitch_line()
    state = {"run": False, "steps_per_tick": 1, "frame": 0}

    def on_update():
        if imgui.Button("Play / Pause"):
            state["run"] = not state["run"]
        imgui.SameLine()
        if imgui.Button("Step Once"):
            world.advance()
            world.retrieve()
            state["frame"] = world.frame()
            sgui.update()
            update_stitch_line()

        changed, spt = imgui.SliderInt("Speed", state["steps_per_tick"], 1, 8)
        if changed:
            state["steps_per_tick"] = int(max(1, spt))

        pa = np.asarray(view(slot_a.geometry().positions()), copy=True).squeeze()
        pb = np.asarray(view(slot_b.geometry().positions()), copy=True).squeeze()
        imgui.Text(f"Frame: {state['frame']}")
        imgui.Text(f"Vert A[0]: {pa[0]}")
        imgui.Text(f"Tri B centroid: {pb.mean(axis=0)}")
        imgui.Text(f"Distance: {np.linalg.norm(pa[0] - pb.mean(axis=0)):.6f}")

        if state["run"]:
            for _ in range(state["steps_per_tick"]):
                world.advance()
                world.retrieve()
                state["frame"] = world.frame()
                state["run"] = False
            sgui.update()
            update_stitch_line()

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
