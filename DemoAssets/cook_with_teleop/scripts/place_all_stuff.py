"""Placement + simulation tool for cooking scene.

Two distinct phases:
  Phase 1 — PLACEMENT (pure polyscope, no UIPC):
    Adjust positions of pan, spatula, broccoli (N instances), noodle grid
    via imgui sliders with instant visual feedback.

  Phase 2 — SIMULATION (full UIPC scene with constitutions):
    Click "Build & Run" to construct the UIPC scene from placed positions.
    Pan/spatula are fixed.  Broccoli and noodles are dynamic under gravity.
    Noodle vertex positions and broccoli positions are recorded each frame
    as maps.  Click "Record & Save" to write everything to JSON.

Usage:
    python place_all_stuff.py [--traj PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Logger, Timer, Transform, builtin, view
from uipc import Engine, Scene, SceneIO, World
from uipc.constitution import (
    AffineBodyShell,
    ElasticModuli,
    HookeanSpring,
    KirchhoffRodBending,
    StableNeoHookean,
)
from uipc.geometry import (
    SimplicialComplexIO,
    flip_inward_triangles,
    ground,
    label_surface,
    label_triangle_orient,
    linemesh,
    mesh_partition,
    tetmesh,
    trimesh,
)
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

from asset_dir import AssetDir
from usd_mesh_loader import load_usd_mesh

_HERE = Path(__file__).resolve().parent
_ASSET_ROOT = _HERE.parent
_COOK_ROOT = _ASSET_ROOT.parent / "cook"

PAN_USD = _COOK_ROOT / "Pan025" / "Pan025.usd"
SPATULA_USD = _COOK_ROOT / "Spatula018" / "Spatula018.usd"
BROCCOLI_OBJ = _ASSET_ROOT / "broccoli.obj"
TOMATO_OBJ = _ASSET_ROOT / "tomato_slice.obj"
TOMATO_NPZ = _ASSET_ROOT / "tomato_slice.npz"
DEFAULT_TRAJ = _COOK_ROOT / "trajectories" / "cooking_demo.json"
DEFAULT_OUTPUT = _ASSET_ROOT / "placement.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quat_pos_to_4x4(quat, pos):
    """[w, x, y, z] quaternion + [x, y, z] position -> 4x4 homogeneous matrix."""
    w, x, y, z = quat
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1 - 2 * (y * y + z * z)
    M[0, 1] = 2 * (x * y - w * z)
    M[0, 2] = 2 * (x * z + w * y)
    M[1, 0] = 2 * (x * y + w * z)
    M[1, 1] = 1 - 2 * (x * x + z * z)
    M[1, 2] = 2 * (y * z - w * x)
    M[2, 0] = 2 * (x * z - w * y)
    M[2, 1] = 2 * (y * z + w * x)
    M[2, 2] = 1 - 2 * (x * x + y * y)
    M[0, 3] = pos[0]
    M[1, 3] = pos[1]
    M[2, 3] = pos[2]
    return M


def _scale_pos_4x4(scale, pos):
    """Uniform scale + translation as 4x4 homogeneous matrix."""
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = M[1, 1] = M[2, 2] = scale
    M[0, 3], M[1, 3], M[2, 3] = pos[0], pos[1], pos[2]
    return M


def _apply_4x4(verts, M):
    """Apply 4x4 homogeneous transform to Nx3 vertices."""
    V4 = np.ones((len(verts), 4), dtype=np.float64)
    V4[:, :3] = verts
    return (V4 @ M.T)[:, :3]


def load_obj_numpy(path):
    """Minimal OBJ loader -> (verts [N,3], faces [M,3])."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def generate_noodle_curves(center, nx, ny, spacing, n_edges, length):
    """Vertical noodle curves (Z-up) for polyscope preview."""
    nv_per = n_edges + 1
    all_v, all_e = [], []
    offset = 0
    x_half = (nx - 1) * spacing / 2.0
    y_half = (ny - 1) * spacing / 2.0
    rng = np.random.default_rng(42)
    perturb = spacing * 0.15
    for ix in range(nx):
        for iy in range(ny):
            bx = center[0] + ix * spacing - x_half
            by = center[1] + iy * spacing - y_half
            for i in range(nv_per):
                dx = rng.uniform(-perturb, perturb)
                dy = rng.uniform(-perturb, perturb)
                all_v.append([bx + dx, by + dy, center[2] + i * (length / n_edges)])
            for i in range(n_edges):
                all_e.append([offset + i, offset + i + 1])
            offset += nv_per
    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32)
    return np.array(all_v, dtype=np.float64), np.array(all_e, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Placement + simulation tool")
    parser.add_argument("--traj", type=str, default=str(DEFAULT_TRAJ))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_path = args.output

    # ---- trajectory frame 0 ----
    with open(args.traj) as f:
        traj = json.load(f)
    frame0 = traj["frames"][0]
    pan_quat = frame0["pan"]["quat"]
    spatula_quat = frame0["spatula"]["quat"]

    # ---- pre-load mesh data for placement preview ----
    print("[place] Loading meshes for preview ...")
    pan_verts_raw, pan_faces = load_usd_mesh(PAN_USD)
    spatula_verts_raw, spatula_faces = load_usd_mesh(SPATULA_USD)
    broc_verts_raw, broc_faces = load_obj_numpy(BROCCOLI_OBJ)
    tomato_verts_raw, tomato_faces = load_obj_numpy(TOMATO_OBJ)
    tomato_tet = np.load(TOMATO_NPZ)
    tomato_tet_Vs = tomato_tet["vertices"].astype(np.float64)
    tomato_tet_Ts = tomato_tet["tetrahedra"].astype(np.int32)
    print("[place] Meshes ready.")

    # ---- editable state ----
    pan_pos = list(frame0["pan"]["pos"])
    spatula_pos = list(frame0["spatula"]["pos"])
    spatula_pos[2] += 0.5

    broc_list: list[dict] = [
        {"pos": [pan_pos[0] + 0.04, pan_pos[1] - 0.06, pan_pos[2] + 0.02],
         "scale": 0.004},
    ]

    tomato_list: list[dict] = [
        {"pos": [pan_pos[0] - 0.02, pan_pos[1] + 0.02, pan_pos[2] + 0.03],
         "scale": 1.0},
    ]

    noodle_center = [pan_pos[0], pan_pos[1], pan_pos[2] + 0.03]
    noodle_nx = [8]
    noodle_ny = [8]
    noodle_spacing = [0.006]
    noodle_n_edges = [50]
    noodle_length = [0.2]
    noodle_thickness = [0.001]
    noodle_bending = [1e4]
    ground_z = [0.7]

    # ---- load saved placement ----
    if Path(output_path).exists():
        try:
            with open(output_path) as f:
                saved = json.load(f)
            if "pan" in saved:
                pan_pos[:] = saved["pan"]["pos"]
            if "spatula" in saved:
                spatula_pos[:] = saved["spatula"]["pos"]
            if "broccoli" in saved:
                bd = saved["broccoli"]
                if isinstance(bd, dict) and "pos" in bd:
                    broc_list[:] = [bd]
                elif isinstance(bd, dict):
                    broc_list[:] = list(bd.values())
                elif isinstance(bd, list):
                    broc_list[:] = bd
            if "tomato" in saved:
                td = saved["tomato"]
                if isinstance(td, dict) and "pos" in td:
                    tomato_list[:] = [td]
                elif isinstance(td, dict):
                    tomato_list[:] = list(td.values())
                elif isinstance(td, list):
                    tomato_list[:] = td
            if "noodles" in saved:
                ns = saved["noodles"]
                noodle_center[:] = ns.get("center", noodle_center)
                noodle_nx[0] = ns.get("grid_nx", noodle_nx[0])
                noodle_ny[0] = ns.get("grid_ny", noodle_ny[0])
                noodle_spacing[0] = ns.get("spacing", noodle_spacing[0])
                noodle_n_edges[0] = ns.get("n_edges", noodle_n_edges[0])
                noodle_length[0] = ns.get("length", noodle_length[0])
                noodle_thickness[0] = ns.get("thickness", noodle_thickness[0])
                noodle_bending[0] = ns.get("bending_stiffness", noodle_bending[0])
            if "ground_height" in saved:
                ground_z[0] = saved["ground_height"]
            print(f"[place] Loaded placement from {output_path}")
        except Exception as e:
            print(f"[place] Could not load {output_path}: {e}")

    # ---- mode state ----
    mode = ["placement"]  # "placement" | "simulation"
    sim_running = [False]
    engine_ref = [None]
    scene_ref = [None]
    world_ref = [None]
    sgui_ref = [None]
    noodles_obj_ref = [None]
    broc_objs_ref: list[list] = [[]]
    tomato_objs_ref: list[list] = [[]]

    # ================================================================
    # Phase 1 — Placement preview (pure polyscope)
    # ================================================================
    def _show_placement():
        ps.remove_all_structures()
        v = _apply_4x4(pan_verts_raw, quat_pos_to_4x4(pan_quat, pan_pos))
        ps.register_surface_mesh("pan", v, pan_faces,
                                 color=(0.6, 0.6, 0.65), smooth_shade=True)
        v = _apply_4x4(spatula_verts_raw, quat_pos_to_4x4(spatula_quat, spatula_pos))
        ps.register_surface_mesh("spatula", v, spatula_faces,
                                 color=(0.5, 0.5, 0.55), smooth_shade=True)
        for bi, bd in enumerate(broc_list):
            v = _apply_4x4(broc_verts_raw, _scale_pos_4x4(bd["scale"], bd["pos"]))
            ps.register_surface_mesh(f"broc_{bi}", v, broc_faces,
                                     color=(0.2, 0.6, 0.15), smooth_shade=True)
        for ti, td in enumerate(tomato_list):
            v = _apply_4x4(tomato_verts_raw,
                           _scale_pos_4x4(td["scale"], td["pos"]))
            ps.register_surface_mesh(f"tomato_{ti}", v, tomato_faces,
                                     color=(0.85, 0.15, 0.1), smooth_shade=True)
        nv, ne = generate_noodle_curves(
            noodle_center, noodle_nx[0], noodle_ny[0],
            noodle_spacing[0], noodle_n_edges[0], noodle_length[0])
        if len(nv) > 0:
            cn = ps.register_curve_network("noodles", nv, ne)
            cn.set_radius(noodle_thickness[0] * 3, relative=False)
            cn.set_color((0.92, 0.86, 0.55))
        h = ground_z[0]
        sz = 2.0
        gv = np.array([[-sz, -sz, h], [sz, -sz, h], [sz, sz, h], [-sz, sz, h]])
        gf = np.array([[0, 1, 2], [0, 2, 3]])
        gm = ps.register_surface_mesh("ground", gv, gf, color=(0.85, 0.85, 0.8))
        gm.set_transparency(0.4)

    def _update_pan():
        v = _apply_4x4(pan_verts_raw, quat_pos_to_4x4(pan_quat, pan_pos))
        ps.register_surface_mesh("pan", v, pan_faces,
                                 color=(0.6, 0.6, 0.65), smooth_shade=True)

    def _update_spatula():
        v = _apply_4x4(spatula_verts_raw, quat_pos_to_4x4(spatula_quat, spatula_pos))
        ps.register_surface_mesh("spatula", v, spatula_faces,
                                 color=(0.5, 0.5, 0.55), smooth_shade=True)

    def _update_broccoli():
        for bi, bd in enumerate(broc_list):
            v = _apply_4x4(broc_verts_raw, _scale_pos_4x4(bd["scale"], bd["pos"]))
            ps.register_surface_mesh(f"broc_{bi}", v, broc_faces,
                                     color=(0.2, 0.6, 0.15), smooth_shade=True)

    def _update_tomato():
        for ti, td in enumerate(tomato_list):
            v = _apply_4x4(tomato_verts_raw,
                           _scale_pos_4x4(td["scale"], td["pos"]))
            ps.register_surface_mesh(f"tomato_{ti}", v, tomato_faces,
                                     color=(0.85, 0.15, 0.1), smooth_shade=True)

    def _update_noodles():
        nv, ne = generate_noodle_curves(
            noodle_center, noodle_nx[0], noodle_ny[0],
            noodle_spacing[0], noodle_n_edges[0], noodle_length[0])
        if len(nv) > 0:
            cn = ps.register_curve_network("noodles", nv, ne)
            cn.set_radius(noodle_thickness[0] * 3, relative=False)
            cn.set_color((0.92, 0.86, 0.55))

    # ================================================================
    # Phase 2 — Build UIPC simulation scene
    # ================================================================
    def _build_and_run():
        ps.remove_all_structures()

        Logger.set_level(Logger.Level.Info)
        Timer.enable_all()

        workspace = AssetDir.output_path(__file__)
        engine = Engine("cuda", workspace)
        w = World(engine)

        config = Scene.default_config()
        config["dt"] = 0.002
        config["gravity"] = [[0.0], [0.0], [-9.8]]
        config["contact"]["enable"] = True
        config["contact"]["friction"]["enable"] = False
        config["contact"]["d_hat"] = 0.0005
        config["newton"]["semi_implicit"] = True
        config["newton"]["velocity_tol"] = 1
        config["newton"]["transrate_tol"] = 10
        config["linear_system"]["tol_rate"] = 1e-4

        scene = Scene(config)
        scene.animator().substep(1)

        ab_shell = AffineBodyShell()
        snh = StableNeoHookean()
        hs = HookeanSpring()
        krb = KirchhoffRodBending()

        tabular = scene.contact_tabular()
        tabular.default_model(0.01, 1.0 * GPa)
        default_elem = tabular.default_element()
        noodle_elem = tabular.create("noodle")
        tomato_elem = tabular.create("tomato")

        # ---- Pan (fixed) ----
        pan_mesh = trimesh(*load_usd_mesh(PAN_USD))
        label_surface(pan_mesh)
        pan_mesh.triangles().create(builtin.orient, 1)
        ab_shell.apply_to(pan_mesh, 100.0 * MPa, thickness=0.0001)
        default_elem.apply_to(pan_mesh)
        view(pan_mesh.transforms())[0] = quat_pos_to_4x4(pan_quat, pan_pos)
        view(pan_mesh.instances().find(builtin.is_fixed))[0] = 1
        pan_obj = scene.objects().create("pan")
        pan_obj.geometries().create(pan_mesh)

        # ---- Spatula (fixed) ----
        spat_mesh = trimesh(*load_usd_mesh(SPATULA_USD))
        label_surface(spat_mesh)
        spat_mesh.triangles().create(builtin.orient, 1)
        ab_shell.apply_to(spat_mesh, 100.0 * MPa, thickness=0.001)
        default_elem.apply_to(spat_mesh)
        view(spat_mesh.transforms())[0] = quat_pos_to_4x4(spatula_quat, spatula_pos)
        view(spat_mesh.instances().find(builtin.is_fixed))[0] = 1
        spat_obj = scene.objects().create("spatula")
        spat_obj.geometries().create(spat_mesh)

        # ---- Broccoli (dynamic, N instances) ----
        broc_objs = []
        for bi, bd in enumerate(broc_list):
            pre_tf = Transform.Identity()
            pre_tf.scale(bd["scale"])
            bio = SimplicialComplexIO(pre_tf)
            bmesh = bio.read(str(BROCCOLI_OBJ))
            label_surface(bmesh)
            bmesh.triangles().create(builtin.orient, 1)
            ab_shell.apply_to(bmesh, 100.0 * MPa, thickness=0.001)
            default_elem.apply_to(bmesh)
            M = np.eye(4, dtype=np.float64)
            M[:3, 3] = bd["pos"]
            view(bmesh.transforms())[0] = M
            bobj = scene.objects().create(f"broc_{bi}")
            bobj.geometries().create(bmesh)
            broc_objs.append(bobj)

        # ---- Tomato slices (FEM deformable, dynamic) ----
        tomato_objs = []
        tomato_moduli = ElasticModuli.youngs_poisson(0.01 * MPa, 0.45)
        for ti, td in enumerate(tomato_list):
            sc = td["scale"]
            Vs_scaled = tomato_tet_Vs * sc
            Vs_offset = Vs_scaled + np.array(td["pos"], dtype=np.float64)
            tmesh = tetmesh(Vs_offset, tomato_tet_Ts)
            label_surface(tmesh)
            label_triangle_orient(tmesh)
            tmesh = flip_inward_triangles(tmesh)
            snh.apply_to(tmesh, tomato_moduli)
            default_elem.apply_to(tmesh)
            tobj = scene.objects().create(f"tomato_{ti}")
            tobj.geometries().create(tmesh)
            tomato_objs.append(tobj)

        # ---- Noodles (deformable, dynamic) ----
        noodles_obj = scene.objects().create("noodles")
        nx, ny = noodle_nx[0], noodle_ny[0]
        ne = noodle_n_edges[0]
        nv = ne + 1
        length = noodle_length[0]
        spacing = noodle_spacing[0]
        x_half = (nx - 1) * spacing / 2.0
        y_half = (ny - 1) * spacing / 2.0
        rng = np.random.default_rng(42)
        perturb = spacing * 0.15
        for ix in range(nx):
            for iy in range(ny):
                bx = noodle_center[0] + ix * spacing - x_half
                by = noodle_center[1] + iy * spacing - y_half
                Vs = np.zeros((nv, 3), dtype=np.float64)
                for i in range(nv):
                    dx = rng.uniform(-perturb, perturb)
                    dy = rng.uniform(-perturb, perturb)
                    Vs[i] = [bx + dx, by + dy,
                             noodle_center[2] + i * (length / ne)]
                Es = np.array([[j, j + 1] for j in range(ne)], dtype=np.int32)
                mesh = linemesh(Vs, Es)
                label_surface(mesh)
                hs.apply_to(mesh, thickness=noodle_thickness[0])
                krb.apply_to(mesh, noodle_bending[0])
                default_elem.apply_to(mesh)
                mesh_partition(mesh, 16)
                noodles_obj.geometries().create(mesh)

        # ---- Ground ----
        ground_obj = scene.objects().create("ground")
        ground_obj.geometries().create(ground(ground_z[0], np.array([0.0, 0.0, 1.0])))

        # ---- Init ----
        w.init(scene)
        print(f"[sim] world.init() done  ({len(broc_list)} broc, "
              f"{len(tomato_list)} tomato, {nx*ny} noodles)")

        sgui = SceneGUI(scene, "merge")
        sgui.register()

        engine_ref[0] = engine
        scene_ref[0] = scene
        world_ref[0] = w
        sgui_ref[0] = sgui
        noodles_obj_ref[0] = noodles_obj
        broc_objs_ref[0] = broc_objs
        tomato_objs_ref[0] = tomato_objs
        mode[0] = "simulation"
        sim_running[0] = True

        w.advance()
        w.retrieve()
        sgui.update()

    def _back_to_placement():
        ps.remove_all_structures()
        engine_ref[0] = None
        scene_ref[0] = None
        world_ref[0] = None
        sgui_ref[0] = None
        noodles_obj_ref[0] = None
        broc_objs_ref[0] = []
        tomato_objs_ref[0] = []
        mode[0] = "placement"
        sim_running[0] = False
        _show_placement()

    # ================================================================
    # Save
    # ================================================================
    save_msg = [""]
    save_t = [0.0]

    def _capture_sim_state():
        """Read current geometry state from UIPC scene after retrieve()."""
        scene = scene_ref[0]
        noodle_verts = {}
        broc_tfs = {}
        tomato_verts = {}

        if noodles_obj_ref[0] is not None:
            geo_ids = noodles_obj_ref[0].geometries().ids()
            for i, gid in enumerate(geo_ids):
                slot, _ = scene.geometries().find(int(gid))
                geo = slot.geometry()
                pos = np.array(view(geo.positions()), dtype=np.float64)
                noodle_verts[f"noodle_{i}"] = pos.tolist()

        for bi, bobj in enumerate(broc_objs_ref[0]):
            geo_ids = bobj.geometries().ids()
            slot, _ = scene.geometries().find(int(geo_ids[0]))
            geo = slot.geometry()
            tf = np.array(view(geo.transforms())[0], dtype=np.float64)
            broc_tfs[f"broc_{bi}"] = tf.tolist()

        for ti, tobj in enumerate(tomato_objs_ref[0]):
            geo_ids = tobj.geometries().ids()
            slot, _ = scene.geometries().find(int(geo_ids[0]))
            geo = slot.geometry()
            pos = np.array(view(geo.positions()), dtype=np.float64)
            tomato_verts[f"tomato_{ti}"] = pos.tolist()

        return noodle_verts, broc_tfs, tomato_verts

    def _save(with_record: bool):
        broc_map = {}
        for bi, bd in enumerate(broc_list):
            broc_map[f"broc_{bi}"] = {"pos": list(bd["pos"]), "scale": bd["scale"]}
        tomato_map = {}
        for ti, td in enumerate(tomato_list):
            tomato_map[f"tomato_{ti}"] = {"pos": list(td["pos"]),
                                          "scale": td["scale"]}
        data = {
            "pan": {"pos": list(pan_pos), "quat": list(pan_quat)},
            "spatula": {"pos": list(spatula_pos), "quat": list(spatula_quat)},
            "broccoli": broc_map,
            "tomato": tomato_map,
            "noodles": {
                "center": list(noodle_center),
                "grid_nx": noodle_nx[0],
                "grid_ny": noodle_ny[0],
                "spacing": noodle_spacing[0],
                "n_edges": noodle_n_edges[0],
                "length": noodle_length[0],
                "thickness": noodle_thickness[0],
                "bending_stiffness": noodle_bending[0],
            },
            "ground_height": ground_z[0],
        }
        if with_record:
            noodle_verts, broc_tfs, tomato_vs = _capture_sim_state()
            data["recorded_state"] = {
                "noodles": noodle_verts,
                "broccoli": broc_tfs,
                "tomato": tomato_vs,
            }
            print(f"[place] captured {len(noodle_verts)} noodles, "
                  f"{len(broc_tfs)} broccoli, {len(tomato_vs)} tomato")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        tag = " (+state)" if with_record else ""
        save_msg[0] = f"Saved{tag} -> {Path(output_path).name}"
        save_t[0] = time.time()
        print(f"[place] {save_msg[0]}")

    # ================================================================
    # Polyscope init
    # ================================================================
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    _show_placement()

    noodle_dirty = [False]

    # ================================================================
    # GUI callback
    # ================================================================
    def on_update():
        is_placement = mode[0] == "placement"

        # ======================== Pan ========================
        imgui.Text("=== Pan ===")
        pan_changed = False
        c, pan_pos[0] = imgui.SliderFloat("Pan X", pan_pos[0], -0.5, 1.5)
        pan_changed |= c
        c, pan_pos[1] = imgui.SliderFloat("Pan Y", pan_pos[1], -1.0, 1.0)
        pan_changed |= c
        c, pan_pos[2] = imgui.SliderFloat("Pan Z", pan_pos[2], 0.0, 2.0)
        pan_changed |= c
        if pan_changed and is_placement:
            _update_pan()
        imgui.Separator()

        # ======================== Spatula ========================
        imgui.Text("=== Spatula ===")
        spat_changed = False
        c, spatula_pos[0] = imgui.SliderFloat("Spat X", spatula_pos[0], -0.5, 1.5)
        spat_changed |= c
        c, spatula_pos[1] = imgui.SliderFloat("Spat Y", spatula_pos[1], -1.0, 1.0)
        spat_changed |= c
        c, spatula_pos[2] = imgui.SliderFloat("Spat Z", spatula_pos[2], 0.0, 2.0)
        spat_changed |= c
        if spat_changed and is_placement:
            _update_spatula()
        imgui.Separator()

        # ======================== Broccoli ========================
        imgui.Text(f"=== Broccoli ({len(broc_list)}) ===")
        if is_placement and imgui.Button("Add Broccoli"):
            broc_list.append({
                "pos": [pan_pos[0], pan_pos[1], pan_pos[2] + 0.05],
                "scale": 0.004,
            })
            _update_broccoli()

        broc_changed = False
        to_remove = None
        for bi, bd in enumerate(broc_list):
            imgui.Text(f"-- broc_{bi} --")
            c, bd["pos"][0] = imgui.SliderFloat(f"X##b{bi}", bd["pos"][0], -0.5, 1.5)
            broc_changed |= c
            c, bd["pos"][1] = imgui.SliderFloat(f"Y##b{bi}", bd["pos"][1], -1.0, 1.0)
            broc_changed |= c
            c, bd["pos"][2] = imgui.SliderFloat(f"Z##b{bi}", bd["pos"][2], 0.0, 2.0)
            broc_changed |= c
            c, bd["scale"] = imgui.SliderFloat(f"Scale##b{bi}", bd["scale"], 0.001, 0.05)
            broc_changed |= c
            if is_placement and len(broc_list) > 1:
                if imgui.Button(f"Remove##b{bi}"):
                    to_remove = bi
        if to_remove is not None:
            broc_list.pop(to_remove)
            broc_changed = True
        if broc_changed and is_placement:
            _update_broccoli()
        imgui.Separator()

        # ======================== Tomato ========================
        imgui.Text(f"=== Tomato ({len(tomato_list)}) ===")
        if is_placement and imgui.Button("Add Tomato"):
            tomato_list.append({
                "pos": [pan_pos[0], pan_pos[1], pan_pos[2] + 0.05],
                "scale": 1.0,
            })
            _update_tomato()

        tomato_changed = False
        tom_remove = None
        for ti, td in enumerate(tomato_list):
            imgui.Text(f"-- tomato_{ti} --")
            c, td["pos"][0] = imgui.SliderFloat(
                f"X##t{ti}", td["pos"][0], -0.5, 1.5)
            tomato_changed |= c
            c, td["pos"][1] = imgui.SliderFloat(
                f"Y##t{ti}", td["pos"][1], -1.0, 1.0)
            tomato_changed |= c
            c, td["pos"][2] = imgui.SliderFloat(
                f"Z##t{ti}", td["pos"][2], 0.0, 2.0)
            tomato_changed |= c
            c, td["scale"] = imgui.SliderFloat(
                f"Scale##t{ti}", td["scale"], 0.1, 5.0)
            tomato_changed |= c
            if is_placement and len(tomato_list) > 1:
                if imgui.Button(f"Remove##t{ti}"):
                    tom_remove = ti
        if tom_remove is not None:
            tomato_list.pop(tom_remove)
            tomato_changed = True
        if tomato_changed and is_placement:
            _update_tomato()
        imgui.Separator()

        # ======================== Noodles ========================
        imgui.Text("=== Noodles ===")
        c, noodle_center[0] = imgui.SliderFloat("Noodle X", noodle_center[0], -0.5, 1.5)
        noodle_dirty[0] |= c
        c, noodle_center[1] = imgui.SliderFloat("Noodle Y", noodle_center[1], -1.0, 1.0)
        noodle_dirty[0] |= c
        c, noodle_center[2] = imgui.SliderFloat("Noodle Z", noodle_center[2], 0.0, 2.0)
        noodle_dirty[0] |= c
        c, noodle_nx[0] = imgui.InputInt("Grid NX", noodle_nx[0])
        noodle_dirty[0] |= c
        c, noodle_ny[0] = imgui.InputInt("Grid NY", noodle_ny[0])
        noodle_dirty[0] |= c
        c, noodle_spacing[0] = imgui.SliderFloat("Spacing", noodle_spacing[0], 0.002, 0.03)
        noodle_dirty[0] |= c
        c, noodle_n_edges[0] = imgui.InputInt("Edges/noodle", noodle_n_edges[0])
        noodle_dirty[0] |= c
        c, noodle_length[0] = imgui.SliderFloat("Length", noodle_length[0], 0.02, 0.5)
        noodle_dirty[0] |= c
        c, noodle_thickness[0] = imgui.SliderFloat("Thickness", noodle_thickness[0], 0.0005, 0.005)
        noodle_dirty[0] |= c
        c, noodle_bending[0] = imgui.InputFloat("Bend Stiff", noodle_bending[0])
        noodle_dirty[0] |= c
        noodle_nx[0] = max(1, noodle_nx[0])
        noodle_ny[0] = max(1, noodle_ny[0])
        noodle_n_edges[0] = max(3, noodle_n_edges[0])
        total = noodle_nx[0] * noodle_ny[0]
        imgui.Text(f"Total: {total} noodles")
        if noodle_dirty[0] and is_placement:
            _update_noodles()
            noodle_dirty[0] = False
        imgui.Separator()

        # ======================== Ground ========================
        imgui.Text("=== Ground ===")
        c, ground_z[0] = imgui.SliderFloat("Ground Z", ground_z[0], -1.0, 1.0)
        if c and is_placement:
            _show_placement()
        imgui.Separator()

        # ======================== Controls ========================
        if is_placement:
            imgui.Text("[PLACEMENT MODE]")
            if imgui.Button("Build & Run"):
                _build_and_run()
            imgui.SameLine()
            if imgui.Button("Save Placement"):
                _save(with_record=False)
        else:
            frame = world_ref[0].frame() if world_ref[0] else 0
            imgui.Text(f"[SIMULATION]  frame={frame}")

            if imgui.Button("Run / Stop"):
                sim_running[0] = not sim_running[0]

            if sim_running[0]:
                world_ref[0].advance()
                world_ref[0].retrieve()
                sgui_ref[0].update()

            if imgui.Button("Back to Placement"):
                _back_to_placement()

            imgui.SameLine()
            if imgui.Button("Record & Save"):
                world_ref[0].advance()
                world_ref[0].retrieve()
                sgui_ref[0].update()
                _save(with_record=True)

        imgui.Separator()
        imgui.Text(f"Output: {Path(output_path).name}")
        if save_msg[0] and (time.time() - save_t[0]) < 3.0:
            imgui.Text(save_msg[0])

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
