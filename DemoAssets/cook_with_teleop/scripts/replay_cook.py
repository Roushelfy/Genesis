"""UIPC scene: replay pan & spatula trajectories with full contact physics.

Reads the cooking teleop trajectory (pos + quat per frame) and drives
pan / spatula as kinematic affine bodies via SoftTransformConstraint.
Noodles, broccoli, and tomato slices are dynamic and respond to contact.

Assets are loaded directly from the teleop-provided USD files under
``DemoAssets/cook/`` using OpenUSD (pxr).

Usage:
    python replay_cook.py [--no-gui] [--frames N] [--traj PATH] [--speed 0.4]
    python replay_cook.py --export-seq ../seq       # export sequence for Genesis replay
    python replay_cook.py --export-recover 5000 --export-seq ../seq  # export from dump
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from uipc import Animation, Logger, Timer, Transform, builtin, view
from uipc import Engine, Scene, SceneIO, World
from uipc.constitution import (
    AffineBodyConstitution, AffineBodyShell, ElasticModuli, HookeanSpring,
    KirchhoffRodBending, SoftTransformConstraint, StableNeoHookean,
)
from uipc.geometry import (
    SimplicialComplexIO, flip_inward_triangles, ground, label_surface,
    label_triangle_orient, linemesh, mesh_partition, tetmesh, trimesh,
)
from uipc.unit import GPa, MPa

from asset_dir import AssetDir
from usd_mesh_loader import load_usd_mesh

_HERE = Path(__file__).resolve().parent
_ASSET_ROOT = _HERE.parent
_COOK_ROOT = _ASSET_ROOT.parent / "cook"

# Teleop-provided USD assets
PAN_USD = _COOK_ROOT / "Pan025" / "Pan025.usd"
SPATULA_USD = _COOK_ROOT / "Spatula018" / "Spatula018.usd"
BROCCOLI_OBJ = _ASSET_ROOT / "broccoli.obj"
TOMATO_NPZ = _ASSET_ROOT / "tomato_slice.npz"
MUSHROOM_NPZ = _ASSET_ROOT / "mushroom_slice.npz"
MUSHROOM_OBJ = _ASSET_ROOT / "mushroom_slice.obj"
DEFAULT_TRAJ = _COOK_ROOT / "trajectories" / "cooking_keyframed.json"
DEFAULT_PLACEMENT = _ASSET_ROOT / "placement.json"


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


def _nlerp(q0, q1, t):
    """Normalized linear interpolation for quaternions [w,x,y,z]."""
    a = np.array(q0, dtype=np.float64)
    b = np.array(q1, dtype=np.float64)
    if np.dot(a, b) < 0.0:
        b = -b
    q = (1.0 - t) * a + t * b
    return (q / np.linalg.norm(q)).tolist()


def _pos_4x4(pos):
    """Translation-only 4x4."""
    M = np.eye(4, dtype=np.float64)
    M[0, 3], M[1, 3], M[2, 3] = pos[0], pos[1], pos[2]
    return M


def load_mesh_usd(usd_path: Path):
    """Load mesh from a USD file via pxr, return a UIPC SimplicialComplex."""
    if not usd_path.exists():
        raise FileNotFoundError(f"USD asset not found: {usd_path}")
    print(f"[scene] Loading USD: {usd_path}")
    verts, faces = load_usd_mesh(usd_path)
    return trimesh(verts, faces)


# ---------------------------------------------------------------------------
# Sequence exporter (for Genesis replay via ipc_robot_cook.py)
# ---------------------------------------------------------------------------


class CookSequenceExporter:
    """Accumulate per-frame cooking scene state and write a NPY sequence.

    Object types:
      rigid — ABD shells (pan, spatula, broccoli): per-frame 4x4 transforms
      fem   — deformable tets (tomato slices): per-frame vertex positions
      rod   — line meshes (noodles): per-frame vertex positions (concatenated)
    """

    def __init__(self, scene, dt: float):
        self._scene = scene
        self._dt = dt
        self._frame_ids: list[int] = []
        self._rigid_objs: dict[str, object] = {}
        self._fem_objs: dict[str, object] = {}
        self._noodles_obj = None
        self._rigid_frames: dict[str, list[np.ndarray]] = {}
        self._fem_frames: dict[str, list[np.ndarray]] = {}
        self._noodles_frames: list[np.ndarray] = []
        self._mesh_cache: dict[str, tuple] = {}

    def register_rigid(self, name, obj, rest_verts=None, rest_faces=None):
        self._rigid_objs[name] = obj
        if rest_verts is not None and rest_faces is not None:
            self._mesh_cache[name] = (
                np.array(rest_verts, dtype=np.float64),
                np.array(rest_faces, dtype=np.int32),
                "tri",
            )

    def register_fem(self, name, obj, surface_verts=None, surface_faces=None):
        self._fem_objs[name] = obj
        if surface_verts is not None and surface_faces is not None:
            self._mesh_cache[name] = (
                np.array(surface_verts, dtype=np.float64),
                np.array(surface_faces, dtype=np.int32),
                "tri",
            )

    def register_noodles(self, obj):
        self._noodles_obj = obj

    def capture(self, frame_id: int) -> None:
        self._frame_ids.append(frame_id)
        for name, obj in self._rigid_objs.items():
            geo_ids = obj.geometries().ids()
            slot, _ = self._scene.geometries().find(int(geo_ids[0]))
            geo = slot.geometry()
            tf = np.array(view(geo.transforms())[0], dtype=np.float64, copy=True)
            self._rigid_frames.setdefault(name, []).append(tf)

        for name, obj in self._fem_objs.items():
            geo_ids = obj.geometries().ids()
            slot, _ = self._scene.geometries().find(int(geo_ids[0]))
            geo = slot.geometry()
            pos = np.array(
                view(geo.positions()), dtype=np.float64, copy=True
            ).reshape(-1, 3)
            self._fem_frames.setdefault(name, []).append(pos)

        if self._noodles_obj is not None:
            geo_ids = self._noodles_obj.geometries().ids()
            parts = []
            for gid in geo_ids:
                slot, _ = self._scene.geometries().find(int(gid))
                geo = slot.geometry()
                pos = np.array(
                    view(geo.positions()), dtype=np.float64, copy=True
                ).reshape(-1, 3)
                parts.append(pos)
            if parts:
                self._noodles_frames.append(np.concatenate(parts, axis=0))

    # ------------------------------------------------------------------

    def save(self, seq_dir: Path) -> None:
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_skip = 1
        if len(self._frame_ids) >= 2:
            frame_skip = max(1, self._frame_ids[1] - self._frame_ids[0])
        meta: dict = {
            "frame_count": len(self._frame_ids),
            "frame_ids": self._frame_ids,
            "frame_skip": frame_skip,
            "dt": self._dt,
            "objects": {},
        }

        for name, frames in self._rigid_frames.items():
            obj_dir = seq_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(frames, axis=0)
            np.save(str(obj_dir / "transforms.npy"), arr)
            self._write_mesh(name, obj_dir)
            meta["objects"][name] = {
                "type": "rigid",
                "data": f"{name}/transforms.npy",
            }

        for name, frames in self._fem_frames.items():
            obj_dir = seq_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(frames, axis=0)
            np.save(str(obj_dir / "positions.npy"), arr)
            self._write_mesh(name, obj_dir)
            meta["objects"][name] = {
                "type": "fem",
                "data": f"{name}/positions.npy",
            }

        if self._noodles_frames:
            obj_dir = seq_dir / "noodles"
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(self._noodles_frames, axis=0)
            np.save(str(obj_dir / "positions.npy"), arr)
            self._write_noodles_mesh(obj_dir)
            meta["objects"]["noodles"] = {
                "type": "rod",
                "data": "noodles/positions.npy",
            }

        meta_path = seq_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        n_rigid = len(self._rigid_frames)
        n_fem = len(self._fem_frames)
        n_rod = 1 if self._noodles_frames else 0
        print(
            f"[seq-export] saved {len(self._frame_ids)} frames to {seq_dir}\n"
            f"  {n_rigid} rigid, {n_fem} fem, {n_rod} rod objects"
        )

    def _write_mesh(self, name: str, obj_dir: Path) -> None:
        mesh_dst = obj_dir / "mesh.obj"
        if mesh_dst.exists():
            return
        if name not in self._mesh_cache:
            return
        verts, topo, mesh_type = self._mesh_cache[name]
        with open(mesh_dst, "w") as f:
            for v in verts:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            if mesh_type == "tri":
                for tri in topo:
                    f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    def _write_noodles_mesh(self, obj_dir: Path) -> None:
        mesh_dst = obj_dir / "mesh.obj"
        if mesh_dst.exists() or not self._noodles_frames:
            return
        first_pos = self._noodles_frames[0]
        geo_ids = self._noodles_obj.geometries().ids()
        offset = 0
        edges: list[list[int]] = []
        for gid in geo_ids:
            slot, _ = self._scene.geometries().find(int(gid))
            geo = slot.geometry()
            n_v = len(
                np.array(view(geo.positions()), dtype=np.float64).reshape(-1, 3)
            )
            for j in range(n_v - 1):
                edges.append([offset + j, offset + j + 1])
            offset += n_v
        with open(mesh_dst, "w") as f:
            for v in first_pos:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="UIPC kinematic replay of cooking teleop")
    parser.add_argument("--traj", type=str, default=str(DEFAULT_TRAJ),
                        help="Path to trajectory JSON")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Simulation timestep (smaller = more accurate)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed multiplier")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without polyscope GUI (headless)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Max traj frames to replay (0 = all)")
    parser.add_argument("--placement", type=str, default=str(DEFAULT_PLACEMENT),
                        help="Placement JSON from place_all_stuff.py (with recorded_state)")
    parser.add_argument("--export-seq", type=str, default="",
                        help="Directory to export sequence for Genesis replay")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Capture every N-th frame during export (default 1)")
    parser.add_argument("--export-recover", type=int, default=-1,
                        help="Recover up to N frames from dump and export (headless)")
    args = parser.parse_args()

    # ---- Load placement (recorded state from simulation) ----
    placement = {}
    rec_state = {}
    if Path(args.placement).exists():
        with open(args.placement) as f:
            placement = json.load(f)
        rec_state = placement.get("recorded_state", {})
        print(f"[init] Loaded placement: {args.placement}")
        if rec_state:
            n_broc = len(rec_state.get("broccoli", {}))
            n_nood = len(rec_state.get("noodles", {}))
            print(f"  recorded_state: {n_broc} broccoli, {n_nood} noodles")
    else:
        print(f"[init] No placement file, using defaults")

    # ---- Load trajectory ----
    with open(args.traj) as f:
        traj_data = json.load(f)
    frames = traj_data["frames"]
    n_traj = len(frames)
    if args.frames > 0:
        n_traj = min(n_traj, args.frames)
        frames = frames[:n_traj]
    traj_times = [f.get("sim_time", i / 60.0) for i, f in enumerate(frames)]
    traj_t0 = traj_times[0]
    traj_t_end = traj_times[-1]
    traj_duration = traj_t_end - traj_t0
    print(f"Loaded {n_traj} traj frames from {args.traj}")
    print(f"  sim_time range: {traj_t0:.3f} - {traj_t_end:.3f} s "
          f"(duration {traj_duration:.3f}s)")
    print(f"  dt={args.dt}, speed={args.speed}")

    LEADIN_DURATION = 0.06  # 30 frames at dt=0.002
    SPATULA_HIGH_Z  = 1.25  # spatula start height (matches place_all_stuff)
    cur_traj_time = [traj_t0 - LEADIN_DURATION]  # negative = lead-in phase
    speed_val = [args.speed]

    # ---- UIPC engine / scene ----
    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = args.dt
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["contact"]["d_hat"] = 0.0005
    config["newton"]["semi_implicit"] = False
    config["newton"]["velocity_tol"] = 0.5
    config["newton"]["transrate_tol"] = 10
    config["newton"]["max_iter"] = 1024
    config["linear_system"]["tol_rate"] = 1e-4
    config["sanity_check"]["enable"] = 0
    scene = Scene(config)
    scene.animator().substep(1)

    ab_shell = AffineBodyShell()
    snh = StableNeoHookean()
    stc = SoftTransformConstraint()

    # ---- Contact tabular: pan-spatula OFF, broccoli-{pan,spatula} ON ----
    tabular = scene.contact_tabular()
    tabular.default_model(0.0, 1.0 * GPa, False)  # default: no contact

    pan_contact = tabular.create("pan")
    spatula_contact = tabular.create("spatula")
    broccoli_contact = tabular.create("broccoli")

    noodle_contact = tabular.create("noodle")
    tomato_contact = tabular.create("tomato")
    mushroom_contact = tabular.create("mushroom")

    pan_friction = 0.01
    tabular.insert(pan_contact, spatula_contact, 0.0, 1.0 * GPa, enable=True)
    tabular.insert(broccoli_contact, pan_contact, 0.1, 1.0 * GPa, enable=True)
    tabular.insert(broccoli_contact, spatula_contact, 0.5, 1.0 * GPa, enable=True)
    tabular.insert(broccoli_contact, broccoli_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(noodle_contact, pan_contact, pan_friction, 1.0 * GPa, enable=True)
    tabular.insert(noodle_contact, spatula_contact, 0.5, 1.0 * GPa, enable=True)
    tabular.insert(noodle_contact, broccoli_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(noodle_contact, noodle_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(tomato_contact, pan_contact, pan_friction, 1.0 * GPa, enable=True)
    tabular.insert(tomato_contact, spatula_contact, 0.5, 1.0 * GPa, enable=True)
    tabular.insert(tomato_contact, broccoli_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(tomato_contact, noodle_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(tomato_contact, tomato_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, pan_contact, pan_friction, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, spatula_contact, 0.5, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, broccoli_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, noodle_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, tomato_contact, 0.3, 1.0 * GPa, enable=True)
    tabular.insert(mushroom_contact, mushroom_contact, 0.0, 0.2 * GPa, enable=True)

    # ---- Initial placement from trajectory frame 0 ----
    frame0 = frames[0]
    _raw_scale = placement.get("pan", {}).get("scale", [1.0, 1.0, 1.0])
    if isinstance(_raw_scale, (int, float)):
        pan_scale = np.array([_raw_scale, _raw_scale, _raw_scale], dtype=np.float64)
    else:
        pan_scale = np.array(_raw_scale, dtype=np.float64)

    def _setup_kinematic_mesh(usd_path, entity_name, contact_elem, thickness,
                              scale=None, stc_strength=None):
        """Load USD mesh for kinematic replay (shell + STC + contact element)."""
        if scale is not None:
            verts, faces = load_usd_mesh(usd_path)
            mesh = trimesh(verts * scale, faces)
        else:
            mesh = load_mesh_usd(usd_path)
        label_surface(mesh)
        mesh.triangles().create(builtin.orient, 1)
        ab_shell.apply_to(mesh, 100.0 * MPa, thickness=thickness)
        stc.apply_to(mesh, stc_strength)
        contact_elem.apply_to(mesh)
        data = frame0.get(entity_name)
        if data and "pos" in data and "quat" in data:
            init_pos = list(data["pos"])
            if entity_name == "spatula":
                init_pos[2] = SPATULA_HIGH_Z
            init_mat = quat_pos_to_4x4(data["quat"], init_pos)
            view(mesh.transforms())[0] = init_mat
            print(f"[scene] {entity_name} initial pos: {init_pos}")
        obj = scene.objects().create(entity_name)
        obj.geometries().create(mesh)
        return obj

    pan_obj = _setup_kinematic_mesh(PAN_USD, "pan", pan_contact, 0.0001,
                                    scale=pan_scale, stc_strength=np.array([1.0, 1.0]))
    spatula_obj = _setup_kinematic_mesh(SPATULA_USD, "spatula", spatula_contact, 0.001, stc_strength=np.array([0.1, 0.1]))

    # ---- Broccoli (dynamic, from placement recorded_state) ----
    broc_cfg = placement.get("broccoli", {})
    broc_rec = rec_state.get("broccoli", {})
    broc_objs: dict[str, object] = {}
    broc_mesh_data: dict[str, tuple] = {}
    for key in sorted(broc_cfg.keys()):
        bd = broc_cfg[key]
        scale = bd.get("scale", 0.004)
        pre_tf = Transform.Identity()
        pre_tf.scale(scale)
        bio = SimplicialComplexIO(pre_tf)
        bmesh = bio.read(str(BROCCOLI_OBJ))
        label_surface(bmesh)
        bmesh.triangles().create(builtin.orient, 1)
        bv = np.array(view(bmesh.positions()), dtype=np.float64, copy=True).reshape(-1, 3)
        bt = np.array(view(bmesh.triangles().topo()), dtype=np.int32, copy=True).reshape(-1, 3)
        broc_mesh_data[key] = (bv, bt)
        ab_shell.apply_to(bmesh, 100.0 * MPa, thickness=0.001, mass_density=600.0)
        broccoli_contact.apply_to(bmesh)
        if key in broc_rec:
            view(bmesh.transforms())[0] = np.array(broc_rec[key], dtype=np.float64)
            print(f"[scene] {key}: transform from recorded_state")
        else:
            view(bmesh.transforms())[0] = _pos_4x4(bd["pos"])
            print(f"[scene] {key}: pos from placement config")
        bobj = scene.objects().create(key)
        bobj.geometries().create(bmesh)
        broc_objs[key] = bobj
    print(f"[scene] {len(broc_cfg)} broccoli loaded")

    # ---- Noodles (deformable) ----
    noodle_rec = rec_state.get("noodles", {})
    noodle_cfg = placement.get("noodles", {})
    thickness = noodle_cfg.get("thickness", 0.001)
    bending = noodle_cfg.get("bending_stiffness", 1e4)

    hs = HookeanSpring()
    krb = KirchhoffRodBending()
    noodles_obj = scene.objects().create("noodles")
    noodle_count = 0

    if noodle_rec:
        for key in sorted(noodle_rec.keys()):
            verts = np.array(noodle_rec[key], dtype=np.float64)
            n_v = len(verts)
            edges = np.array([[j, j + 1] for j in range(n_v - 1)], dtype=np.int32)
            nmesh = linemesh(verts, edges)
            label_surface(nmesh)
            hs.apply_to(nmesh, thickness=thickness, mass_density=600.0)
            krb.apply_to(nmesh, bending)
            noodle_contact.apply_to(nmesh)
            mesh_partition(nmesh, 16)
            noodles_obj.geometries().create(nmesh)
            noodle_count += 1
        print(f"[scene] {noodle_count} noodles from recorded_state")
    elif noodle_cfg:
        nx = noodle_cfg.get("grid_nx", 8)
        ny = noodle_cfg.get("grid_ny", 8)
        ne = noodle_cfg.get("n_edges", 50)
        nv = ne + 1
        length = noodle_cfg.get("length", 0.2)
        spacing = noodle_cfg.get("spacing", 0.006)
        center = noodle_cfg.get("center", [0.5, 0.0, 0.9])
        x_half = (nx - 1) * spacing / 2.0
        y_half = (ny - 1) * spacing / 2.0
        rng = np.random.default_rng(42)
        perturb = spacing * 0.15
        for ix in range(nx):
            for iy in range(ny):
                bx = center[0] + ix * spacing - x_half
                by = center[1] + iy * spacing - y_half
                Vs = np.zeros((nv, 3), dtype=np.float64)
                for i in range(nv):
                    dx = rng.uniform(-perturb, perturb)
                    dy = rng.uniform(-perturb, perturb)
                    Vs[i] = [bx + dx, by + dy,
                             center[2] + i * (length / ne)]
                Es = np.array([[j, j + 1] for j in range(ne)], dtype=np.int32)
                nmesh = linemesh(Vs, Es)
                label_surface(nmesh)
            hs.apply_to(nmesh, thickness=thickness, mass_density=600.0)
            krb.apply_to(nmesh, bending)
            noodle_contact.apply_to(nmesh)
            mesh_partition(nmesh, 16)
            noodles_obj.geometries().create(nmesh)
            noodle_count += 1
        print(f"[scene] {noodle_count} noodles generated from placement config")

    # ---- Tomato slices (FEM deformable) ----
    tomato_cfg = placement.get("tomato", {})
    tomato_rec = rec_state.get("tomato", {})
    tomato_tet_data = np.load(TOMATO_NPZ)
    tomato_base_Vs = tomato_tet_data["vertices"].astype(np.float64)
    tomato_base_Ts = tomato_tet_data["tetrahedra"].astype(np.int32)
    tomato_moduli = ElasticModuli.youngs_poisson(0.01 * MPa, 0.45)
    tomato_count = 0
    tomato_objs: dict[str, object] = {}
    tomato_surf_data: dict[str, tuple] = {}

    if tomato_rec:
        for key in sorted(tomato_rec.keys()):
            verts = np.array(tomato_rec[key], dtype=np.float64)
            tmesh = tetmesh(verts, tomato_base_Ts)
            label_surface(tmesh)
            label_triangle_orient(tmesh)
            tmesh = flip_inward_triangles(tmesh)
            tv = np.array(view(tmesh.positions()), dtype=np.float64, copy=True).reshape(-1, 3)
            tt = np.array(view(tmesh.triangles().topo()), dtype=np.int32, copy=True).reshape(-1, 3)
            tomato_surf_data[key] = (tv, tt)
            snh.apply_to(tmesh, tomato_moduli, mass_density=600.0)
            tomato_contact.apply_to(tmesh)
            tobj = scene.objects().create(key)
            tobj.geometries().create(tmesh)
            tomato_objs[key] = tobj
            tomato_count += 1
        print(f"[scene] {tomato_count} tomato slices from recorded_state")
    elif tomato_cfg:
        for key in sorted(tomato_cfg.keys()):
            td = tomato_cfg[key]
            sc = td.get("scale", 1.0)
            pos = td.get("pos", [0, 0, 0])
            Vs = tomato_base_Vs * sc + np.array(pos, dtype=np.float64)
            tmesh = tetmesh(Vs, tomato_base_Ts)
            label_surface(tmesh)
            label_triangle_orient(tmesh)
            tmesh = flip_inward_triangles(tmesh)
            tv = np.array(view(tmesh.positions()), dtype=np.float64, copy=True).reshape(-1, 3)
            tt = np.array(view(tmesh.triangles().topo()), dtype=np.int32, copy=True).reshape(-1, 3)
            tomato_surf_data[key] = (tv, tt)
            snh.apply_to(tmesh, tomato_moduli, mass_density=600.0)
            tomato_contact.apply_to(tmesh)
            tobj = scene.objects().create(key)
            tobj.geometries().create(tmesh)
            tomato_objs[key] = tobj
            tomato_count += 1
        print(f"[scene] {tomato_count} tomato slices from placement config")

    # ---- Mushroom slices (FEM deformable, dynamic) ----
    mushroom_cfg = placement.get("mushroom", {})
    mushroom_rec = rec_state.get("mushroom", {})
    mushroom_objs: dict[str, object] = {}
    mushroom_surf_data: dict[str, tuple] = {}
    mushroom_count = 0

    mush_tet_data = np.load(MUSHROOM_NPZ)
    mush_base_Vs = mush_tet_data["vertices"].astype(np.float64)
    mush_base_Ts = mush_tet_data["tetrahedra"].astype(np.int32)
    mushroom_moduli = ElasticModuli.youngs_poisson(0.005 * MPa, 0.45)
    print(f"[scene] Mushroom tet mesh: {len(mush_base_Vs)} nodes, {len(mush_base_Ts)} tets")

    def _make_mushroom_tetmesh(sc, pos_or_verts, from_verts=False):
        if from_verts:
            Vs = np.array(pos_or_verts, dtype=np.float64).reshape(-1, 3)
        else:
            Vs = mush_base_Vs * sc + np.array(pos_or_verts, dtype=np.float64)
        mmesh = tetmesh(Vs, mush_base_Ts)
        label_surface(mmesh)
        label_triangle_orient(mmesh)
        mmesh = flip_inward_triangles(mmesh)
        sv = np.array(view(mmesh.positions()), dtype=np.float64, copy=True).reshape(-1, 3)
        st = np.array(view(mmesh.triangles().topo()), dtype=np.int32, copy=True).reshape(-1, 3)
        snh.apply_to(mmesh, mushroom_moduli, mass_density=600.0)
        mushroom_contact.apply_to(mmesh)
        return mmesh, sv, st

    if mushroom_rec:
        for key in sorted(mushroom_rec.keys()):
            md = mushroom_cfg.get(key, {})
            sc = md.get("scale", 1.0)
            mmesh, sv, st = _make_mushroom_tetmesh(sc, mushroom_rec[key], from_verts=True)
            mushroom_surf_data[key] = (sv, st)
            print(f"[scene] {key}: verts from recorded_state")
            mobj = scene.objects().create(key)
            mobj.geometries().create(mmesh)
            mushroom_objs[key] = mobj
            mushroom_count += 1
        print(f"[scene] {mushroom_count} mushrooms from recorded_state")
    elif mushroom_cfg:
        for key in sorted(mushroom_cfg.keys()):
            md = mushroom_cfg[key]
            sc = md.get("scale", 1.0)
            mmesh, sv, st = _make_mushroom_tetmesh(sc, md.get("pos", [0, 0, 0]))
            mushroom_surf_data[key] = (sv, st)
            print(f"[scene] {key}: pos from placement config")
            mobj = scene.objects().create(key)
            mobj.geometries().create(mmesh)
            mushroom_objs[key] = mobj
            mushroom_count += 1
        print(f"[scene] {mushroom_count} mushrooms from placement config")

    # ---- Ground ----
    ground_h = placement.get("ground_height", 0.0)
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(ground_h, np.array([0.0, 0.0, 1.0])))

    # ---- Animator callbacks (interpolating trajectory) ----
    def _interp_frame(entity_name, t_traj):
        """Interpolate pos/quat for entity at trajectory time t_traj."""
        idx = bisect.bisect_right(traj_times, t_traj) - 1
        idx = max(0, min(idx, n_traj - 1))
        d0 = frames[idx].get(entity_name)
        if d0 is None or "pos" not in d0 or "quat" not in d0:
            return None, None
        if idx >= n_traj - 1:
            return d0["pos"], d0["quat"]
        d1 = frames[idx + 1].get(entity_name)
        if d1 is None or "pos" not in d1 or "quat" not in d1:
            return d0["pos"], d0["quat"]
        dt_seg = traj_times[idx + 1] - traj_times[idx]
        alpha = (t_traj - traj_times[idx]) / dt_seg if dt_seg > 0 else 0.0
        alpha = max(0.0, min(1.0, alpha))
        p0, p1 = np.array(d0["pos"]), np.array(d1["pos"])
        pos = ((1.0 - alpha) * p0 + alpha * p1).tolist()
        quat = _nlerp(d0["quat"], d1["quat"], alpha)
        return pos, quat

    def _make_replay_cb(entity_name: str):
        def _cb(info: Animation.UpdateInfo):
            t = cur_traj_time[0]
            if t > traj_t_end:
                return
            if t < traj_t0:
                # lead-in phase: pan at frame 0, spatula descends from high
                pos, quat = _interp_frame(entity_name, traj_t0)
                if pos is None:
                    return
                if entity_name == "spatula":
                    alpha = (t - (traj_t0 - LEADIN_DURATION)) / LEADIN_DURATION
                    alpha = max(0.0, min(1.0, alpha))
                    pos = list(pos)
                    pos[2] = SPATULA_HIGH_Z + alpha * (pos[2] - SPATULA_HIGH_Z)
            else:
                pos, quat = _interp_frame(entity_name, t)
                if pos is None:
                    return
            geo = info.geo_slots()[0].geometry()
            view(geo.instances().find(builtin.is_constrained))[0] = 1
            mat = quat_pos_to_4x4(quat, pos)
            view(geo.instances().find(builtin.aim_transform))[0] = mat
        return _cb

    animator = scene.animator()
    animator.insert(pan_obj, _make_replay_cb("pan"))
    animator.insert(spatula_obj, _make_replay_cb("spatula"))

    # ---- Init ----
    world.init(scene)
    print("[init] world.init() done")

    # ---- Sequence exporter setup ----
    exporter = None
    seq_dir = Path(args.export_seq) if args.export_seq else None
    if seq_dir is not None or args.export_recover >= 0:
        exporter = CookSequenceExporter(scene, args.dt)
        pan_v, pan_f = load_usd_mesh(PAN_USD)
        exporter.register_rigid("pan", pan_obj, pan_v * pan_scale, pan_f)
        spat_v, spat_f = load_usd_mesh(SPATULA_USD)
        exporter.register_rigid("spatula", spatula_obj, spat_v, spat_f)
        for key, bobj in broc_objs.items():
            bv, bt = broc_mesh_data[key]
            exporter.register_rigid(key, bobj, bv, bt)
        for key, tobj in tomato_objs.items():
            tv, tt = tomato_surf_data[key]
            exporter.register_fem(key, tobj, tv, tt)
        for key, mobj in mushroom_objs.items():
            mv, mt = mushroom_surf_data[key]
            exporter.register_fem(key, mobj, mv, mt)
        if noodle_count > 0:
            exporter.register_noodles(noodles_obj)
        print(f"[export] exporter ready  frame_skip={args.frame_skip}")

    # ---- Export-recover mode (headless, from dump) ----
    if args.export_recover >= 0:
        if seq_dir is None:
            seq_dir = _ASSET_ROOT / "seq"
        world.retrieve()
        exporter.capture(world.frame())
        exported = 0
        for target in range(1, args.export_recover + 1):
            if not world.recover(target):
                print(f"[export-recover] no dump at frame {target}, stopping")
                break
            world.retrieve()
            if target % max(1, args.frame_skip) == 0:
                exporter.capture(target)
                exported += 1
                if exported % 100 == 0:
                    print(f"[export-recover] {exported} frames captured ...")
        exporter.save(seq_dir)
        print(f"[export-recover] Done: {exported} frames")
        return

    # ---- Run ----
    sio = SceneIO(scene)

    if args.no_gui:
        t_start = time.time()
        step = 0
        while cur_traj_time[0] < traj_t_end:
            cur_traj_time[0] += args.dt * speed_val[0]
            world.advance()
            world.retrieve()
            world.dump()
            if exporter and step % max(1, args.frame_skip) == 0:
                exporter.capture(world.frame())
            sio.write_surface(f"{workspace}/surface_{world.frame()}.obj")
            step += 1
            if step % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  step {step}  traj_t={cur_traj_time[0]:.3f}s  "
                      f"({elapsed:.1f}s wall)")
        elapsed = time.time() - t_start
        print(f"[done] {step} steps in {elapsed:.1f}s "
              f"({step/elapsed:.1f} steps/s)")
        if exporter and seq_dir:
            exporter.save(seq_dir)
    else:
        import polyscope as ps
        from polyscope import imgui
        from uipc.gui import SceneGUI

        ps.init()
        sgui = SceneGUI(scene, "split")
        sgui.register()

        run = False
        recover_run = False
        export_surface = [True]
        seq_exported = [False]

        def on_update():
            nonlocal run, recover_run
            if imgui.Button("Run / Stop"):
                run = not run
            if imgui.Button("Recover Run"):
                recover_run = True
            _, speed_val[0] = imgui.SliderFloat(
                "Speed", speed_val[0], 0.05, 3.0)
            _, export_surface[0] = imgui.Checkbox(
                "Export Surface", export_surface[0])
            done = cur_traj_time[0] >= traj_t_end
            total_dur = traj_duration + LEADIN_DURATION
            progress = (cur_traj_time[0] - (traj_t0 - LEADIN_DURATION)) / total_dur
            phase = "lead-in" if cur_traj_time[0] < traj_t0 else "traj"
            imgui.Text(f"Step: {world.frame()}  [{phase}]  "
                       f"t={cur_traj_time[0]:.3f} / {traj_t_end:.3f} s  "
                       f"({progress*100:.0f}%)")
            if done:
                imgui.Text("[FINISHED]")
            if run and not done:
                cur_traj_time[0] += args.dt * speed_val[0]
                world.advance()
                world.retrieve()
                world.dump()
                if exporter and world.frame() % max(1, args.frame_skip) == 0:
                    exporter.capture(world.frame())
                if export_surface[0]:
                    sio.write_surface(
                        f"{workspace}/surface_{world.frame()}.obj")
                sgui.update()
            if recover_run:
                if world.recover(world.frame() + 1):
                    world.retrieve()
                    if exporter and world.frame() % max(1, args.frame_skip) == 0:
                        exporter.capture(world.frame())
                    if export_surface[0]:
                        sio.write_surface(
                            f"{workspace}/surface_{world.frame()}.obj")
                    sgui.update()
                else:
                    recover_run = False

            if exporter and seq_dir:
                if done and not seq_exported[0]:
                    exporter.save(seq_dir)
                    seq_exported[0] = True
                    imgui.Text("[SEQUENCE EXPORTED]")
                elif imgui.Button("Export Sequence Now"):
                    exporter.save(seq_dir)
                    seq_exported[0] = True
                if seq_exported[0]:
                    imgui.Text(f"Exported to {seq_dir}")

        ps.set_user_callback(on_update)
        ps.show()


if __name__ == "__main__":
    main()
