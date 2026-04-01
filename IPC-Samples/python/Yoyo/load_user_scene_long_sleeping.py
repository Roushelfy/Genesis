"""Load the complete yoyo scene (ball + string + bearing) into an existing UIPC scene.

References IPC-Samples/python/Yoyo/1_throw_yoyo_7_with_bearing.py for the
loading pattern.  Gravity direction is Z-down to match the robot's frame.

Provides a global-transform mechanism via ``apply_global_transform``:
ABD objects use ``view(geo.transforms())[0] = global @ original``;
FEM objects (string) transform original vertex positions directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uipc import Animation, Scene, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    ElasticModuli,
    HookeanSpring,
    KirchhoffRodBending,
    RotatingMotor,
    SoftPositionConstraint,
    SoftVertexTriangleStitch,
)
from uipc.geometry import (
    SimplicialComplexIO,
    apply_region,
    apply_transform,
    closest_vertex_triangle_pairs,
    label_region,
    label_surface,
    mesh_partition,
)
from uipc.unit import GPa, MPa

_SCRIPT_DIR = Path(__file__).resolve().parent
_BALL_OBJ = _SCRIPT_DIR / "results" / "v3" / "yoyo-ball.obj"
_STRING_OBJ = _SCRIPT_DIR / "results" / "v3" / "yoyo_string_6153.obj"
_BEARING_OUTER_OBJ = _SCRIPT_DIR / "results" / "v1" / "bearing_outer.obj"
_BEARING_SPHERES_OBJ = _SCRIPT_DIR / "results" / "v1" / "bearing_spheres.obj"

STRING_TOP_UP_SPEED = 5.0

_user_objects: dict[str, object] = {}
_user_geo_slots: dict[str, object] = {}
_user_rest_geo_slots: dict[str, object] = {}
_original_transforms: dict[str, np.ndarray] = {}
_original_positions: dict[str, np.ndarray] = {}
_user_obj_sources: dict[str, Path] = {}
_FEM_OBJECTS = {"yoyo_string"}

def user_load_scene(scene: Scene, world: World) -> None:
    # return None
    """Add the full yoyo (ball + string + bearings) to the scene.

    Also overrides the scene config to match
    ``1_throw_yoyo_7_with_bearing.py`` (gravity kept Z-down for robot frame).

    After loading, call ``apply_global_transform`` to position the
    entire scene before ``world.init()``.
    """
    _user_objects.clear()
    _user_geo_slots.clear()
    _user_rest_geo_slots.clear()
    _original_transforms.clear()
    _original_positions.clear()
    _user_obj_sources.clear()

    io = SimplicialComplexIO()

    # ---- Sync simulation parameters with 1_throw_yoyo_7_with_bearing.py ----
    cfg = scene.config()
    view(cfg.find("dt"))[0] = 0.001
    view(cfg.find("contact/enable"))[0] = 1
    view(cfg.find("contact/friction/enable"))[0] = 1
    view(cfg.find("contact/d_hat"))[0] = 0.0001
    view(cfg.find("newton/velocity_tol"))[0] = 0.1
    view(cfg.find("newton/transrate_tol"))[0] = 10
    view(cfg.find("sanity_check/enable"))[0] = 1
    # view(cfg.find("collision_detection/method"))[0] = "stackless_bvh"

    tabular = scene.contact_tabular()
    tabular.default_model(0.3, 1e9, True)

    # ---- Contacts ----
    ball_contact = tabular.create("ball")
    string_contact = tabular.create("string")
    bearing_contact = tabular.create("bearing")
    scene.contact_tabular().insert(ball_contact, string_contact, 0.7, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(string_contact, string_contact, 0.2, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, bearing_contact, 0.01, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, ball_contact, 0.05, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, string_contact, 0.8, 800.0 * MPa, enable=True)

    # Robot contact ↔ yoyo parts
    robot_contact = tabular.default_element()
    tabular.insert(ball_contact, robot_contact, 0.5, 800.0 * MPa, True)
    tabular.insert(string_contact, robot_contact, 0.7, 800.0 * MPa, True)
    tabular.insert(bearing_contact, robot_contact, 0.3, 800.0 * MPa, True)

    # ---- Yoyo Ball (free, spinning) ----
    ball_mesh = io.read(str(_BALL_OBJ))
    label_surface(ball_mesh)
    AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
    ball_contact.apply_to(ball_mesh)
    view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 1

    motor = RotatingMotor()
    motor.apply_to(
        ball_mesh,
        strength=1,
        motor_axis=np.array([[0.0], [1.0], [0.0]], dtype=np.float64),
        motor_rot_vel=-2.0 * np.pi * 10,
    )

    ball_obj = scene.objects().create("yoyo_ball")
    ball_gs, _ = ball_obj.geometries().create(ball_mesh)
    _user_objects["yoyo_ball"] = ball_obj
    _user_geo_slots["yoyo_ball"] = ball_gs
    _user_obj_sources["yoyo_ball"] = _BALL_OBJ
    _original_transforms["yoyo_ball"] = np.array(view(ball_mesh.transforms())[0], copy=True)

    def rotate_yoyo_ball(info: Animation.UpdateInfo) -> None:
        geo = info.geo_slots()[0].geometry()
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        RotatingMotor.animate(geo, info.dt())

    scene.animator().insert(ball_obj, rotate_yoyo_ball)

    # ---- String (top vertex fixed) ----
    string_mesh = io.read(str(_STRING_OBJ))
    label_surface(string_mesh)
    HookeanSpring().apply_to(string_mesh, 1.0 * GPa, thickness=0.00038, mass_density=100.0)
    KirchhoffRodBending().apply_to(string_mesh, 1.0e4)
    # SoftPositionConstraint().apply_to(string_mesh, 100.0)
    string_contact.apply_to(string_mesh)
    mesh_partition(string_mesh, 16)
    # is_fixed = string_mesh.vertices().find(builtin.is_fixed)
    # view(is_fixed)[0] = 1

    string_obj = scene.objects().create("yoyo_string")
    string_gs, string_rest_gs = string_obj.geometries().create(string_mesh)
    _user_objects["yoyo_string"] = string_obj
    _user_geo_slots["yoyo_string"] = string_gs
    _user_rest_geo_slots["yoyo_string"] = string_rest_gs
    _user_obj_sources["yoyo_string"] = _STRING_OBJ
    _original_transforms["yoyo_string"] = np.array(view(string_mesh.transforms())[0], copy=True)
    _original_positions["yoyo_string"] = np.array(
        view(string_gs.geometry().positions()), copy=True
    ).reshape(-1, 3)

    # def pull_string_segment_up(info: Animation.UpdateInfo) -> None:
    #     geo = info.geo_slots()[0].geometry()
    #     is_constrained = view(geo.vertices().find(builtin.is_constrained))
    #     aim_position = view(geo.vertices().find(builtin.aim_position))
    #     cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

    #     is_constrained[:] = 0
    #     t_now = info.frame() * info.dt()
    #     pull_start = 1.2
    #     pull_end = 1.25
    #     if pull_start <= t_now < pull_end:
    #         max_vid = min(51, cur_positions.shape[0])
    #         for vid in range(1, max_vid):
    #             is_constrained[vid] = 1
    #             target = cur_positions[vid].copy()
    #             target[2] += STRING_TOP_UP_SPEED * info.dt()
    #             aim_position[vid] = target.reshape(3, 1)

    # scene.animator().insert(string_obj, pull_string_segment_up)

    # ---- Bearing Outer ----
    bearing_outer_mesh = io.read(str(_BEARING_OUTER_OBJ))
    label_surface(bearing_outer_mesh)
    AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
    bearing_contact.apply_to(bearing_outer_mesh)
    bearing_outer_obj = scene.objects().create("bearing_outer")
    bo_gs, _ = bearing_outer_obj.geometries().create(bearing_outer_mesh)
    _user_objects["bearing_outer"] = bearing_outer_obj
    _user_geo_slots["bearing_outer"] = bo_gs
    _user_obj_sources["bearing_outer"] = _BEARING_OUTER_OBJ
    _original_transforms["bearing_outer"] = np.array(view(bearing_outer_mesh.transforms())[0], copy=True)

    # ---- Bearing Spheres (split by region) ----
    bearing_spheres_mesh = io.read(str(_BEARING_SPHERES_OBJ))
    label_surface(bearing_spheres_mesh)
    label_region(bearing_spheres_mesh)
    sphere_regions = apply_region(bearing_spheres_mesh)
    for i, sphere_mesh in enumerate(sphere_regions):
        label_surface(sphere_mesh)
        AffineBodyConstitution().apply_to(sphere_mesh, 200.0 * MPa)
        bearing_contact.apply_to(sphere_mesh)
        sp_obj = scene.objects().create(f"bearing_sphere_{i}")
        sp_gs, _ = sp_obj.geometries().create(sphere_mesh)
        _user_objects[f"bearing_sphere_{i}"] = sp_obj
        _user_geo_slots[f"bearing_sphere_{i}"] = sp_gs
        _original_transforms[f"bearing_sphere_{i}"] = np.array(
            view(sphere_mesh.transforms())[0], copy=True
        )
        split_obj_path = _SCRIPT_DIR / "results" / "v3" / f"bearing_sphere_{i}.obj"
        if not split_obj_path.exists():
            io.write(str(split_obj_path), sphere_mesh)
        _user_obj_sources[f"bearing_sphere_{i}"] = split_obj_path

    print(f"[user_scene] Yoyo loaded: ball + string + bearing_outer + {len(sphere_regions)} bearing spheres")


def get_user_objects() -> dict[str, object]:
    """Return the dict of user-scene UIPC objects (name -> scene object)."""
    return _user_objects


def apply_global_transform(mat44: np.ndarray) -> None:
    """Apply a global 4x4 transform to all user-scene objects.

    ABD objects: ``view(geo.transforms())[0] = mat44 @ orig_transform``.
    FEM objects (string): original positions already live in ``T_orig``
    space, so the delta is ``mat44 @ inv(T_orig)`` applied to stored
    positions.
    """
    for name, gs in _user_geo_slots.items():
        orig_tf = _original_transforms.get(name, np.eye(4))
        geo = gs.geometry()
        if name in _FEM_OBJECTS:
            orig_pts = _original_positions[name]
            delta = mat44 @ np.linalg.inv(orig_tf)
            R = delta[:3, :3]
            t = delta[:3, 3]
            pts = np.array(view(geo.positions()), copy=False).reshape(-1, 3)
            pts[:] = orig_pts @ R.T + t

            rest_gs = _user_rest_geo_slots.get(name)
            if rest_gs is not None:
                rest_pts = np.array(view(rest_gs.geometry().positions()), copy=False).reshape(-1, 3)
                rest_pts[:] = orig_pts @ R.T + t
        else:
            view(geo.transforms())[0] = mat44 @ orig_tf


def stitch_string_to_gripper(
    scene: Scene,
    gripper_geo_slot: object,
    gripper_rest_geo_slot: object,
) -> None:
    """Stitch vertex 0 of the yoyo string to the nearest triangle on the gripper.

    Must be called after ``user_load_scene`` and ``create_ipc_bodies``,
    before ``world.init()``.
    """
    string_gs = _user_geo_slots.get("yoyo_string")
    string_rest_gs = _user_rest_geo_slots.get("yoyo_string")
    if string_gs is None or string_rest_gs is None:
        print("[stitch] yoyo_string not loaded, skipping")
        return

    string_geo = string_gs.geometry()
    gripper_geo = gripper_geo_slot.geometry()

    gripper_rest_geo = gripper_rest_geo_slot.geometry()
    gripper_tf = np.array(view(gripper_geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
    view(gripper_rest_geo.transforms())[0] = gripper_tf

    string_pos = np.array(view(string_geo.positions()), copy=False).reshape(-1, 3)
    gripper_world = apply_transform(gripper_geo)[0]
    gripper_world_pos = np.array(view(gripper_world.positions()), copy=False).reshape(-1, 3)
    gripper_world_tris = np.array(view(gripper_world.triangles().topo()), copy=False).reshape(-1, 3)

    v0 = string_pos[0]
    best_tri = 0
    best_dist = float("inf")
    for ti in range(gripper_world_tris.shape[0]):
        tri_verts = gripper_world_pos[gripper_world_tris[ti]]
        cp = _closest_point_on_triangle(v0, tri_verts[0], tri_verts[1], tri_verts[2])
        d = float(np.linalg.norm(v0 - cp))
        if d < best_dist:
            best_dist = d
            best_tri = ti

    pairs = np.array([[0, best_tri]], dtype=np.int32)
    svts = SoftVertexTriangleStitch()
    stitch_geo = svts.create_geometry(
        (string_gs, gripper_geo_slot),
        (string_rest_gs, gripper_rest_geo_slot),
        pairs,
        ElasticModuli.youngs_poisson(1.0e5, 0.49),
    )
    stitch_obj = scene.objects().create("string_gripper_stitch")
    stitch_obj.geometries().create(stitch_geo)

    _stitch_vis["string_gs"] = string_gs
    _stitch_vis["gripper_gs"] = gripper_geo_slot
    _stitch_vis["pairs"] = pairs
    _stitch_vis["gripper_tris"] = np.array(
        view(gripper_geo.triangles().topo()), copy=True
    ).reshape(-1, 3)

    print(f"[stitch] string vertex 0 -> gripper triangle {best_tri} (dist={best_dist:.4f})")


_stitch_vis: dict[str, object] = {}


def get_stitch_vis_data() -> dict[str, object]:
    """Return stitch visualisation data set by ``stitch_string_to_gripper``."""
    return _stitch_vis


def build_stitch_line_nodes() -> tuple[np.ndarray, np.ndarray] | None:
    """Build nodes and edges arrays for the stitch curve network.

    Returns ``(nodes (2*N, 3), edges (N, 2))`` or ``None`` if no stitch data.
    """
    if "string_gs" not in _stitch_vis:
        return None

    string_pos = np.array(
        view(_stitch_vis["string_gs"].geometry().positions()), copy=False
    ).reshape(-1, 3)
    gripper_geo = _stitch_vis["gripper_gs"].geometry()
    gripper_local = np.array(view(gripper_geo.positions()), copy=False).reshape(-1, 3)
    gripper_tris = _stitch_vis["gripper_tris"]

    tf44 = np.array(view(gripper_geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
    R = tf44[:3, :3]
    t = tf44[:3, 3]
    gripper_pos = gripper_local @ R.T + t

    pairs = _stitch_vis["pairs"]
    n = pairs.shape[0]
    if n == 0:
        return None

    nodes = np.zeros((2 * n, 3), dtype=np.float64)
    edges = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        vid, tid = int(pairs[i, 0]), int(pairs[i, 1])
        pt = string_pos[vid]
        nodes[2 * i] = pt
        tri = gripper_tris[tid]
        nodes[2 * i + 1] = _closest_point_on_triangle(
            pt, gripper_pos[tri[0]], gripper_pos[tri[1]], gripper_pos[tri[2]]
        )
        edges[i] = [2 * i, 2 * i + 1]
    return nodes, edges


def _closest_point_on_triangle(
    p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Closest point on triangle (v0,v1,v2) to point p.  Ericson §5.1.5."""
    ab = v1 - v0
    ac = v2 - v0
    ap = p - v0
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    if d1 <= 0 and d2 <= 0:
        return v0.copy()
    bp = p - v1
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    if d3 >= 0 and d4 <= d3:
        return v1.copy()
    cp = p - v2
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    if d6 >= 0 and d5 <= d6:
        return v2.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        w = d1 / (d1 - d3)
        return v0 + w * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return v0 + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return v1 + w * (v2 - v1)
    denom = 1.0 / (va + vb + vc)
    sv = vb * denom
    sw = vc * denom
    return v0 + sv * ab + sw * ac


# ---------------------------------------------------------------------------
# Sequence exporter (consolidated NPY + meta.json)
# ---------------------------------------------------------------------------

import json as _json
import shutil as _shutil


class SequenceExporter:
    """Accumulate per-frame simulation state and write consolidated NPY files.

    Usage::

        exporter = SequenceExporter(joint_names, dt=0.001)
        for frame in frames:
            exporter.capture(frame)
        exporter.save(seq_dir)
    """

    def __init__(self, joint_names: list[str], dt: float = 0.001, urdf_rel: str = ""):
        self._joint_names = list(joint_names)
        self._dt = dt
        self._urdf_rel = urdf_rel
        self._frame_ids: list[int] = []
        self._rigid_frames: dict[str, list[np.ndarray]] = {}
        self._fem_frames: dict[str, list[np.ndarray]] = {}
        self._joint_frames: list[np.ndarray] = []

    def capture(self, frame: int, joint_state: dict[str, float] | None = None) -> None:
        """Snapshot current geo_slot state and optional joint angles."""
        self._frame_ids.append(frame)

        for name, gs in _user_geo_slots.items():
            geo = gs.geometry()
            if name in _FEM_OBJECTS:
                pos = np.array(view(geo.positions()), copy=True).reshape(-1, 3)
                self._fem_frames.setdefault(name, []).append(pos)
            else:
                tf = np.array(view(geo.transforms()), copy=True).reshape(-1, 4, 4)[0]
                self._rigid_frames.setdefault(name, []).append(tf)

        if joint_state is not None:
            jv = np.array([joint_state.get(n, 0.0) for n in self._joint_names], dtype=np.float64)
            self._joint_frames.append(jv)

    def save(self, seq_dir: Path) -> None:
        """Write consolidated NPY arrays, mesh OBJs, and meta.json."""
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
            npy_name = "transforms.npy"
            np.save(str(obj_dir / npy_name), arr)
            self._copy_mesh_obj(name, obj_dir)
            meta["objects"][name] = {"type": "rigid", "data": f"{name}/{npy_name}"}

        for name, frames in self._fem_frames.items():
            obj_dir = seq_dir / name
            obj_dir.mkdir(parents=True, exist_ok=True)
            arr = np.stack(frames, axis=0)
            npy_name = "positions.npy"
            np.save(str(obj_dir / npy_name), arr)
            self._copy_mesh_obj(name, obj_dir)
            meta["objects"][name] = {"type": "fem", "data": f"{name}/{npy_name}"}

        if self._joint_frames:
            jarr = np.stack(self._joint_frames, axis=0)
            np.save(str(seq_dir / "joints.npy"), jarr)
            meta["joints"] = {
                "data": "joints.npy",
                "names": self._joint_names,
            }

        if self._urdf_rel:
            meta["urdf"] = self._urdf_rel

        (seq_dir / "meta.json").write_text(
            _json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(f"[seq-export] saved {len(self._frame_ids)} frames to {seq_dir}")

    @staticmethod
    def _copy_mesh_obj(name: str, obj_dir: Path) -> None:
        mesh_dst = obj_dir / "mesh.obj"
        if not mesh_dst.exists() and name in _user_obj_sources:
            src = _user_obj_sources[name]
            if src.exists():
                _shutil.copy2(str(src), str(mesh_dst))