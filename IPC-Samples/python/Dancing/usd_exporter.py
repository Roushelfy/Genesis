"""USD exporter for UIPC simulation data.

Writes animated mesh geometry to a USD (``.usdc``) file.

Two geometry types are supported:

* **Deformable** -- vertex positions change every frame (cloth, strings).
  Written as a ``UsdGeom.Mesh`` with time-sampled ``points``.
* **Rigid / ABD** -- a 4x4 transform changes every frame; the local mesh is
  constant (robot links, rigid objects).
  Written as a ``UsdGeom.Xform`` with time-sampled transform, containing a
  ``UsdGeom.Mesh`` with static ``points``.

Usage::

    from usd_exporter import UsdExporter, read_obj_faces

    exporter = UsdExporter("output/animation.usdc", fps=100.0)
    faces = read_obj_faces("cloth_init.obj")
    exporter.add_deformable("cloth", cloth_geo_slot, faces)
    exporter.add_rigid("link_0", link_geo_slot, node.local_vertices, node.faces)

    for _ in range(num_frames):
        world.recover(frame)
        world.retrieve()
        exporter.capture_frame()

    exporter.close()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def read_obj_faces(path: str | Path) -> np.ndarray:
    """Parse triangle face indices from a Wavefront OBJ file.

    Returns an ``(M, 3)`` int32 array of zero-based face indices.
    """
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or not line.startswith("f "):
                continue
            parts = line.split()[1:]
            if len(parts) < 3:
                continue
            tri = [int(p.split("/")[0]) - 1 for p in parts[:3]]
            faces.append(tri)
    if not faces:
        raise ValueError(f"No faces found in {path}")
    return np.asarray(faces, dtype=np.int32)


def _read_positions(geo_slot: Any) -> np.ndarray:
    from uipc import view

    return np.array(view(geo_slot.geometry().positions()), copy=False).reshape(-1, 3)


def _read_transform(geo_slot: Any) -> np.ndarray:
    from uipc import view

    return np.array(view(geo_slot.geometry().transforms()), copy=False).reshape(-1, 4, 4)[0]


def _sanitize_name(name: str) -> str:
    """Make a string safe for use as a USD prim name."""
    out = name.replace("-", "_").replace(".", "_").replace(" ", "_")
    if out and out[0].isdigit():
        out = "_" + out
    return out


def _np_to_vt_vec3f(arr: np.ndarray) -> Vt.Vec3fArray:
    pts = arr.astype(np.float32)
    return Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])


def _faces_to_vt(faces: np.ndarray) -> tuple[Vt.IntArray, Vt.IntArray]:
    """Convert (M, 3) face indices to USD faceVertexIndices + faceVertexCounts."""
    face_counts = Vt.IntArray([3] * len(faces))
    face_indices = Vt.IntArray(faces.flatten().tolist())
    return face_indices, face_counts


def _np44_to_gf_matrix(m: np.ndarray) -> Gf.Matrix4d:
    """Convert a UIPC 4x4 transform (column-vector) to USD Gf.Matrix4d (row-vector).

    UIPC uses column-vector convention (translation in last column),
    USD uses row-vector convention (translation in last row), so we transpose.
    """
    t = m.T
    return Gf.Matrix4d(
        t[0][0], t[0][1], t[0][2], t[0][3],
        t[1][0], t[1][1], t[1][2], t[1][3],
        t[2][0], t[2][1], t[2][2], t[2][3],
        t[3][0], t[3][1], t[3][2], t[3][3],
    )


@dataclass
class _DeformableEntry:
    name: str
    geo_slot: Any
    faces: np.ndarray
    mesh_prim: Any = None


@dataclass
class _RigidEntry:
    name: str
    geo_slot: Any
    local_vertices: np.ndarray
    faces: np.ndarray
    xform_prim: Any = None


class UsdExporter:
    """Write UIPC simulation data to a USD file.

    Parameters
    ----------
    path : str | Path
        Destination ``.usdc`` / ``.usda`` / ``.usd`` file path.
    fps : float
        Frames per second for the time codes.
    up_axis : str
        Up axis, ``"Z"`` or ``"Y"``.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 100.0,
        up_axis: str = "Z",
        root_prim: str = "",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._frame_count = 0
        self._root = f"/{_sanitize_name(root_prim)}" if root_prim else ""

        self._stage = Usd.Stage.CreateNew(str(self._path))
        self._stage.SetMetadata("metersPerUnit", 1.0)
        self._stage.SetStartTimeCode(0)
        self._stage.SetTimeCodesPerSecond(fps)
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)

        if self._root:
            UsdGeom.Xform.Define(self._stage, self._root)

        self._deformables: list[_DeformableEntry] = []
        self._rigids: list[_RigidEntry] = []
        print(f"[usd] creating {self._path} at {fps} fps (root={self._root or '/'})")

    def add_deformable(self, name: str, geo_slot: Any, faces: np.ndarray) -> None:
        safe = _sanitize_name(name)
        prim_path = f"{self._root}/deformable/{safe}"
        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        face_indices, face_counts = _faces_to_vt(faces)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        entry = _DeformableEntry(name=name, geo_slot=geo_slot, faces=faces, mesh_prim=mesh)
        self._deformables.append(entry)
        print(f"[usd] added deformable '{safe}': {len(faces)} faces")

    def add_rigid(
        self,
        name: str,
        geo_slot: Any,
        local_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        safe = _sanitize_name(name)
        prim_path = f"{self._root}/rigid/{safe}"
        xform = UsdGeom.Xform.Define(self._stage, prim_path)
        mesh_path = f"{prim_path}/mesh"
        mesh = UsdGeom.Mesh.Define(self._stage, mesh_path)

        verts = _np_to_vt_vec3f(local_vertices)
        mesh.GetPointsAttr().Set(verts)
        face_indices, face_counts = _faces_to_vt(faces)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        mesh.GetFaceVertexCountsAttr().Set(face_counts)

        entry = _RigidEntry(
            name=name, geo_slot=geo_slot,
            local_vertices=local_vertices, faces=faces,
            xform_prim=xform,
        )
        self._rigids.append(entry)
        print(f"[usd] added rigid '{safe}': {len(local_vertices)} verts, {len(faces)} faces")

    def capture_frame(self) -> None:
        tc = float(self._frame_count)

        for entry in self._deformables:
            pos = _read_positions(entry.geo_slot)
            verts = _np_to_vt_vec3f(pos)
            entry.mesh_prim.GetPointsAttr().Set(verts, tc)

        for entry in self._rigids:
            tf = _read_transform(entry.geo_slot)
            gf_mat = _np44_to_gf_matrix(tf)
            xform_op = entry.xform_prim.MakeMatrixXform()
            xform_op.Set(gf_mat, tc)

        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._frame_count > 0:
            self._stage.SetEndTimeCode(float(self._frame_count - 1))
        self._stage.GetRootLayer().Save()
        print(f"[usd] saved {self._path} ({self._frame_count} frames)")

    def __enter__(self) -> "UsdExporter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
