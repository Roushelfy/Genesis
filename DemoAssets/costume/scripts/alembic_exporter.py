"""Alembic / OBJ-sequence exporter for UIPC simulation data.

Captures animated mesh geometry produced by UIPC simulations and writes it to
Alembic (``.abc``) files that can be loaded in DCC applications such as Houdini,
Maya, and Blender.  When the Alembic Python bindings are not available the
exporter falls back to per-frame OBJ files that can be loaded in the same tools.

Two geometry types are supported:

* **Deformable** -- vertex positions change every frame (cloth, strings).
* **Rigid / ABD** -- a 4x4 transform changes every frame; the local mesh is
  constant (robot links, rigid objects).

Installation -- PyAlembic
-------------------------
The Alembic Python bindings (``alembic.Abc``, ``alembic.AbcGeom``, ``imath``)
are **not** available via ``pip``.  You can obtain them by:

1. Building from source with ``-DUSE_PYALEMBIC=ON``
   (https://github.com/alembic/alembic).
2. Using the Python shipped with a DCC application (Houdini and Maya include
   PyAlembic out of the box).

Quick-start example
-------------------
::

    from alembic_exporter import create_exporter, read_obj_faces

    exporter = create_exporter("output/animation.abc", fps=100.0)

    # Register a deformable cloth mesh (faces parsed from the OBJ file)
    faces = read_obj_faces("cloth_init.obj")
    exporter.add_deformable("cloth", cloth_geo_slot, faces)

    # Register a rigid robot link (faces from MeshNode)
    exporter.add_rigid("link_0", link_geo_slot, node.local_vertices, node.faces)

    # Main simulation loop
    for _ in range(num_frames):
        world.advance()
        world.retrieve()
        exporter.capture_frame()

    exporter.close()

The ``create_exporter`` factory automatically picks ``AlembicExporter`` when
PyAlembic is importable, otherwise it returns an ``ObjSequenceExporter``.
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from alembic.Abc import OArchive, TimeSampling  # type: ignore[import-untyped]
    from alembic.AbcGeom import (  # type: ignore[import-untyped]
        OPolyMesh,
        OPolyMeshSchemaSample,
        OXform,
        XformSample,
    )
    from imath import M44d, V3fArray  # type: ignore[import-untyped]
    from imath import IntArray as ImathIntArray  # type: ignore[import-untyped]

    HAS_ALEMBIC = True
except ImportError:
    HAS_ALEMBIC = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def read_obj_faces(path: str | Path) -> np.ndarray:
    """Parse triangle face indices from a Wavefront OBJ file.

    Only the first three vertex indices of each ``f`` line are used (quads and
    higher-order polygons are triangulated by taking the first three vertices).
    Vertex-index-only and ``v/vt/vn`` notations are both handled.

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
    """Read current vertex positions from a UIPC geometry slot."""
    from uipc import view  # deferred so the module can be imported without uipc

    return np.array(view(geo_slot.geometry().positions()), copy=False).reshape(-1, 3)


def _read_transform(geo_slot: Any) -> np.ndarray:
    """Read the current 4x4 instance transform from a UIPC geometry slot."""
    from uipc import view

    return np.array(view(geo_slot.geometry().transforms()), copy=False).reshape(-1, 4, 4)[0]


# ---------------------------------------------------------------------------
# Internal data holders
# ---------------------------------------------------------------------------


@dataclass
class _DeformableEntry:
    name: str
    geo_slot: Any
    faces: np.ndarray
    first_sample: bool = True


@dataclass
class _RigidEntry:
    name: str
    geo_slot: Any
    local_vertices: np.ndarray
    faces: np.ndarray
    first_sample: bool = True


# ---------------------------------------------------------------------------
# BaseExporter
# ---------------------------------------------------------------------------


class BaseExporter(abc.ABC):
    """Abstract base for simulation exporters."""

    @abc.abstractmethod
    def add_deformable(self, name: str, geo_slot: Any, faces: np.ndarray) -> None:
        """Register a deformable mesh whose vertex positions animate.

        Parameters
        ----------
        name : str
            Unique identifier for this object in the output.
        geo_slot : object
            UIPC geometry slot (returned by ``obj.geometries().create(...)``).
        faces : np.ndarray
            ``(M, 3)`` int32 triangle indices (zero-based).
        """

    @abc.abstractmethod
    def add_rigid(
        self,
        name: str,
        geo_slot: Any,
        local_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Register a rigid body whose 4x4 transform animates.

        Parameters
        ----------
        name : str
            Unique identifier for this object in the output.
        geo_slot : object
            UIPC geometry slot.
        local_vertices : np.ndarray
            ``(N, 3)`` float64 vertex positions in local/body space.
        faces : np.ndarray
            ``(M, 3)`` int32 triangle indices (zero-based).
        """

    @abc.abstractmethod
    def capture_frame(self) -> None:
        """Write one time-sample for every registered object."""

    @abc.abstractmethod
    def close(self) -> None:
        """Finalize and flush all output."""

    @property
    @abc.abstractmethod
    def frame_count(self) -> int:
        """Number of frames captured so far."""

    # Context-manager protocol ------------------------------------------------

    def __enter__(self) -> "BaseExporter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# AlembicExporter
# ---------------------------------------------------------------------------


class AlembicExporter(BaseExporter):
    """Write UIPC simulation data to a single Alembic (``.abc``) file.

    Each deformable object becomes an ``OXform`` (identity) containing an
    ``OPolyMesh`` with animated vertex positions.  Each rigid object becomes an
    ``OXform`` with animated 4x4 transforms containing an ``OPolyMesh`` whose
    vertices are written once (constant topology *and* constant local
    positions).

    Parameters
    ----------
    path : str or Path
        Destination ``.abc`` file path.
    fps : float
        Frames per second (determines Alembic ``TimeSampling``).
    """

    def __init__(self, path: str | Path, fps: float = 24.0) -> None:
        if not HAS_ALEMBIC:
            raise ImportError(
                "Alembic Python bindings not found.\n"
                "Install via one of:\n"
                "  - Build from source with -DUSE_PYALEMBIC=ON\n"
                "    https://github.com/alembic/alembic\n"
                "  - Use the Python shipped with Houdini / Maya"
            )
        self._path = str(path)
        self._fps = float(fps)
        self._archive = OArchive(self._path)

        ts = TimeSampling(1.0 / self._fps, 0.0)
        self._ts_idx = self._archive.addTimeSampling(ts)

        self._deformables: list[_DeformableEntry] = []
        self._rigids: list[_RigidEntry] = []
        self._abc_deformable_schemas: list[Any] = []
        self._abc_rigid_schemas: list[tuple[Any, Any]] = []  # (xform_schema, mesh_schema)
        self._abc_objects: list[Any] = []  # prevent GC of OXform / OPolyMesh
        self._frame_count = 0
        self._closed = False

        log.info("AlembicExporter: writing to %s at %.1f fps", self._path, self._fps)

    # -- Registration ---------------------------------------------------------

    def add_deformable(self, name: str, geo_slot: Any, faces: np.ndarray) -> None:
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        entry = _DeformableEntry(name=name, geo_slot=geo_slot, faces=faces)
        self._deformables.append(entry)

        top = self._archive.getTop()
        xform_obj = OXform(top, name)
        xform_obj.getSchema().setTimeSampling(self._ts_idx)
        mesh_obj = OPolyMesh(xform_obj, f"{name}_mesh")
        mesh_schema = mesh_obj.getSchema()
        mesh_schema.setTimeSampling(self._ts_idx)

        self._abc_objects.extend([xform_obj, mesh_obj])
        self._abc_deformable_schemas.append(mesh_schema)
        log.info("  + deformable '%s' (%d faces)", name, faces.shape[0])

    def add_rigid(
        self,
        name: str,
        geo_slot: Any,
        local_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        local_vertices = np.asarray(local_vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        entry = _RigidEntry(
            name=name,
            geo_slot=geo_slot,
            local_vertices=local_vertices,
            faces=faces,
        )
        self._rigids.append(entry)

        top = self._archive.getTop()
        xform_obj = OXform(top, name)
        xform_schema = xform_obj.getSchema()
        xform_schema.setTimeSampling(self._ts_idx)

        mesh_obj = OPolyMesh(xform_obj, f"{name}_mesh")
        mesh_schema = mesh_obj.getSchema()

        self._abc_objects.extend([xform_obj, mesh_obj])
        self._abc_rigid_schemas.append((xform_schema, mesh_schema))
        log.info("  + rigid '%s' (%d verts, %d faces)", name, local_vertices.shape[0], faces.shape[0])

    # -- Per-frame capture ----------------------------------------------------

    @staticmethod
    def _to_v3f_array(positions: np.ndarray) -> Any:
        """Convert (N, 3) float array to imath.V3fArray."""
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        n = positions.shape[0]
        arr = V3fArray(n)
        for i in range(n):
            arr[i] = (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))
        return arr

    @staticmethod
    def _face_arrays(faces: np.ndarray) -> tuple[Any, Any]:
        """Convert (M, 3) face indices to Alembic (indices, counts) IntArrays."""
        flat = faces.ravel()
        indices = ImathIntArray(len(flat))
        for i, v in enumerate(flat):
            indices[i] = int(v)
        counts = ImathIntArray(faces.shape[0])
        for i in range(faces.shape[0]):
            counts[i] = 3
        return indices, counts

    @staticmethod
    def _to_m44d(mat: np.ndarray) -> Any:
        """Convert a (4, 4) numpy array to imath.M44d (row-major)."""
        m = np.asarray(mat, dtype=np.float64).reshape(4, 4)
        return M44d(
            (m[0, 0], m[0, 1], m[0, 2], m[0, 3]),
            (m[1, 0], m[1, 1], m[1, 2], m[1, 3]),
            (m[2, 0], m[2, 1], m[2, 2], m[2, 3]),
            (m[3, 0], m[3, 1], m[3, 2], m[3, 3]),
        )

    def capture_frame(self) -> None:
        if self._closed:
            raise RuntimeError("Exporter is already closed")

        for entry, mesh_schema in zip(self._deformables, self._abc_deformable_schemas):
            positions = _read_positions(entry.geo_slot)
            verts = self._to_v3f_array(positions)
            indices, counts = self._face_arrays(entry.faces)
            mesh_schema.set(OPolyMeshSchemaSample(verts, indices, counts))

        for entry, (xform_schema, mesh_schema) in zip(self._rigids, self._abc_rigid_schemas):
            xf = _read_transform(entry.geo_slot)
            xform_samp = XformSample()
            xform_samp.setMatrix(self._to_m44d(xf))
            xform_schema.set(xform_samp)

            if entry.first_sample:
                verts = self._to_v3f_array(entry.local_vertices)
                indices, counts = self._face_arrays(entry.faces)
                mesh_sample = OPolyMeshSchemaSample(verts, indices, counts)
                mesh_schema.set(mesh_sample)
                entry.first_sample = False

        self._frame_count += 1

    # -- Finalization ---------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # OArchive is finalized when the Python object is garbage-collected.
        # Drop all Alembic references so the file is flushed to disk.
        self._abc_deformable_schemas.clear()
        self._abc_rigid_schemas.clear()
        self._abc_objects.clear()
        del self._archive
        log.info("AlembicExporter: closed %s (%d frames)", self._path, self._frame_count)


# ---------------------------------------------------------------------------
# ObjSequenceExporter
# ---------------------------------------------------------------------------


class ObjSequenceExporter(BaseExporter):
    """Fallback exporter that writes per-frame Wavefront OBJ files.

    Output layout::

        output_dir/
            metadata.json
            cloth_0/
                frame_000000.obj
                frame_000001.obj
                ...
            robot_link_0/
                frame_000000.obj
                ...

    Houdini can load these via a *File SOP* with the path expression
    ``output_dir/cloth_0/frame_$F6.obj``.

    Parameters
    ----------
    output_dir : str or Path
        Directory that will contain the exported sequence folders.
    fps : float
        Stored in ``metadata.json`` for reference.
    """

    def __init__(self, output_dir: str | Path, fps: float = 24.0) -> None:
        self._output_dir = Path(output_dir)
        self._fps = float(fps)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._deformables: list[_DeformableEntry] = []
        self._rigids: list[_RigidEntry] = []
        self._frame_count = 0
        self._closed = False

        log.info("ObjSequenceExporter: writing to %s at %.1f fps", self._output_dir, self._fps)

    # -- Registration ---------------------------------------------------------

    def add_deformable(self, name: str, geo_slot: Any, faces: np.ndarray) -> None:
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        entry = _DeformableEntry(name=name, geo_slot=geo_slot, faces=faces)
        self._deformables.append(entry)
        (self._output_dir / name).mkdir(parents=True, exist_ok=True)
        log.info("  + deformable '%s' (%d faces)", name, faces.shape[0])

    def add_rigid(
        self,
        name: str,
        geo_slot: Any,
        local_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        local_vertices = np.asarray(local_vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        entry = _RigidEntry(
            name=name,
            geo_slot=geo_slot,
            local_vertices=local_vertices,
            faces=faces,
        )
        self._rigids.append(entry)
        (self._output_dir / name).mkdir(parents=True, exist_ok=True)
        log.info("  + rigid '%s' (%d verts, %d faces)", name, local_vertices.shape[0], faces.shape[0])

    # -- Per-frame capture ----------------------------------------------------

    @staticmethod
    def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Write a minimal Wavefront OBJ file."""
        with open(path, "w", encoding="utf-8") as fh:
            for v in vertices:
                fh.write(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
            for f in faces:
                fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    def capture_frame(self) -> None:
        if self._closed:
            raise RuntimeError("Exporter is already closed")

        frame_label = f"frame_{self._frame_count:06d}.obj"

        for entry in self._deformables:
            positions = _read_positions(entry.geo_slot)
            self._write_obj(self._output_dir / entry.name / frame_label, positions, entry.faces)

        for entry in self._rigids:
            xf = _read_transform(entry.geo_slot)
            world_verts = entry.local_vertices @ xf[:3, :3].T + xf[:3, 3]
            self._write_obj(self._output_dir / entry.name / frame_label, world_verts, entry.faces)

        self._frame_count += 1

    # -- Finalization ---------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        all_names = [e.name for e in self._deformables] + [e.name for e in self._rigids]
        metadata = {
            "fps": self._fps,
            "frame_count": self._frame_count,
            "objects": [
                {
                    "name": e.name,
                    "type": "deformable",
                    "num_faces": int(e.faces.shape[0]),
                }
                for e in self._deformables
            ]
            + [
                {
                    "name": e.name,
                    "type": "rigid",
                    "num_vertices": int(e.local_vertices.shape[0]),
                    "num_faces": int(e.faces.shape[0]),
                }
                for e in self._rigids
            ],
        }
        meta_path = self._output_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        log.info(
            "ObjSequenceExporter: closed %s (%d frames, %d objects)",
            self._output_dir,
            self._frame_count,
            len(all_names),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_exporter(path: str | Path, fps: float = 24.0) -> BaseExporter:
    """Create the best available exporter for *path*.

    * If *path* ends with ``.abc`` **and** PyAlembic is importable, returns an
      :class:`AlembicExporter`.
    * Otherwise returns an :class:`ObjSequenceExporter` (treating *path* as
      the output directory).
    """
    p = Path(path)
    if p.suffix.lower() == ".abc" and HAS_ALEMBIC:
        p.parent.mkdir(parents=True, exist_ok=True)
        return AlembicExporter(p, fps=fps)

    if p.suffix.lower() == ".abc" and not HAS_ALEMBIC:
        fallback_dir = p.with_suffix("")
        log.warning(
            "PyAlembic is not available -- falling back to OBJ sequence export in %s",
            fallback_dir,
        )
        return ObjSequenceExporter(fallback_dir, fps=fps)

    return ObjSequenceExporter(p, fps=fps)
