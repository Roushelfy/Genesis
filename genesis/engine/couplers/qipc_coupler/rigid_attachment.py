"""Queued triangle-mesh attachments for QIPC affine bodies.

Attachments add collision vertices to an existing Genesis rigid link without
creating another body or changing the link's authored mass properties.  This
matches fixed collision hardware such as a tape-roll ring mounted on a wheel,
while preserving stable local vertex IDs for contact regions and bond assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import genesis as gs


@dataclass(frozen=True)
class _RigidAttachmentRequest:
    entity: object
    link_local: int
    name: str
    vertices: np.ndarray
    triangles: np.ndarray


class QIPCRigidAttachment:
    """Stable handle for collision geometry appended to one QIPC rigid body."""

    __slots__ = ("_manager", "_request_index")

    def __init__(self, manager: QIPCRigidAttachmentManager, request_index: int) -> None:
        self._manager = manager
        self._request_index = request_index

    def __hash__(self) -> int:
        return hash((id(self._manager), self._request_index))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QIPCRigidAttachment)
            and other._manager is self._manager
            and other._request_index == self._request_index
        )

    @property
    def name(self) -> str:
        return self._manager._requests[self._request_index].name

    @property
    def n_vertices(self) -> int:
        return len(self._manager._requests[self._request_index].vertices)


class QIPCRigidAttachmentManager:
    """Own queued attachments and their post-init global vertex mappings."""

    def __init__(self) -> None:
        self._requests: list[_RigidAttachmentRequest] = []

    def add_request(
        self,
        entity,
        *,
        link_local: int,
        name: str,
        vertices,
        triangles,
    ) -> QIPCRigidAttachment:
        verts = np.ascontiguousarray(vertices, dtype=np.float64)
        tris = np.ascontiguousarray(triangles, dtype=np.int32)
        if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
            gs.raise_exception("QIPCCoupler.add_rigid_attachment: vertices must be finite with shape (n, 3).")
        if len(verts) == 0:
            gs.raise_exception("QIPCCoupler.add_rigid_attachment: vertices must not be empty.")
        if tris.ndim != 2 or tris.shape[1] != 3:
            gs.raise_exception("QIPCCoupler.add_rigid_attachment: triangles must have shape (m, 3).")
        if len(tris) == 0:
            gs.raise_exception("QIPCCoupler.add_rigid_attachment: triangles must not be empty.")
        if int(tris.min()) < 0 or int(tris.max()) >= len(verts):
            gs.raise_exception(
                "QIPCCoupler.add_rigid_attachment: triangle indices must reference attachment-local vertices."
            )
        if not name:
            gs.raise_exception("QIPCCoupler.add_rigid_attachment: name must not be empty.")

        request_index = len(self._requests)
        self._requests.append(
            _RigidAttachmentRequest(
                entity=entity,
                link_local=link_local,
                name=name,
                vertices=verts.copy(),
                triangles=tris.copy(),
            )
        )
        return QIPCRigidAttachment(self, request_index)

    def requests_for_group(
        self,
        entity,
        members: list[int],
    ) -> tuple[tuple[QIPCRigidAttachment, _RigidAttachmentRequest], ...]:
        member_set = set(members)
        return tuple(
            (QIPCRigidAttachment(self, index), request)
            for index, request in enumerate(self._requests)
            if request.entity is entity and request.link_local in member_set
        )

    def request(self, attachment: QIPCRigidAttachment) -> _RigidAttachmentRequest:
        if attachment._manager is not self:
            gs.raise_exception("QIPCCoupler: rigid attachment belongs to a different scene.")
        return self._requests[attachment._request_index]
