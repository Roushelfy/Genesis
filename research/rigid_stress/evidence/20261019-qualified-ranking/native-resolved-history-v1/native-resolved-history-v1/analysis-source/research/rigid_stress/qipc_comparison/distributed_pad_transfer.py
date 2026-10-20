"""Work-conjugate Quadrants transfer for already resolved outer-face pad forces.

This research adapter consumes physical face pressures with a separately checked
constitutive history. It does not infer a patch or certify an instantaneous
point-source Coulomb law. All runtime force assembly stays in Quadrants.
"""

from dataclasses import dataclass

import numpy as np
import quadrants as qd
from scipy.spatial import cKDTree
import torch

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_torch


@dataclass
class ResolvedPadTransfer:
    nodes: qd.Tensor
    source_faces: qd.Tensor

    @classmethod
    def allocate(cls, outer, faces, coordinates, envs):
        if envs != 1:
            raise ValueError("The resolved pad research adapter currently requires one environment")
        triangle = outer[faces]
        samples = np.concatenate((triangle, (triangle + np.roll(triangle, -1, axis=1)) / 2), axis=1)
        distance, indices = cKDTree(coordinates).query(samples)
        assert distance.max() < 1e-14
        weights = np.array([-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9])
        np.testing.assert_allclose(np.einsum("c,fci->fi", weights, coordinates[indices]), triangle.mean(axis=1),
                                   rtol=0, atol=1e-14)
        nodes = V_VEC(6, dtype=gs.qd_int, shape=(len(faces),))
        nodes.from_numpy(indices.astype(gs.np_int, copy=False))
        source_faces = V(dtype=gs.qd_int, shape=(len(faces), envs))
        source_faces.fill(-1)
        return cls(nodes, source_faces)

    def scatter(self, selected, contacts, state):
        view = qd_to_torch(self.source_faces, transpose=True, copy=False)
        view.fill_(-1)
        view[0, :len(selected)] = torch.as_tensor(selected, device=gs.device)
        scatter_resolved_outer(self.nodes, self.source_faces, contacts, state)


@qd.kernel
def scatter_resolved_outer(
    nodes: qd.Tensor,
    source_faces: qd.Tensor,
    contacts: StressContactState,
    state: StressState,
):
    for source, env in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[source, env]:
            face = source_faces[source, env]
            force = contacts.force[source, env]
            finite = True
            for axis in qd.static(range(3)):
                finite = finite and qd.abs(force[axis]) < qd.math.inf
            contacts.status[source, env] = 1
            if face >= 0 and face < nodes.shape[0] and finite:
                contacts.status[source, env] = 0
                shape = qd.Vector([-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9])
                for corner in qd.static(range(6)):
                    node = nodes[face][corner]
                    state.force[node, env] += shape[corner] * force
