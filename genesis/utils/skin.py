"""Linear-blend-skinning runtime state.

Loaded once per skin from a ``gs.options.SkinSpec`` and stored on
``RigidEntity._skin_states``; renderers consume it to deform the skin mesh
each frame.

Math
----
For each scan vertex ``v`` and bone ``L`` indexed in ``bone_link_names``::

    M[L]   = T_live[L] @ inv(T_canon[L])
    V_t[v] = sum_L W[v, L] · M[L] · V_rest_h[v]

``V_rest_h`` is ``V_rest`` with a trailing 1 column. ``T_live[L]`` is the live
world transform of the URDF link whose name is ``bone_link_names[L]``; bones
whose link can't be resolved on the host entity fall back to identity (LBS
math tolerates this whenever the corresponding ``W`` column is zero, which is
how the typical glove rig allocates "auxiliary" bones like base/tip links).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import trimesh

import genesis as gs
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu
from genesis.utils.geom import quat_to_R


class SkinState:
    """Materialised LBS state for one ``SkinSpec`` bound to one ``RigidEntity``.

    Owns:
      - the rest-pose ``gs.Mesh`` (verts/faces/UVs + surface with material);
      - the inverse canonical bone transforms ``T_canon_inv``;
      - the sparse skin weight matrix ``W`` (CSR);
      - cached URDF link refs, one per bone, for live ``T`` lookups.
    """

    def __init__(self, spec, entity) -> None:
        self.spec = spec
        self.entity = entity
        self._parse_skin_mesh()
        self._load_canonical_link_T()
        self._load_weights()
        self._resolve_links()

    # ------------------------------------------------------------------ setup

    def _parse_skin_mesh(self) -> None:
        """Load ``spec.skin_mesh`` through Genesis's mesh parser. Embedded PBR
        material flows into ``self.skin_mesh.surface`` automatically; if the
        caller set ``spec.skin_surface`` it overrides the embedded material."""
        spec = self.spec
        surface = spec.skin_surface if spec.skin_surface is not None else gs.surfaces.Default()
        if spec.skin_mesh.lower().endswith((".glb", ".gltf")):
            meshes = gltf_utils.parse_mesh_glb(
                spec.skin_mesh,
                group_by_material=False,
                scale=1.0,
                is_mesh_zup=True,
                surface=surface,
            )
        else:
            meshes = mu.parse_mesh_trimesh(
                spec.skin_mesh,
                group_by_material=False,
                scale=1.0,
                is_mesh_zup=True,
                surface=surface,
            )
        if not meshes:
            gs.raise_exception(f"SkinSpec.skin_mesh produced no meshes: {spec.skin_mesh}")
        if len(meshes) > 1:
            gs.logger.warning(
                f"SkinSpec.skin_mesh produced {len(meshes)} sub-meshes; using the first. "
                f"Consider pre-merging materials for {spec.skin_mesh}."
            )
        self.skin_mesh = meshes[0]
        self.V_rest = np.asarray(self.skin_mesh.verts, dtype=np.float64)
        self.V_rest_h = np.concatenate([self.V_rest, np.ones((len(self.V_rest), 1))], axis=1)

    def _load_canonical_link_T(self) -> None:
        """Load the rest-pose bone transforms and pre-invert them.

        Bone order in the NPZ matches the order of ``spec.bone_link_names`` by
        position; names in the NPZ are typically the standalone URDF's
        unprefixed names while ``bone_link_names`` carries the host URDF's
        prefixed names, so we don't compare them.
        """
        canon = np.load(self.spec.canonical_link_T_file, allow_pickle=True)
        T_canon = np.asarray(canon["link_T"], dtype=np.float64)
        assert T_canon.shape[0] == len(self.spec.bone_link_names), (
            f"canonical_link_T_file n_bones={T_canon.shape[0]} doesn't match "
            f"len(bone_link_names)={len(self.spec.bone_link_names)}"
        )
        self.T_canon_inv = np.linalg.inv(T_canon)
        self.n_bones = T_canon.shape[0]

    def _load_weights(self) -> None:
        W_sparse = scipy.sparse.load_npz(self.spec.weights_file)
        if W_sparse.shape != (len(self.V_rest), self.n_bones):
            raise ValueError(
                f"weights_file shape {W_sparse.shape} doesn't match skin V={len(self.V_rest)} or n_bones={self.n_bones}"
            )
        self.W = W_sparse.tocsr()

    def _resolve_links(self) -> None:
        """Resolve each bone's local link index on the host entity. Storing
        indices (not Link refs) means ``lbs_mesh`` can pull every bone's
        transform in a single batched solver call — one GPU→CPU sync per
        frame instead of ``2 * n_bones``. Bones whose link is missing get
        index ``-1`` and fall back to identity ``T_live`` (safe whenever the
        corresponding ``W`` column is zero)."""
        idx = np.full(self.n_bones, -1, dtype=np.int64)
        for i, name in enumerate(self.spec.bone_link_names):
            try:
                idx[i] = self.entity.get_link(name).idx_local
            except Exception:
                pass
        self._bone_link_idx_local = idx
        self._resolved_mask = idx >= 0
        self._resolved_idx_local = idx[self._resolved_mask].tolist()
        self._n_resolved = int(self._resolved_mask.sum())

    # --------------------------------------------------------------- per frame

    def lbs_mesh(self) -> trimesh.Trimesh:
        """Return a fresh ``trimesh.Trimesh`` carrying the LBS-deformed verts,
        the static rest-pose faces, and (lazily) per-vertex normals derived
        from the current verts.

        Renderers consume this the same way they consume the output of
        ``particles_to_mesh`` in the recon path:
          * pyrender reads ``mesh.vertices`` and lets ``jit.update_normal``
            recompute normals on the GPU — ``vertex_normals`` is never
            evaluated.
          * Luisa reads ``mesh.vertex_normals`` (triggering trimesh's lazy
            compute) and passes them through ``update_deformable``.
        """
        T_live = np.tile(np.eye(4), (self.n_bones, 1, 1))
        if self._n_resolved:
            # get_links_pos/get_links_quat always return torch.Tensor — one
            # cpu()-sync per call instead of `2 * n_bones` from per-link reads.
            pos_np = self.entity.get_links_pos(links_idx_local=self._resolved_idx_local).cpu().numpy()
            quat_np = self.entity.get_links_quat(links_idx_local=self._resolved_idx_local).cpu().numpy()
            pos_np = np.asarray(pos_np, dtype=np.float64).reshape(self._n_resolved, 3)
            quat_np = np.asarray(quat_np, dtype=np.float64).reshape(self._n_resolved, 4)
            T_live[self._resolved_mask, :3, :3] = quat_to_R(quat_np)
            T_live[self._resolved_mask, :3, 3] = pos_np

        M = T_live @ self.T_canon_inv  # (n_bones, 4, 4)
        M_flat = M.reshape(self.n_bones, 16)
        per_v_T = (self.W @ M_flat).reshape(-1, 4, 4)  # (V, 4, 4)
        V_t = np.einsum("vij,vj->vi", per_v_T, self.V_rest_h)[:, :3]

        return trimesh.Trimesh(
            vertices=V_t.astype(np.float32),
            faces=self.skin_mesh.faces,
            process=False,
        )
