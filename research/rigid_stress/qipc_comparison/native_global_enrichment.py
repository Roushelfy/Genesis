"""Integer-only global coarse controls through the existing fixed_nodes asset.

No material coordinate, force, stiffness or constitutive value is prepared
here. Production numerical prolongation and operators remain in Quadrants.
"""

from pathlib import Path

from .artifacts import file_id


def add_global_enrichment_argument(parser):
    parser.add_argument(
        "--global-enrichment",
        choices=("none", "surface"),
        default="none",
        help="Promote all shell face centers or complete material-boundary volume P2 midpoints through fixed_nodes.",
    )


def _integer_vector(np, values, name):
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer node list.")
    if (values < 0).any():
        raise ValueError(f"{name} must contain nonnegative node IDs.")
    return np.unique(values.astype(np.int64, copy=False))


def _boundary_components(edges):
    parents = {int(node): int(node) for edge in edges for node in edge}

    def root(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for first, second in edges:
        a, b = root(int(first)), root(int(second))
        if a != b:
            parents[b] = a
    return len({root(node) for node in parents})


def surface_fixed_nodes(np, backend, mesh, parent_nodes=None):
    """Return fixed physical IDs using authored parentage or complete tet faces."""
    if backend == "shell":
        if (
            parent_nodes is None
            or parent_nodes.ndim != 2
            or parent_nodes.shape[1] != 3
            or not np.issubdtype(parent_nodes.dtype, np.integer)
        ):
            raise ValueError("Shell global enrichment requires integer authored parent_nodes[n,3].")
        count = np.count_nonzero(parent_nodes >= 0, axis=1)
        selected = np.flatnonzero(count == 3).astype(np.int64)
        if not len(selected):
            raise ValueError("No authored three-parent shell face centers are available for this coarse control.")
        return selected, dict(
            kind="all_authored_shell_three_parent_face_centers",
            physical_nodes=len(parent_nodes),
            promoted_nodes=len(selected),
            derivation="Integer parent_count == 3; no positions or selected contacts.",
        )
    if backend != "volume":
        raise ValueError("Global enrichment backend must be shell or volume.")
    with np.load(mesh, allow_pickle=False) as asset:
        tetrahedra = asset["tetrahedra"]
        selected_surface = asset["surface_triangles"]
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or not len(tetrahedra)
        or not np.issubdtype(tetrahedra.dtype, np.integer)
    ):
        raise ValueError("Volume global enrichment requires integer P1 tetrahedra[n,4].")
    tetrahedra = tetrahedra.astype(np.int64, copy=False)
    n_original = int(tetrahedra.max()) + 1
    if not np.array_equal(np.unique(tetrahedra), np.arange(n_original)):
        raise ValueError("Native volume P1 vertices must be contiguous and all referenced by tetrahedra.")
    local_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)
    edges = np.unique(np.sort(tetrahedra[:, local_edges].reshape((-1, 2)), axis=1), axis=0)
    faces, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
    if (counts > 2).any():
        raise ValueError("Nonmanifold tetrahedron faces do not define this complete material-boundary control.")
    boundary = faces[counts == 1]
    boundary_edges = np.unique(np.sort(boundary[:, [[0, 1], [0, 2], [1, 2]]].reshape((-1, 2)), axis=1), axis=0)
    keys = edges[:, 0] * n_original + edges[:, 1]
    boundary_keys = boundary_edges[:, 0] * n_original + boundary_edges[:, 1]
    indices = np.searchsorted(keys, boundary_keys)
    if (indices >= len(edges)).any() or not np.array_equal(edges[indices], boundary_edges):
        raise ValueError("Complete boundary edges do not map to the native global P2 edge ordering.")
    fixed = n_original + indices
    return fixed.astype(np.int64), dict(
        kind="all_complete_material_boundary_volume_p2_midpoints",
        original_p1_nodes=n_original,
        physical_nodes=n_original + len(edges),
        promoted_nodes=len(fixed),
        complete_material_boundary_faces=len(boundary),
        complete_boundary_components=_boundary_components(boundary_edges),
        asset_selected_surface_faces=len(selected_surface),
        uses_selected_surface_to_choose_nodes=False,
        derivation="All tet faces with incidence one, then all their edges; native lexicographic unique-edge P2 IDs. Inner/outer/nonconvex boundary identities follow integer material topology.",
    )


def write_global_basis(
    np,
    path,
    *,
    backend,
    mesh,
    mode,
    parent_nodes=None,
    fixed_nodes=None,
    existing_basis=None,
):
    """Merge one research global control into a fresh compatible basis asset."""
    if mode not in ("none", "surface"):
        raise ValueError("Unknown global enrichment mode.")
    inherited = np.empty(0, dtype=np.int64)
    existing_identity = None
    if existing_basis is not None:
        existing_basis = Path(existing_basis)
        existing_identity = file_id(existing_basis)
        with np.load(existing_basis, allow_pickle=False) as saved:
            if "parent_nodes" in saved:
                parent_nodes = saved["parent_nodes"]
            if "fixed_nodes" in saved:
                inherited = _integer_vector(np, saved["fixed_nodes"], "existing fixed_nodes")
    if fixed_nodes is not None:
        inherited = np.union1d(inherited, _integer_vector(np, fixed_nodes, "fixed_nodes"))
    promoted, description = (np.empty(0, dtype=np.int64), dict(kind="none", promoted_nodes=0))
    if mode == "surface":
        promoted, description = surface_fixed_nodes(np, backend, mesh, parent_nodes)
    selected = np.union1d(inherited, promoted).astype(np.int64, copy=False)
    arrays = dict(fixed_nodes=selected, global_enrichment=np.array(mode), global_enrichment_fixed_nodes=promoted)
    if backend == "shell":
        if parent_nodes is None:
            raise ValueError("Shell basis assets require authored parent_nodes.")
        arrays["parent_nodes"] = parent_nodes
        physical_nodes = len(parent_nodes)
    elif mode == "surface":
        physical_nodes = description["physical_nodes"]
    else:
        physical_nodes = None
    if physical_nodes is not None and (selected >= physical_nodes).any():
        raise ValueError("Global/fixed enrichment IDs exceed the unchanged physical mother node count.")
    np.savez(path, **arrays)
    return dict(
        mode=mode,
        description=description,
        existing_basis=existing_identity,
        inherited_fixed_count=len(inherited),
        global_promoted_count=len(promoted),
        merged_fixed_count=len(selected),
        fixed_nodes=selected.tolist(),
        basis=file_id(path),
        production_interface="Existing atlas_basis fixed_nodes; no new public production option or physical mesh.",
    )


def existing_basis_enrichment(np, basis):
    """Expose a reused basis tag without deriving anything from floating values."""
    with np.load(basis, allow_pickle=False) as saved:
        fixed_count = len(saved["fixed_nodes"]) if "fixed_nodes" in saved else 0
        mode = (
            str(saved["global_enrichment"])
            if "global_enrichment" in saved
            else "unspecified_fixed_nodes"
            if fixed_count
            else "none"
        )
    return dict(mode=mode, requested_mode="none", inherited_basis=True, fixed_count=fixed_count, basis=file_id(basis))


def actual_enrichment_space(atlas):
    metadata = atlas.metadata()
    groups = metadata.groups[1:]
    coarse_full = len(atlas.basis.coarse_nodes) == len(atlas.basis.parent_nodes)
    reduced = sum(group.blocks < metadata.mother_blocks for group in groups)
    return dict(
        physical_nodes=len(atlas.basis.parent_nodes),
        coarse_physical_nodes=len(atlas.basis.coarse_nodes),
        global_coarse_equals_mother=coarse_full,
        reduced_feature_groups=reduced,
        full_size_feature_groups=sum(group.blocks == metadata.mother_blocks for group in groups),
        feature_raw_scalar_dofs=[3 * group.blocks for group in groups],
        mother_raw_scalar_dofs=3 * metadata.mother_blocks,
        role="full_mother_control" if coarse_full or reduced == 0 else "native_reduced_candidate",
        genuine_reduction=reduced > 0,
    )
