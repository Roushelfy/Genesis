"""Invertible elastic/rigid coordinate basis for the same free Panda reference."""

import numpy as np
from scipy import sparse


def coordinate_map(system, length_scale=0.02):
    """Separate stiff elastic and six weak rigid coordinates without constraints.

    The six QR-selected displacement rows define only the elastic basis. All
    six rigid modes are restored as independent columns, so this is an
    invertible change of coordinates, not a supported-body approximation.
    Rotations are scaled by a declared physical length to retain metre units.
    """
    count = system.body.basis.N
    permutation = system.parts[0].permutation
    held = np.flatnonzero(np.isin(permutation, system.body.gauge_dofs))
    independent = np.setdiff1d(np.arange(count), held)
    modes = system.body.rigid_modes[permutation].copy()
    modes[:, 3:] /= length_scale
    assert np.linalg.matrix_rank(modes[held], tol=1e-10) == 6
    elastic = sparse.coo_matrix(
        (np.ones(len(independent)), (independent, np.arange(len(independent)))), shape=(count, len(independent))
    ).tocsr()
    body_map = sparse.hstack((elastic, sparse.csr_matrix(modes)), format="csr")
    free = system.coordinate_map(False)[0]
    body_full = sparse.vstack((body_map, sparse.csr_matrix((free.shape[0] - count, count))), format="csr")
    mapping = sparse.hstack((body_full, free[:, -2:]), format="csr")
    assert mapping.shape == free.shape
    # Recover an arbitrary full body vector through the six selected rows.
    rng = np.random.default_rng(20261014)
    original = rng.normal(size=count)
    rigid = np.linalg.solve(modes[held], original[held])
    coordinates = np.r_[original[independent] - modes[independent] @ rigid, rigid]
    np.testing.assert_allclose(body_map @ coordinates, original, rtol=1e-12, atol=1e-12)
    return mapping
