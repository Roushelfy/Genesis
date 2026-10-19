"""Independently assemble and directly solve a captured Shell contact linear system."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .artifacts import file_id, runtime_identity, write_json


def scatter_blocks(indices, blocks, n_dofs):
    dofs = (3 * indices[:, :, None] + np.arange(3)).reshape(len(indices), -1)
    rows = np.broadcast_to(dofs[:, :, None], blocks.shape)
    columns = np.broadcast_to(dofs[:, None, :], blocks.shape)
    return sparse.coo_matrix(
        (blocks.reshape(-1), (rows.reshape(-1), columns.reshape(-1))), shape=(n_dofs, n_dofs)
    ).tocsr()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    saved = np.load(args.field)
    receipt = json.loads(args.field.with_suffix(".json").read_text())
    mass, rhs, iterate = saved["mass"], saved["rhs"].reshape(-1), saved["solution"].reshape(-1)
    faces, f, y, stress = (saved[key] for key in ("faces", "face_f", "face_y", "face_stress"))
    # Differentiate the membrane energy in all nine nodal basis directions.
    basis = np.eye(9).reshape(9, 3, 3)
    edge_changes = (basis[:, 1:] - basis[:, :1]).transpose(0, 2, 1)
    df = edge_changes[None] @ y[:, None]
    dg = (f[:, None].transpose(0, 1, 3, 2) @ df + df.transpose(0, 1, 3, 2) @ f[:, None]) / 2
    nu = saved["face_nu"][:, None, None, None]
    ds = saved["face_modulus"][:, None, None, None] * (
        (1 - nu) * dg + nu * np.trace(dg, axis1=-2, axis2=-1)[..., None, None] * np.eye(2)
    )
    flux = (f[:, None] @ ds + df @ stress[:, None]) @ y[:, None].transpose(0, 1, 3, 2)
    first, second = flux[..., 0], flux[..., 1]
    face_matrix = np.stack((-first - second, first, second), axis=2).reshape(len(faces), 9, 9).transpose(0, 2, 1)
    face_matrix *= saved["face_scale"][:, None, None]
    hinge_gradient = saved["hinge_gradient"].transpose(0, 2, 1).reshape(-1, 12)
    hinge_matrix = saved["hinge_scale"][:, None, None] * hinge_gradient[:, :, None] * hinge_gradient[:, None, :]
    contact_faces = faces[saved["contact_faces"]]
    bary, hessian, jacobian = (saved[key] for key in ("contact_bary", "contact_hessian", "contact_jacobian"))
    contact_matrix = np.einsum("fa,fij,fb->faibj", bary, hessian, bary).reshape(-1, 9, 9)
    matrix = sparse.diags(np.repeat(mass, 3), format="csr")
    matrix += scatter_blocks(faces, face_matrix, len(rhs))
    matrix += scatter_blocks(saved["hinge_vertices"], hinge_matrix, len(rhs))
    matrix += scatter_blocks(contact_faces, contact_matrix, len(rhs))
    coupling_values = np.einsum("fa,fij,fkj->faik", bary, hessian, jacobian)
    dofs = (3 * contact_faces[:, :, None] + np.arange(3)).reshape(len(contact_faces), 9)
    n_rigid = jacobian.shape[1]
    coupling = sparse.coo_matrix(
        (
            coupling_values.reshape(-1),
            (
                np.broadcast_to(dofs[:, :, None], (len(dofs), 9, n_rigid)).reshape(-1),
                np.broadcast_to(np.arange(n_rigid), (len(dofs), 9, n_rigid)).reshape(-1),
            ),
        ),
        shape=(len(rhs), n_rigid),
    ).tocsr()
    inverse = saved["schur_inverse"]
    rigid = np.linalg.inv(inverse)
    rigid_contact = np.einsum("fki,fij,flj->kl", jacobian, hessian, jacobian)
    joint = sparse.bmat([[matrix, -coupling], [-coupling.T, sparse.csr_matrix(rigid)]], format="csc")
    asymmetry = sparse.linalg.norm(joint - joint.T) / sparse.linalg.norm(joint)
    assert asymmetry < 1e-12, asymmetry
    factor = splu(joint)
    solved = factor.solve(np.r_[rhs, np.zeros(n_rigid)])
    direct = solved[: len(rhs)]
    direct_residual = rhs - matrix @ direct + coupling @ (inverse @ (coupling.T @ direct))
    residual = rhs - matrix @ iterate + coupling @ (inverse @ (coupling.T @ iterate))
    vector = residual.reshape(-1, 3)
    block_norm = np.einsum("vi,vij,vj->", vector, saved["block_inverse"], vector)
    mass_bound = np.sum(vector**2 / mass[:, None])
    scale = np.linalg.norm(rhs)
    np.savez_compressed(
        args.output / "independent-linear.npz",
        direct_solution=direct.reshape(-1, 3),
        direct_residual=direct_residual.reshape(-1, 3),
        original_iterate=iterate.reshape(-1, 3),
        true_residual=vector,
        recursive_residual=saved["recursive_residual"],
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "capture": file_id(args.field),
            "captured_solver": receipt,
            "unknowns": joint.shape[0],
            "nonzeros": joint.nnz,
            "symmetry_relative": float(asymmetry),
            "minimum_rigid_inertia_eigenvalue": float(np.linalg.eigvalsh(rigid - rigid_contact).min()),
            "direct_relative_residual": float(np.linalg.norm(direct_residual) / scale),
            "iterate_true_relative_residual": float(np.linalg.norm(residual) / scale),
            "iterate_recursive_relative_residual": float(np.linalg.norm(saved["recursive_residual"]) / scale),
            "recursive_true_disagreement_relative": float(np.linalg.norm(vector - saved["recursive_residual"]) / scale),
            "true_block_norm_squared": float(block_norm),
            "true_mass_velocity_bound_squared": float(mass_bound),
            "mass_rms_solution_error_m_per_s": float(
                np.sqrt(np.sum(mass[:, None] * (iterate - direct).reshape(-1, 3) ** 2) / mass.sum())
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "physical_accuracy_qualified": False,
            "scope": "Independent NumPy differentiation of cached plane-stress membrane tangents, hinge rank-one terms "
            "and contact Jacobians, assembled as a sparse joint system. Its shell RHS equals the captured reduced RHS; "
            "the auxiliary rigid RHS is zero, reproducing the same Schur-complement equation. "
            "This diagnoses a frozen linear solve only, not a completed nonlinear or physical trajectory.",
        },
    )


if __name__ == "__main__":
    main()
