"""Compare compatible fibers/resultants and audit the saved shell state's nonlinear force balance."""

import argparse
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils.misc import qd_to_numpy

from ..skfem_reference import SkfemElasticReference
from .common import file_identity, identity, write_json
from .prescribed import add_prescribed_force


def relative_error(actual, reference, areas):
    weights = areas[(slice(None),) + (None,) * (reference.ndim - 1)]
    denominator = np.sum(weights * reference**2)
    return float(np.sqrt(np.sum(weights * (actual - reference) ** 2) / denominator)) if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    rows = []
    for level in (1, 2):
        with np.load(args.fields / f"level{level}/fields.npz") as saved:
            tensor, target = saved["shell_tensor_pa"], saved["reference_tensor_pa"]
            rest, faces, current = saved["rest_vertices"], saved["faces"], saved["final_vertices"]
            applied, face_order = saved["applied_force_n"], saved["face_order"]
        with np.load(args.inputs / f"egg-level{level}.npz") as saved:
            vertices, tets, midpoint, mid_faces = (
                saved[key] for key in ("vertices", "tetrahedra", "mid_vertices", "mid_faces")
            )
        with np.load(args.loads / f"level{level}.npz") as saved:
            displacement = saved["volume_displacement_m"]
        edges = rest[faces[:, 1:]] - rest[faces[:, :1]]
        normal = np.cross(edges[:, 0], edges[:, 1])
        areas = np.linalg.norm(normal, axis=1) / 2
        normal /= 2 * areas[:, None]
        projectors = np.eye(3) - normal[:, :, None] * normal[:, None, :]
        target_tangent = np.einsum("fij,fljk,fkm->flim", projectors, target, projectors)
        count = len(midpoint)
        outer = vertices[:count][mid_faces].mean(axis=1)[face_order]
        inner = vertices[2 * count : 3 * count][mid_faces].mean(axis=1)[face_order]
        centers = rest[faces].mean(axis=1)
        projected_thickness = np.einsum("fi,fi->f", outer - inner, normal)
        assert (projected_thickness > 0).all()
        z = np.einsum("fli,fi->fl", np.stack((inner, centers, outer), axis=1) - centers[:, None], normal)
        gradient = (tensor[:, 2] - tensor[:, 0]) / 0.0005
        same_position_tensor = tensor[:, 1, None] + z[:, :, None, None] * gradient[:, None]
        same_position_tangent = np.einsum("fij,fljk,fkm->flim", projectors, same_position_tensor, projectors)
        # A face-centroid column crosses each of the three tetrahedra in a
        # prismatic layer at thirds. Two Gauss points per segment integrate
        # piecewise-linear stress and its first moment exactly.
        abscissa, weight = np.polynomial.legendre.leggauss(2)
        breaks = np.linspace(-0.5, 0.5, 7)
        fractions = ((breaks[:-1] + breaks[1:])[:, None] / 2 + np.diff(breaks)[:, None] * abscissa / 2).reshape(-1)
        weights = (np.diff(breaks)[:, None] * weight / 2).reshape(-1)
        points = (outer + inner)[:, None] / 2 + fractions[None, :, None] * (outer - inner)[:, None]
        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000.0)
        ref_tensor, _ = reference.stress_at_points(displacement, points.reshape(-1, 3))
        ref_tensor = ref_tensor.reshape(len(faces), len(fractions), 3, 3)
        ref_tangent = np.einsum("fij,fqjk,fkm->fqim", projectors, ref_tensor, projectors)
        jacobian_weights = projected_thickness[:, None] * weights
        ref_n = np.einsum("fq,fqij->fij", jacobian_weights, ref_tangent)
        ref_m = np.einsum("fq,fqij->fij", jacobian_weights * projected_thickness[:, None] * fractions, ref_tangent)
        shell_n = projected_thickness[:, None, None] * same_position_tangent[:, 1]
        shell_m = projected_thickness[:, None, None] ** 3 / 12 * gradient
        # Reassemble internal forces at the retained final state without
        # advancing it. This checks equilibrium beyond the linear PCG residual.
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, 0.0),
            ),
            show_viewer=False,
        )
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=args.inputs / f"egg-mid-level{level}.obj",
                decimate=False,
            ),
            material=gs.materials.Shell(
                rho=2000.0,
                E=1e10,
                nu=0.3,
                thickness=0.0005,
            ),
        )
        scene.build(n_envs=1)
        np.testing.assert_array_equal(entity.init_verts, rest)
        entity.set_verts_pos(current)
        entity.set_verts_vel(0.0)
        solver = scene.sim.shell_solver
        shell.kernel_shell_compute_forces(
            0.01, solver._gravity, solver.shell_state, solver.shell_scratch, solver.shell_info
        )
        force = torch.as_tensor(applied[:, None], dtype=gs.tc_float, device=gs.device)
        add_prescribed_force(0.01, force, solver.shell_scratch)
        residual = qd_to_numpy(solver.shell_scratch.verts_rhs, transpose=True, copy=True)[0] / 0.01
        row = {
            "level": level,
            "field_source": file_identity(args.fields / f"level{level}/fields.npz"),
            "same_position_tangent_tensor_relative_error": relative_error(same_position_tangent, target_tangent, areas),
            "nominal_fiber_full_3d_tensor_relative_error": relative_error(tensor, target, areas),
            "reference_transverse_tensor_fraction": relative_error(target_tangent, target, areas),
            "membrane_resultant_relative_error": relative_error(shell_n, ref_n, areas),
            "bending_resultant_relative_error": relative_error(shell_m, ref_m, areas),
            "minimum_projected_volume_wall_m": float(projected_thickness.min()),
            "maximum_projected_volume_wall_m": float(projected_thickness.max()),
            "static_force_residual_n": float(np.linalg.norm(residual)),
            "static_force_relative_residual": float(np.linalg.norm(residual) / np.linalg.norm(applied)),
            "half_external_work_j": float(np.sum(applied * (current - rest)) / 2),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"level{level}-common-fields.npz",
            shell_tensor_pa=same_position_tangent,
            reference_tensor_pa=target_tangent,
            membrane_shell_n_per_m=shell_n,
            membrane_reference_n_per_m=ref_n,
            bending_shell_n=shell_m,
            bending_reference_n=ref_m,
            areas_m2=areas,
            positions_m=centers,
            static_residual_n=residual,
        )
    write_json(
        args.output / "result.json",
        {"kind": "compatible-prescribed-field-audit", "identity": identity(), "cases": rows},
    )


if __name__ == "__main__":
    main()
