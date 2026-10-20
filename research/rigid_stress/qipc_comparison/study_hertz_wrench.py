"""Measure a single small wrench correction of the retained coarse-egg Hertz traction prior."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .p2_elastic import FreeElasticP2


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--transfer", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.transfer / "result.json"
    result = json.loads(source.read_text())
    assert result["passed"]
    row = next(r for r in result["rows"] if r["case"] == "egg")
    for item in (row["mesh"], row["arrays"]):
        assert file_id(Path(item["path"])) == item
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "source": file_id(source),
            "normal_force_n": 0.2,
            "tangent_ratio": 0.2,
            "declared_mu": 0.5,
        },
    )
    with np.load(row["mesh"]["path"]) as saved:
        vertices, tets, triangles = (saved[k] for k in ("vertices", "tetrahedra", "surface_triangles"))
    with np.load(row["arrays"]["path"]) as saved:
        points, faces = saved["sample_points_m"][1], saved["sample_faces"][1]
        original_normal_nodal_force = saved["native_force_n"][1]
    center = vertices[np.argmax(vertices[:, 2])]
    normal = -original_normal_nodal_force.sum(axis=0) / 0.2
    tangent = np.cross(normal, [1.0, 0, 0])
    tangent /= np.linalg.norm(tangent)
    np.testing.assert_allclose(np.linalg.norm(normal), 1, rtol=1e-12, atol=0)
    angles = len(points) // 3
    t, gauss = np.polynomial.legendre.leggauss(3)
    t, gauss = (1 + t) / 2, gauss / 2
    weights = np.repeat(3 * t * t * gauss / angles, angles)
    force = 0.2 * (-normal + 0.2 * tangent)
    prior = weights[:, None] * force
    lever = ellipse(0.2, 1e10 / (1 - 0.3**2), np.diag([50.0, 80.0]))[0][0]
    relative = (points - center) / lever
    operator = np.zeros((len(points), 6, 3))
    operator[:, :3] = np.eye(3)
    operator[:, 3, 1], operator[:, 3, 2] = -relative[:, 2], relative[:, 1]
    operator[:, 4, 0], operator[:, 4, 2] = relative[:, 2], -relative[:, 0]
    operator[:, 5, 0], operator[:, 5, 1] = -relative[:, 1], relative[:, 0]
    target = np.r_[force, np.zeros(3)]
    prior_wrench = np.einsum("sij,sj->i", operator, prior)
    gram = np.einsum("sik,sjk,s->ij", operator, operator, weights)
    multiplier = np.linalg.solve(gram, target - prior_wrench)
    correction = weights[:, None] * np.einsum("sji,j->si", operator, multiplier)
    corrected = prior + correction
    np.testing.assert_allclose(np.einsum("sij,sj->i", operator, corrected), target, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(prior_wrench[3:]) * lever, row["curved_proportional_tangent_moment_error_nm"], rtol=1e-11, atol=0
    )
    # Match native P2 numbering by its topological edge construction, then use
    # independent scikit-fem elasticity and consistent free-link mass relief.
    edges = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0)
    native_coordinates = np.concatenate((vertices, vertices[edges].mean(axis=1)))
    edge_lookup = {tuple(edge): i + len(vertices) for i, edge in enumerate(edges)}
    nodes = np.column_stack(
        (
            triangles,
            np.array(
                [
                    [edge_lookup[tuple(sorted((face[a], face[b])))] for a, b in ((0, 1), (0, 2), (1, 2))]
                    for face in triangles
                ]
            ),
        )
    )
    origin = vertices[triangles[faces, 0]]
    local_edges = vertices[triangles[faces, 1:3]] - origin[:, None]
    dual = np.linalg.solve(local_edges @ local_edges.swapaxes(1, 2), local_edges)
    local = np.einsum("sij,sj->si", dual, points - origin)
    bary = np.column_stack((1 - local.sum(axis=1), local))
    shape = np.column_stack(
        (bary * (2 * bary - 1), 4 * bary[:, 0] * bary[:, 1], 4 * bary[:, 0] * bary[:, 2], 4 * bary[:, 1] * bary[:, 2])
    )
    np.testing.assert_allclose(
        np.einsum("si,sij->sj", shape, native_coordinates[nodes[faces]]), points, rtol=0, atol=1e-13
    )
    opposite = {}
    for tet in tets:
        for i in range(4):
            key = tuple(sorted(np.delete(tet, i)))
            opposite.setdefault(key, []).append(tet[i])
    opposite_nodes = np.array([opposite[tuple(sorted(face))][0] for face in triangles[faces]])
    material_normal = np.cross(local_edges[:, 0], local_edges[:, 1])
    material_normal *= np.where(np.einsum("si,si->s", material_normal, vertices[opposite_nodes] - origin) > 0, -1, 1)[
        :, None
    ]
    material_normal /= np.linalg.norm(material_normal, axis=1)[:, None]
    cpu = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000)
    permutation = cpu.reference.native_dof_permutation(native_coordinates).reshape(-1, 3)[:, 0] // 3
    fields, loads, metrics = {}, {}, {}
    for name, sample_force in (("prior", prior), ("corrected", corrected)):
        native_load = np.zeros_like(native_coordinates)
        np.add.at(native_load, nodes[faces], shape[..., None] * sample_force[:, None])
        load = np.empty_like(native_load)
        load[permutation] = native_load
        tensor, _, balance, residual = cpu.solve(load)
        assert np.linalg.norm(residual) / np.linalg.norm(load) < 1e-7
        compression = -np.einsum("si,si->s", sample_force, material_normal)
        tangential = sample_force + compression[:, None] * material_normal
        cone_excess = np.maximum(np.linalg.norm(tangential, axis=1) - 0.5 * compression, 0)
        plane_compression = -sample_force @ normal
        plane_tangent = sample_force + plane_compression[:, None] * normal
        plane_ratio = np.linalg.norm(plane_tangent, axis=1) / plane_compression
        plane_excess = np.maximum(np.linalg.norm(plane_tangent, axis=1) - 0.5 * plane_compression, 0)
        eig = np.linalg.eigvalsh(tensor)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
        fields[name], loads[name] = tensor, load
        metrics[name] = {
            "minimum_sample_compression_n": float(compression.min()),
            "contact_plane_minimum_compression_n": float(plane_compression.min()),
            "contact_plane_maximum_required_mu": float(plane_ratio.max()),
            "contact_plane_cone_violations": int((plane_excess > 1e-12 * 0.2).sum()),
            "maximum_required_local_mu": float(np.max(np.linalg.norm(tangential, axis=1) / compression))
            if compression.min() > 0
            else None,
            "declared_mu_cone_excess_sum_n": float(cone_excess.sum()),
            "cone_violations": int((cone_excess > 1e-12 * 0.2).sum()),
            "balance_correction_norm_n": float(np.linalg.norm(balance)),
            "raw_tensile_max_pa": float(eig[..., -1].max()),
            "raw_von_mises_max_pa": float(mises.max()),
        }
    volumes = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
    squared = (fields["corrected"] - fields["prior"]) ** 2
    error = np.sqrt(volumes @ squared.sum(axis=(1, 2, 3)) / (volumes @ (fields["prior"] ** 2).sum(axis=(1, 2, 3))))
    path = args.output / "correction.npz"
    np.savez_compressed(
        path,
        sample_points_m=points,
        prior_sample_force_n=prior,
        corrected_sample_force_n=corrected,
        prior_nodal_force_n=loads["prior"],
        corrected_nodal_force_n=loads["corrected"],
        prior_stress_pa=fields["prior"],
        corrected_stress_pa=fields["corrected"],
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "metrics": metrics,
            "prior_moment_error_nm": float(np.linalg.norm(prior_wrench[3:]) * lever),
            "sample_correction_l1_n": float(np.linalg.norm(correction, axis=1).sum()),
            "sample_correction_weighted_relative": float(
                np.sqrt(np.sum(correction * correction / weights[:, None])) / np.linalg.norm(force)
            ),
            "complete_tensor_relative_change": float(error),
            "arrays": file_id(path),
            "strict_original_preserved": True,
            "engineering_admitted": False,
            "scope": "One unconstrained 6x6 weighted minimum-force correction on the retained faceted egg. Full wrench, separate mu=0.5 reference-facet/contact-plane cone diagnostics, independent mass-relieved complete P2 response and raw peaks are measured. This is a load-model sensitivity, not hard-contact physical accuracy or accepted friction/history behavior.",
        },
    )


if __name__ == "__main__":
    main()
