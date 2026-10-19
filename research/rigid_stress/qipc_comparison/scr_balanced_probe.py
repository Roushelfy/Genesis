"""Independently evaluate captured balanced-source dual fits and their dimensional error scales."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root

from research.rigid_stress.wrench import project_coulomb

from .artifacts import file_id, runtime_identity, write_json


@dataclass(frozen=True)
class DualProblem:
    weights: np.ndarray
    density: np.ndarray
    operator: np.ndarray
    normals: np.ndarray
    friction: np.ndarray
    target: np.ndarray
    scale: float

    def evaluate(self, multiplier):
        value = self.density - np.einsum("qij,i->qj", self.operator, multiplier)
        projected, jacobian = project_coulomb(value, self.normals, self.friction)
        energy = self.target @ multiplier + 0.5 * np.einsum("q,qi,qi->", self.weights, projected, projected)
        gradient = self.target - np.einsum("q,qij,qj->i", self.weights, self.operator, projected)
        hessian = np.einsum("q,qik,qkl,qjl->ij", self.weights, self.operator, jacobian, self.operator)
        return energy, gradient, hessian, self.weights[:, None] * self.scale * projected

    def objective(self, multiplier):
        return self.evaluate(multiplier)[:2]

    def gradient(self, multiplier):
        return self.evaluate(multiplier)[1]

    def hessian(self, multiplier):
        return self.evaluate(multiplier)[2]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for i_r, run in enumerate(args.runs):
        start = json.loads((run / "start.json").read_text())
        mesh = Path(start["stress_options"]["mesh"])
        with np.load(mesh) as saved:
            vertices, faces, tetrahedra = (saved[name] for name in ("vertices", "surface_triangles", "tetrahedra"))
        owners = (tetrahedra[:, None, :, None] == faces[None, :, None, :]).any(axis=2).all(axis=2)
        assert (owners.sum(axis=0) == 1).all()
        owner_tets = tetrahedra[owners.argmax(axis=0)]
        opposite = owner_tets[~(owner_tets[:, :, None] == faces[:, None, :]).any(axis=2)]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals *= np.where(np.einsum("fi,fi->f", normals, vertices[opposite] - triangles[:, 0]) > 0, 1, -1)[:, None]
        path = run / "rejected-state.npz"
        with np.load(path) as saved:
            data = {
                name.removeprefix("stress.0.intrinsic."): saved[name]
                for name in saved.files
                if name.startswith("stress.0.intrinsic.")
            }
        contacts = np.flatnonzero(data["contacts.valid"][0])
        radius = start["args"]["contact_radius"]
        anchor = data["contacts.center"][0, 0]
        point = data["contacts.position"][0, 0]
        scale = data["tractions.scale"][0, 0]
        force = data["contacts.force"][0, 0]
        moment = data["sources.anchor_moment"][0, 0]
        prior = data["tractions.priors.force"][0, contacts]
        positions, sample_normals, profiles, source = [], [], [], []
        # The scalar tensor stores the coordinate last; the generic failure exporter moved it to the front.
        barycentric_uv = np.moveaxis(data["tractions.sample_barycentric"], 0, -1)[:, :, 0]
        quadrature = start["stress_options"]["quadrature"]
        for i_c in contacts:
            count = data["tractions.sample_count"][0, i_c]
            profile = data["tractions.profile"][0, :count, i_c]
            selected = profile > 0
            indices = data["tractions.sample_indices"][0, :count, i_c][selected]
            face = indices // (3 * quadrature**2)
            face[indices >= 3 * len(faces) * quadrature**2] = data["tractions.circle.face"][0, i_c]
            uv = barycentric_uv[:count, i_c][selected]
            barycentric = np.column_stack((1 - uv.sum(axis=1), uv))
            positions.append(np.einsum("qi,qij->qj", barycentric, triangles[face]))
            sample_normals.append(normals[face])
            profiles.append(profile[selected])
            source.append(np.full(selected.sum(), i_c))
        positions, sample_normals, profiles, source = map(np.concatenate, (positions, sample_normals, profiles, source))
        totals = data["tractions.priors.weight_sum"][0]
        relative = (positions - anchor) / radius
        operator = np.zeros((len(positions), 6, 3))
        operator[:, :3] = np.eye(3)
        x, y, z = relative.T
        operator[:, 3, 1], operator[:, 3, 2] = -z, y
        operator[:, 4, 0], operator[:, 4, 2] = z, -x
        operator[:, 5, 0], operator[:, 5, 1] = -y, x
        problem = DualProblem(
            profiles / profiles.sum(),
            data["tractions.priors.force"][0, source] * (profiles.sum() / totals[source] / scale)[:, None],
            operator,
            sample_normals,
            data["tractions.support.friction"][0, source],
            np.concatenate((force / scale, moment / (radius * scale))),
            scale,
        )
        fitted = minimize(
            problem.objective,
            np.zeros(6),
            jac=True,
            hess=problem.hessian,
            method="trust-exact",
            options={"gtol": 1e-13, "maxiter": 1000},
        )
        corrected = root(problem.gradient, fitted.x, jac=problem.hessian, options={"xtol": 1e-12})
        multiplier = min((fitted.x, corrected.x), key=lambda value: np.linalg.norm(problem.gradient(value)))
        force_scale = np.linalg.norm(prior, axis=1).sum()
        moment_scale = np.sum(
            np.linalg.norm(data["tractions.priors.anchor_moment"][0, contacts], axis=1)
            + (radius + np.linalg.norm(data["contacts.center"][0, contacts] - point, axis=1))
            * np.linalg.norm(prior, axis=1)
        )
        rows = []
        for label, dual in (("native_rejected", data["tractions.dual"][0, 0]), ("independent_cpu", multiplier)):
            _, gradient, hessian, loads = problem.evaluate(dual)
            force_error = np.linalg.norm(loads.sum(axis=0) - force)
            moment_error = np.linalg.norm(
                np.cross(positions - point, loads).sum(axis=0) - data["sources.free_moment"][0, 0]
            )
            compression = np.einsum("qi,qi->q", loads, sample_normals)
            cone_excess = (
                np.linalg.norm(loads - compression[:, None] * sample_normals, axis=1) - problem.friction * compression
            )
            rows.append(
                {
                    "label": label,
                    "force_error_n": force_error,
                    "moment_error_nm": moment_error,
                    "force_error_over_source_scale": force_error / force_scale,
                    "moment_error_over_source_scale": moment_error / moment_scale,
                    "gradient_norm": np.linalg.norm(gradient),
                    "hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
                    "minimum_compression_n": compression.min(),
                    "maximum_cone_excess_n": cone_excess.max(),
                }
            )
        np.savez_compressed(
            args.output / f"source-priors-{i_r}.npz",
            anchor=data["contacts.center"][0, contacts],
            point=data["contacts.position"][0, contacts],
            force=prior,
            radius=np.full(len(contacts), radius),
            friction=data["contacts.friction"][0, contacts],
            anchor_face=data["contacts.anchor_face"][0, contacts],
            moment=data["tractions.priors.anchor_moment"][0, contacts],
        )
        reports.append(
            {
                "input": file_id(path),
                "mesh": file_id(mesh),
                "sample_count": len(positions),
                "physical_force_scale_n": force_scale,
                "physical_moment_scale_nm": moment_scale,
                "net_force_n": np.linalg.norm(force),
                "native_fit_status": int(data["tractions.status"][0, 0]),
                "native_evaluations": int(data["tractions.evaluations"][0, 0]),
                "cpu_minimize_success": bool(fitted.success),
                "cpu_root_success": bool(corrected.success),
                "evaluations": rows,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "runs": reports,
            "scope": "Independent CPU dual solve on exact captured positive quadrature and local cones. No change to native acceptance or stress field.",
        },
    )


if __name__ == "__main__":
    main()
