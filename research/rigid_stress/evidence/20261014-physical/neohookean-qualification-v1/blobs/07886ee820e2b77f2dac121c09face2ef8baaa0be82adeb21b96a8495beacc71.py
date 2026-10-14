"""Check nonlinear bulk derivatives, objectivity, analytic compression and saved states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem import MeshTet

from .neohookean_reference import FORMULATION_SOURCE, NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cube = MeshTet.init_tensor(*([np.linspace(0, 0.001, 3)] * 3))
    rng = np.random.default_rng(2026101409)
    controls = []
    for young, poisson in ((1e6, 0.0), (1e10, 0.3)):
        ref = SkfemElasticReference(cube.p.T, cube.t.T, young, poisson, 1000)
        bulk = NeoHookeanBulk(ref)
        zero = np.zeros(ref.basis.N)
        energy, gradient, matrix = bulk.evaluate(zero)
        stiffness_error = np.linalg.norm((matrix - ref.stiffness).data) / np.linalg.norm(ref.stiffness.data)
        assert energy == 0 and not gradient.any()
        assert stiffness_error < 1e-12
        vector = np.zeros(ref.basis.N)
        vector[ref.component_dofs] = ref.coordinates * [-0.07, 0.02, 0.01]
        vector += rng.normal(size=len(vector)) * 1e-6
        direction = rng.normal(size=len(vector))
        direction /= np.linalg.norm(direction)
        energy, gradient, matrix = bulk.evaluate(vector)
        derivative_rows = []
        for step in (1e-8, 3e-9):
            plus, gp, _ = bulk.evaluate(vector + step * direction, False)
            minus, gm, _ = bulk.evaluate(vector - step * direction, False)
            energy_error = abs((plus - minus) / (2 * step) - gradient @ direction) / abs(gradient @ direction)
            hessian_error = np.linalg.norm((gp - gm) / (2 * step) - matrix @ direction) / np.linalg.norm(
                matrix @ direction
            )
            assert energy_error < 1e-6 and hessian_error < 1e-6
            derivative_rows.append(
                {
                    "step_m": step,
                    "energy_gradient_relative_error": float(energy_error),
                    "gradient_hessian_relative_error": float(hessian_error),
                }
            )
        symmetry_error = np.linalg.norm((matrix - matrix.T).data) / np.linalg.norm(matrix.data)
        assert symmetry_error < 1e-12
        rotation_rows = []
        for angle in (0.003, 0.6, 1.2):
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            rotated = zero.copy()
            rotated[ref.component_dofs] = ref.coordinates @ (rotation - np.eye(3)).T + [0.002, -0.001, 0.0004]
            e, g, _ = bulk.evaluate(rotated, False)
            relative_force = np.linalg.norm(g) / (young * 0.001**2)
            assert relative_force < 1e-12 and abs(e) < 1e-16 * young * 0.001**3
            lower = bulk.determinant_lower_bound(rotated)
            assert lower > 1 - 1e-12
            rotation_rows.append(
                {
                    "angle_rad": angle,
                    "energy_J": e,
                    "force_scaled_error": float(relative_force),
                    "whole_volume_determinant_lower_bound": lower,
                }
            )
        compressed = zero.copy()
        stretch = np.array([0.93, 1.0, 1.0])
        compressed[ref.component_dofs] = ref.coordinates * (stretch - 1)
        e, g, _ = bulk.evaluate(compressed, False)
        lam, mu = ref.lame
        piola = np.diag(mu * (stretch - 1 / stretch) + lam * np.log(stretch.prod()) / stretch)
        boundary_load = ref.uniform_stress_load(piola)
        analytic_error = np.linalg.norm(g - boundary_load) / np.linalg.norm(boundary_load)
        assert analytic_error < 1e-12
        controls.append(
            {
                "young_Pa": young,
                "poisson": poisson,
                "rest_stiffness_relative_error": float(stiffness_error),
                "derivatives": derivative_rows,
                "hessian_symmetry_relative_error": float(symmetry_error),
                "rigid_rotation": rotation_rows,
                "analytic_compression_boundary_error": float(analytic_error),
                "analytic_first_piola_Pa": piola.tolist(),
                "compression_energy_J": e,
            }
        )
    rows, identities = [], {}
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "solution/configuration.json").read_text())
        with np.load(run / "solution/fields.npz") as saved:
            displacement, rest = saved["displacement"], saved["rest_positions"]
        offset = 0
        for name in ("body", "left_shoe", "right_shoe"):
            mesh_path = Path(config["mesh"]) if name == "body" else run / f"solution/{name}_mesh.npz"
            young, poisson, density = (1e10, 0.3, 2000) if name == "body" else (config["pad_young"], 0, 1000)
            with np.load(mesh_path) as saved:
                ref = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density)
            count = ref.basis.N
            np.testing.assert_allclose(rest[offset // 3 : (offset + count) // 3], ref.coordinates, rtol=0, atol=1e-14)
            vector = np.empty(count)
            vector[ref.component_dofs.reshape(-1)] = displacement[offset : offset + count]
            elastic = ref.align_rigid_gauge(vector)
            linear_force = ref.stiffness @ elastic
            bulk = NeoHookeanBulk(ref)
            e, force, _ = bulk.evaluate(vector, False)
            _, aligned_force, _ = bulk.evaluate(elastic, False)
            rows.append(
                {
                    "run": str(run),
                    "part": name,
                    "linear_energy_J": float(0.5 * elastic @ linear_force),
                    "neo_hookean_energy_at_saved_positions_J": e,
                    "raw_nonlinear_minus_linear_bulk_force_relative_norm": float(
                        np.linalg.norm(force - linear_force) / np.linalg.norm(linear_force)
                    ),
                    "gauge_aligned_nonlinear_minus_linear_force_relative_norm": float(
                        np.linalg.norm(aligned_force - linear_force) / np.linalg.norm(linear_force)
                    ),
                    "whole_volume_determinant_lower_bound": bulk.determinant_lower_bound(vector),
                }
            )
            offset += count
        assert offset == len(displacement)
        path = run / "solution/fields.npz"
        identities[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "scope": __doc__,
        "formulation_source": FORMULATION_SOURCE,
        "controls": controls,
        "saved_state_rows": rows,
        "input_sha256": identities,
        "goal_completion": False,
        "limitations": "This qualifies nonlinear bulk assembly and audits a different constitutive law at saved linear positions. It is not a solved nonlinear contact reference. Rigid-gauge alignment of an infinitesimal rotation changes actual finite geometry; raw/aligned force differences diagnose assumptions rather than establish a physical stress error. Nonlinear contact, quadrature/spatial/history convergence and matched Franka feedback remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
