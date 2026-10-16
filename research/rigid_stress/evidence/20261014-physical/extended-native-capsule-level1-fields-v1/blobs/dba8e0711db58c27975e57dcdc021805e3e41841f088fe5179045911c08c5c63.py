"""Check wide whole-volume certificates against independent interior barycentric stress samples."""

import argparse
import json
from pathlib import Path

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.audit_run)
    assert int((args.audit_run / "returncode.txt").read_text()) == 0
    report = json.loads((args.audit_run / "results/summary.json").read_text())
    source = Path(report["source_run"])
    closed_source(source)
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.audit_run / "results/body_geometry.npz") as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        permutation = saved["component_permutation"].copy()
    bulk = NeoHookeanBulk(body)
    bary = np.vstack((np.eye(4), np.full((1, 4), 0.25), np.random.default_rng(2041).dirichlet(np.ones(4), 64)))
    bary = np.asarray(bary, dtype=np.longdouble)
    bary /= bary.sum(axis=1, keepdims=True)
    bary[:, -1] = 1 - bary[:, :3].sum(axis=1)
    assert np.all(bary >= 0)
    rows = []
    for step in report["saved_snapshot_steps"]:
        with np.load(source / f"results/step{step:05d}.npz") as saved:
            vector = np.empty(body.basis.N, dtype=np.longdouble)
            vector[permutation] = saved["displacement"][: body.basis.N]
        corners = bulk.corner_deformation_gradients(vector)
        f = np.einsum("vc,ecij->evij", np.asarray(bary, dtype=np.longdouble), corners, optimize=True)
        determinant = np.sum(f[..., 0, :] * np.cross(f[..., 1, :], f[..., 2, :]), axis=-1)
        # The deviatoric Cauchy stress has no lambda term. Use F F^T directly,
        # independently of the cancellation-resistant corner stress evaluator.
        left = f @ f.swapaxes(-1, -2)
        dev = left - np.trace(left, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = bulk.mu * np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1))) / determinant
        bound = report["rows"][step - 1]
        assert bound["step"] == step
        maximum = float(vm.max())
        # Direct F F^T evaluation has O(eps*mu) absolute cancellation;
        # retain that declared arithmetic allowance near zero stress.
        allowance = float(256 * np.finfo(np.longdouble).eps * bulk.mu * np.max(np.sum(f * f, axis=(-2, -1))))
        reported_j_rounding = 4 * np.finfo(float).eps * max(1, abs(bound["whole_volume_determinant_lower_bound"]))
        assert float(determinant.min()) >= bound["whole_volume_determinant_lower_bound"] - reported_j_rounding
        assert maximum <= bound["raw_global_peak_upper_Pa"] + allowance
        rows.append(
            {
                "step": step,
                "interior_and_corner_samples": int(vm.size),
                "sampled_maximum_Pa": maximum,
                "whole_volume_upper_Pa": bound["raw_global_peak_upper_Pa"],
                "minimum_sampled_J": float(determinant.min()),
                "whole_volume_J_lower_bound": bound["whole_volume_determinant_lower_bound"],
                "reported_FP64_J_scalar_rounding_allowance": reported_j_rounding,
                "independent_FFT_stress_arithmetic_allowance_Pa": allowance,
            }
        )
    result = {
        "scope": __doc__,
        "audit_run": str(args.audit_run),
        "rows": rows,
        "total_sampled_points": sum(r["interior_and_corner_samples"] for r in rows),
        "barycentric_points_per_element": len(bary),
        "maximum_barycentric_partition_error": float(abs(bary.sum(axis=1) - 1).max()),
        "goal_completion": False,
        "limitations": "Barycentric samples independently check the stored Bernstein certificates; sampling alone is not a whole-volume maximum proof. The certificate follows the affine-F determinant and quadratic deviatoric Kirchhoff Bernstein construction. The explicit long-double stress cancellation allowance is recorded separately. This validates computed fields, not continuum/contact accuracy or complete application history.",
    }
    np.savez_compressed(args.output / "barycentric_samples.npz", barycentric_coordinates=bary)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_checked": len(rows), "total_sampled_points": result["total_sampled_points"]}))


if __name__ == "__main__":
    main()
