"""Compare every accepted original state with the corresponding wide-coordinate history."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def tensor_from_f(f, lame):
    """Evaluate Cauchy stress in long double without double-precision matrix determinant."""
    h = np.asarray(f, dtype=np.longdouble) - np.eye(3, dtype=np.longdouble)
    tr = np.trace(h, axis1=-2, axis2=-1)
    det_h = np.sum(h[..., 0, :] * np.cross(h[..., 1, :], h[..., 2, :]), axis=-1)
    delta = tr + (tr**2 - np.einsum("...ij,...ji->...", h, h)) / 2 + det_h
    assert np.all(delta > -1)
    lam, mu = lame
    kirchhoff = mu * (h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2))
    kirchhoff += lam * np.log1p(delta)[..., None, None] * np.eye(3, dtype=np.longdouble)
    return kirchhoff / (1 + delta)[..., None, None]


def peak(tensor):
    deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return float(np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1))).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.baseline) == len(args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for baseline, candidate in zip(args.baseline, args.candidate):
        for run in [baseline, candidate]:
            closed_source(run)
        assert int((candidate / "returncode.txt").read_text()) == 0
        configs = [json.loads((run / "results/configuration.json").read_text()) for run in [baseline, candidate]]
        for key in ["capture", "dt", "steps", "control_steps", "dhat", "captured_initial_velocity"]:
            assert configs[0][key] == configs[1][key], key
        assert len({hashlib.sha256(Path(c["mjcf"]).read_bytes()).hexdigest() for c in configs}) == 1
        with (
            np.load(baseline / "results/closure_state.npz") as a,
            np.load(candidate / "results/closure_state.npz") as b,
        ):
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        with (
            np.load(baseline / "results/reference_geometry.npz") as a,
            np.load(candidate / "results/reference_geometry.npz") as b,
        ):
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            body = SkfemElasticReference(a["vertices"], a["tetrahedra"], 1e10, 0.3, 2000)
            permutation = a["body_component_permutation"].copy()
        bulk = NeoHookeanBulk(body)
        original_steps = sorted((baseline / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
        assert original_steps
        rows = []
        for step, path in enumerate(original_steps, 1):
            assert path.name == f"step{step:04d}.npz"
            with np.load(path) as a, np.load(candidate / "results" / path.name) as b:
                assert a.files == b.files
                errors = {}
                for key in a.files:
                    assert a[key].shape == b[key].shape
                    difference = np.asarray(a[key], dtype=np.longdouble) - b[key]
                    errors[key] = {
                        "maximum_absolute_difference": float(abs(difference).max(initial=0)),
                        "L2_difference": float(np.linalg.norm(difference)),
                    }
                for key in ["applied_force_N", "body_component_permutation", "arm_support_generalized_effort"]:
                    np.testing.assert_array_equal(a[key], b[key])
                vectors = []
                for saved in [a, b]:
                    vector = np.empty(body.basis.N, dtype=np.longdouble)
                    vector[permutation] = saved["displacement"][: body.basis.N]
                    vectors.append(vector)
            corners = [tensor_from_f(bulk.corner_deformation_gradients(u), body.lame) for u in vectors]
            quadrature = [
                tensor_from_f(bulk.deformation_gradients(u).transpose(2, 3, 0, 1), body.lame) for u in vectors
            ]
            corner_change = float(np.linalg.norm(corners[0] - corners[1]) / np.linalg.norm(corners[1]))
            quadrature_change = float(np.linalg.norm(quadrature[0] - quadrature[1]) / np.linalg.norm(quadrature[1]))
            raw_peak_change = abs(peak(corners[0]) / peak(corners[1]) - 1)
            assert max(corner_change, quadrature_change, raw_peak_change) <= 1e-8
            rounded = np.asarray(vectors[1], dtype=float).astype(np.longdouble)
            rounded_corners = tensor_from_f(bulk.corner_deformation_gradients(rounded), body.lame)
            rows.append(
                {
                    "step": step,
                    "time_s": step * configs[0]["dt"],
                    "all_saved_array_differences": errors,
                    "all_corner_tensor_relative_change": corner_change,
                    "all_quadrature_tensor_relative_change": quadrature_change,
                    "raw_corner_peak_relative_change": raw_peak_change,
                    "wide_coordinate_raw_corner_peak_Pa": peak(corners[1]),
                    "rounded_coordinate_peak_absolute_difference_Pa": abs(peak(rounded_corners) - peak(corners[1])),
                    "rounded_coordinate_all_corner_tensor_relative_difference": float(
                        np.linalg.norm(rounded_corners - corners[1]) / np.linalg.norm(corners[1])
                    ),
                }
            )
        cases.append(
            {
                "baseline": str(baseline),
                "candidate": str(candidate),
                "all_original_accepted_states_checked": len(rows),
                "rows": rows,
            }
        )
    result = {
        "scope": __doc__,
        "cases": cases,
        "goal_completion": False,
        "limitations": "Every saved original accepted state, all original array fields and full corner/quadrature tensors are compared. Original incomplete histories remain incomplete. Independent complete force audits are separate requirements. Rounding diagnostics quantify observable arithmetic; they do not establish physical spatial/contact accuracy or original-history matching.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "original_states_checked": sum(len(c["rows"]) for c in cases)}))


if __name__ == "__main__":
    main()
