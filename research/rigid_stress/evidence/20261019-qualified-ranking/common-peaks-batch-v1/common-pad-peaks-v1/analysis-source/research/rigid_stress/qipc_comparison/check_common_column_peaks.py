"""Independent invariant, frame and 3D-component controls for common peaks."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import runtime_identity, write_json
from .common_column_peaks import CommonColumnMeasures, tangent_invariants


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(93)
    stress = rng.normal(size=(37, 3, 3))
    stress = stress + stress.transpose(0, 2, 1)
    stress[:, 2] = 0
    stress[:, :, 2] = 0
    mises, tension = tangent_invariants(stress[:, :2, :2])
    dev = stress - np.trace(stress, axis1=-2, axis2=-1)[:, None, None] / 3 * np.eye(3)
    np.testing.assert_allclose(mises, np.sqrt(1.5 * np.einsum("qij,qij->q", dev, dev)), rtol=1e-14)
    np.testing.assert_allclose(tension, np.maximum(np.linalg.eigvalsh(stress)[:, -1], 0), atol=1e-14)
    rotation = Rotation.from_rotvec([0.5, -0.3, 0.2]).as_matrix()
    local = np.array([[3.0, 1.0, 2.0], [1.0, -2.0, -1.0], [2.0, -1.0, 4.0]])
    expected = None
    for transform in (np.eye(3), rotation):
        basis = np.broadcast_to(transform[:, :2], (2, 3, 2))
        normal = np.broadcast_to(transform[:, 2], (2, 3))
        outer = np.array([[0.0, 0, 1], [1, 0, 1]]) @ transform.T
        inner = outer - normal
        tensor = np.broadcast_to(transform @ local @ transform.T, (2, 3, 3))
        tangent = basis.transpose(0, 2, 1) @ tensor @ basis
        measures = CommonColumnMeasures(basis, normal, outer, inner, (False, False))
        measures.update_peaks((2 * tangent, tangent), np.array([0.25, 0.75]))
        measures.integrate_transverse((2 * tensor, tensor), np.array([0.4, 0.6]))
        result = measures.result()
        np.testing.assert_allclose(result["full3d_only_components"][0]["rms_pa"], [8, 4], atol=1e-14)
        np.testing.assert_allclose(result["full3d_only_components"][1]["rms_pa"], [2 * np.sqrt(5), np.sqrt(5)])
        assert all(abs(c["relative_l2_error"] - 1) < 1e-14 for c in result["full3d_only_components"])
        for peak in result["tangent_peaks"]:
            assert abs(peak["relative_difference"] - 1) < 1e-14
        values = [p["reference"]["stress_pa"] for p in result["tangent_peaks"]]
        if expected is not None:
            np.testing.assert_allclose(values, expected, rtol=1e-14)
        expected = values
        measures.shell = (True, False)
        missing = measures.result()["full3d_only_components"]
        assert all(c["rms_pa"][0] is None and c["relative_l2_error"] is None for c in missing)
    write_json(args.output / "result.json", {"identity": runtime_identity(), "passed": True})


if __name__ == "__main__":
    main()
