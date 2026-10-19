"""Check the long-index factor against independent solves and singular rejection."""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .artifacts import file_id, runtime_identity, write_json
from .cvxopt_factor import CvxoptCholeskyFactor


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from cvxopt import cholmod

    original_options = dict(cholmod.options)
    n = 512
    matrix = sparse.diags([-np.ones(n - 1), np.full(n, 2.00001), -np.ones(n - 1)], [-1, 0, 1], format="csc")
    factor = CvxoptCholeskyFactor(matrix)
    assert dict(cholmod.options) == original_options
    direct = splu(matrix)
    rhs = np.column_stack((np.ones(n), np.linspace(-1, 1, n), np.sin(np.arange(n))))
    cases = []
    for selected in (rhs[:, 0], rhs):
        actual = factor.solve(selected)
        expected = direct.solve(selected)
        relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        residual = float(np.linalg.norm(matrix @ actual - selected) / np.linalg.norm(selected))
        assert relative < 1e-10 and residual < 1e-10
        cases.append({"rhs_shape": list(selected.shape), "relative_error": relative, "relative_residual": residual})
    rejected = False
    try:
        CvxoptCholeskyFactor(sparse.diags([1.0, -1.0], format="csc"))
    except ArithmeticError:
        rejected = True
    assert rejected and dict(cholmod.options) == original_options
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "binary": file_id(Path(factor.module_path)),
            "integer_bytes": factor.index_bytes,
            "cases": cases,
            "indefinite_rejected": rejected,
            "passed": True,
        },
    )


if __name__ == "__main__":
    main()
