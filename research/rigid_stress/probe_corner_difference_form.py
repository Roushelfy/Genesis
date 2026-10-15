"""Measure an equivalent quadratic corner difference formula on complete frozen native fields."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs

from .diagnose_native_corner_arithmetic import difference
from .summarize_fine_references import closed_source


@qd.kernel
def kernel_corner_formula(
    nodes: qd.types.ndarray(dtype=qd.i32, ndim=3),
    gradients: qd.types.ndarray(dtype=qd.f64, ndim=4),
    midpoint_slots: qd.types.ndarray(dtype=qd.i32, ndim=2),
    displacement: qd.types.ndarray(dtype=qd.f64, ndim=2),
    components: qd.types.ndarray(dtype=qd.f64, ndim=3),
    use_difference: qd.template(),
):
    for i_element, i_corner in qd.ndrange(nodes.shape[0], 4):
        derivative = qd.Matrix.zero(qd.f64, 3, 3)
        if qd.static(use_difference):
            i_origin = nodes[i_element, i_corner, i_corner]
            for i_other in range(4):
                if i_other != i_corner:
                    i_vertex = nodes[i_element, i_corner, i_other]
                    i_midpoint = nodes[i_element, i_corner, midpoint_slots[i_corner, i_other]]
                    for i_row, i_column in qd.static(qd.ndrange(3, 3)):
                        tangent = 4 * (displacement[i_midpoint, i_row] - displacement[i_origin, i_row]) - (
                            displacement[i_vertex, i_row] - displacement[i_origin, i_row]
                        )
                        derivative[i_row, i_column] -= tangent * gradients[i_element, i_corner, i_other, i_column]
        else:
            for i_local in range(7):
                i_node = nodes[i_element, i_corner, i_local]
                for i_row, i_column in qd.static(qd.ndrange(3, 3)):
                    derivative[i_row, i_column] += (
                        displacement[i_node, i_row] * gradients[i_element, i_corner, i_local, i_column]
                    )
        lame_mu = 1e10 / (2 * (1 + 0.3))
        lame_lambda = 1e10 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
        tensor = lame_mu * (derivative + derivative.transpose())
        for i_axis in qd.static(range(3)):
            tensor[i_axis, i_axis] += lame_lambda * derivative.trace()
        components[i_element, i_corner, 0] = tensor[0, 0]
        components[i_element, i_corner, 1] = tensor[1, 1]
        components[i_element, i_corner, 2] = tensor[2, 2]
        components[i_element, i_corner, 3] = tensor[0, 1]
        components[i_element, i_corner, 4] = tensor[0, 2]
        components[i_element, i_corner, 5] = tensor[1, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-run", type=Path, nargs="+", required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    inputs, rows = {}, []
    for i_case, run in enumerate(args.diagnostic_run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        summary_path = run / "results/summary.json"
        summary = json.loads(summary_path.read_text())
        geometry = run / "results/geometry.npz"
        for path in [summary_path, geometry]:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(geometry) as saved:
            nodes = np.asarray(saved["native_corner_nodes"], dtype=np.int32)
            gradients = saved["native_corner_gradients"].copy()
            elements = saved["native_elements"].copy()
        midpoint_slots = np.full((4, 4), -1, dtype=np.int32)
        for i_corner in range(4):
            i_slot = 4
            for i_edge, (i_first, i_second) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
                if i_corner in [i_first, i_second]:
                    i_other = i_second if i_corner == i_first else i_first
                    midpoint_slots[i_corner, i_other] = i_slot
                    np.testing.assert_array_equal(nodes[:, i_corner, i_slot], elements[:, 4 + i_edge])
                    i_slot += 1
        for original in summary["rows"]:
            mode = original["mode"]
            field_path = run / f"results/{mode}.npz"
            inputs[str(field_path)] = hashlib.sha256(field_path.read_bytes()).hexdigest()
            with np.load(field_path) as saved:
                displacement = saved["frozen_native_displacement_m"].copy()
                reference = saved["stored_reference_components_Pa"].copy()
                extended = saved["independent_extended_geometry_components_Pa"].copy()
                previous = saved["rebuilt_native_cached_components_Pa"].copy()
            ordinary, difference_form = [np.empty_like(reference) for _ in range(2)]
            kernel_corner_formula(nodes, gradients, midpoint_slots, displacement, ordinary, False)
            kernel_corner_formula(nodes, gradients, midpoint_slots, displacement, difference_form, True)
            row = {
                "diagnostic_run": str(run),
                "mode": mode,
                "tetrahedra": len(elements),
                "ordinary_vs_saved_native": difference(ordinary, previous),
                "ordinary_vs_original_reference": difference(ordinary, reference),
                "difference_form_vs_original_reference": difference(difference_form, reference),
                "ordinary_vs_extended_geometry_reference": difference(ordinary, extended),
                "difference_form_vs_extended_geometry_reference": difference(difference_form, extended),
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"case{i_case:02d}-{mode}.npz",
                ordinary_components_Pa=ordinary,
                difference_form_components_Pa=difference_form,
                original_reference_components_Pa=reference,
                extended_geometry_reference_components_Pa=extended,
            )
            print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "production_modified": False,
        "new_physical_state_accepted": False,
        "goal_completion": False,
        "limitations": "The exact quadratic corner derivative uses the three outgoing edge tangents, each4*(midpoint-origin)-(vertex-origin), with unchanged stored barycentric gradients. Every element/corner/component and both original and extended reference comparisons remain. These are measured candidate arithmetic fields on frozen states, with unchanged existing component budgets. A CPU/candidate result alone qualifies neither production CUDA behavior nor physical/model convergence or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
