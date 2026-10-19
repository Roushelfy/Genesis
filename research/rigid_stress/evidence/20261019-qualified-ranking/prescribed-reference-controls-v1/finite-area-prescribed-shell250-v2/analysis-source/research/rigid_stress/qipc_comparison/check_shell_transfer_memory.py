"""Compare identical load maps and measured allocation peaks before/after the weight-view fix."""

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import tracemalloc

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def load_transfer(path):
    # Isolate the unchanged numerical class from simulator imports. Both
    # complete source-file hashes remain recorded; execute actual definitions.
    parsed = ast.parse(path.read_text())
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("ShellLoadTransfer", "wrench")
    ]
    assert len(nodes) == 2
    namespace = {"np": np, "dataclass": dataclass}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ShellLoadTransfer"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--before-source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    after_source = Path(__file__).with_name("shell_load_replay.py")
    nodes = 51
    x, y = np.meshgrid(np.linspace(0, 1, nodes), np.linspace(0, 1, nodes))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    corner = np.arange(nodes**2).reshape(nodes, nodes)[:-1, :-1].ravel()
    faces = np.concatenate(
        (
            np.column_stack((corner, corner + 1, corner + nodes)),
            np.column_stack((corner + 1, corner + nodes + 1, corner + nodes)),
        )
    )
    selected = np.arange(0, len(faces), 25)
    points = vertices[faces[selected]].mean(axis=1)
    transfers, peaks = [], []
    for source in (args.before_source, after_source):
        cls = load_transfer(source)
        tracemalloc.start()
        transfer = cls.allocate(vertices, faces, points, 1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        np.testing.assert_array_equal(transfer.selected, faces[selected])
        np.testing.assert_allclose(transfer.barycentric, 1 / 3, rtol=0, atol=2e-14)
        transfers.append(transfer)
        peaks.append(peak)
    before, after = transfers
    for name in ("vertices", "selected", "barycentric", "relative_corners", "inverse_moment"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    rng = np.random.default_rng(1692)
    actual_points = points + rng.normal(scale=0.001, size=points.shape)
    force = rng.normal(size=points.shape)
    old_force, _ = before.apply(actual_points, force)
    new_force, defect = after.apply(actual_points, force)
    np.testing.assert_array_equal(old_force, new_force)
    linear, angular = rng.normal(size=(2, 3))
    work = np.sum(force * (linear + np.cross(angular, actual_points)))
    transferred_work = np.sum(new_force * (linear + np.cross(angular, vertices)))
    np.testing.assert_allclose(work, transferred_work, rtol=1e-12, atol=1e-12)
    assert peaks[1] < peaks[0] / 4, "Candidate-face arrays are still retained per source"
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "before": file_id(args.before_source),
            "after": file_id(after_source),
            "faces": len(faces),
            "sources": len(points),
            "peak_traced_bytes_before_after": peaks,
            "force_arrays_identical": True,
            "point_mapping_arrays_identical": True,
            "wrench_defect": defect.tolist(),
            "rigid_virtual_work_defect": float(transferred_work - work),
            "scope": "Actual class definitions on5000 planar faces/200 interior source points. Same mapping, "
            "loads and rigid work; measured Python/NumPy traced allocation peak. This is not a physical "
            "Shell stress or whole-process memory qualification. The original full250um replay OOM remains retained.",
        },
    )


if __name__ == "__main__":
    main()
