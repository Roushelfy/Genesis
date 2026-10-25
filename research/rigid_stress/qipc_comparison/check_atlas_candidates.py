"""Bounded CPU controls for continuous few-mesh selection and common-mother recovery.

The fixture is an exactly planar clamped shell with independently integrated
finite pressure and eccentric friction. It supplies spatial-approximation
and algebra controls, separate from native evolving Franka performance.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from research.rigid_stress.atlas_study.shell.plate_atlas import L, Plate, knots, measure, refined_knots, rel

from .artifacts import runtime_identity, write_json


def embedding(model, mother):
    points = mother.m.vertices[:, :2]
    ix = np.clip(np.searchsorted(model.x, points[:, 0], side="right") - 1, 0, len(model.x) - 2)
    iy = np.clip(np.searchsorted(model.y, points[:, 1], side="right") - 1, 0, len(model.y) - 2)
    u = (points[:, 0] - model.x[ix]) / (model.x[ix + 1] - model.x[ix])
    v = (points[:, 1] - model.y[iy]) / (model.y[iy + 1] - model.y[iy])
    weights = np.column_stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v))
    nodes = model.m.quads[iy * (len(model.x) - 1) + ix]
    rows = np.broadcast_to(6 * np.arange(len(points))[:, None, None] + np.arange(6), (*nodes.shape, 6))
    columns = 6 * nodes[:, :, None] + np.arange(6)
    values = np.broadcast_to(weights[:, :, None], columns.shape)
    return sparse.coo_matrix(
        (values.ravel(), (rows.ravel(), columns.ravel())), shape=(mother.m.ndof, model.m.ndof)
    ).tocsr()


def weights(point, anchors, mode, overlap):
    distances = np.linalg.norm(anchors - point, axis=1)
    if mode == "nearest":
        values = np.zeros(len(anchors))
        values[np.argmin(distances)] = 1
    elif mode == "overlap":
        t = np.maximum(0, 1 - (distances - distances.min()) / overlap)
        values = t * t * (3 - 2 * t)
    else:
        support = 0.02
        values = np.maximum(0, 1 - (distances / support) ** 2) ** 2
        if mode == "top_two":
            values[np.argsort(distances)[2:]] = 0
    return values / values.sum()


def fields(models, contacts):
    loads, solutions, tensors = [], [], []
    for model in models:
        load = np.column_stack([model.load(c)[0] for c in contacts])
        solution = model.solve(load)
        loads.append(load)
        solutions.append(solution)
        tensors.append(np.asarray(model.S @ solution).reshape((2, -1, 3, len(contacts))))
    return loads, solutions, tensors


def condensation(output):
    axis = knots(-L / 2, L / 2, 0.005)
    fine_axis = np.sort(np.unique(np.r_[axis, [-0.0075, -0.0025, 0.0025, 0.0075]]))
    probes = np.array([[x, y] for x in [-0.0131, -0.0033, 0.0061] for y in [-0.0123, 0.0021, 0.0117]])
    coarse = Plate("coarse_condensation", axis, axis, probes)
    mother = Plate("mother_condensation", fine_axis, fine_axis, probes)
    interpolation = embedding(coarse, mother)[mother.free][:, coarse.free]
    coarse_vertices = {(round(x, 12), round(y, 12)) for x, y in coarse.m.vertices[:, :2]}
    is_coarse = np.repeat([(round(x, 12), round(y, 12)) in coarse_vertices for x, y in mother.m.vertices[:, :2]], 6)
    bubble_rows = np.flatnonzero(~is_coarse[mother.free])
    bubble = sparse.eye(len(mother.free), format="csc")[:, bubble_rows]
    transform = sparse.hstack((interpolation, bubble), format="csc")
    assert transform.shape[0] == transform.shape[1]
    stiffness = (transform.T @ mother.m.K[mother.free][:, mother.free] @ transform).tocsc()
    contacts = [(-0.006, 0.001, 0.04, 0.2, -0.1, 0.002), (0.007, -0.002, 0.07, -0.1, 0.3, 0.0025)]
    load = np.column_stack([mother.load(c)[0] for c in contacts])
    rhs = transform.T @ load[mother.free]
    reference = mother.solve(load)
    rows = []
    for reverse in [False, True]:
        coarse_ids = np.arange(len(coarse.free))
        bubble_ids = np.arange(len(coarse.free), len(mother.free))
        retained, eliminated = (bubble_ids, coarse_ids) if reverse else (coarse_ids, bubble_ids)
        started = time.perf_counter()
        a = stiffness[eliminated][:, eliminated]
        b = stiffness[eliminated][:, retained]
        factor = splu(a)
        harmonic = factor.solve(b.toarray())
        schur = stiffness[retained][:, retained].toarray() - b.T @ harmonic
        schur_factor = splu(sparse.csc_matrix(schur))
        prepared = time.perf_counter() - started
        samples = []
        result = None
        for _ in range(5):
            started = time.perf_counter()
            particular = factor.solve(rhs[eliminated])
            q_retained = schur_factor.solve(rhs[retained] - b.T @ particular)
            solution = np.zeros_like(rhs)
            solution[retained] = q_retained
            solution[eliminated] = particular - harmonic @ q_retained
            result = transform @ solution
            samples.append(time.perf_counter() - started)
        error = rel(result, reference[mother.free])
        residual = rel(mother.m.K[mother.free][:, mother.free] @ result, load[mother.free])
        rows.append(
            dict(
                direction="eliminate_coarse" if reverse else "eliminate_local",
                retained=len(retained),
                eliminated=len(eliminated),
                preparation_seconds=prepared,
                solve_seconds=samples,
                harmonic_bytes=harmonic.nbytes,
                schur_bytes=schur.nbytes,
                displacement_relative=error,
                residual_relative=residual,
            )
        )
        assert max(error, residual) < 1e-8
    write_json(
        output / "condensation.json",
        dict(
            passed=True,
            rows=rows,
            scope="Exact hierarchical coarse/local Schur coupling, all distributed load terms retained.",
        ),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        identity=runtime_identity(require_gpu_inventory=False), complete=False, rows=[], switches=[], mother_identity=[]
    )
    write_json(args.output / "result.json", result)
    p = np.linspace(-0.0193, 0.0193, 29)
    probes = np.array([[x, y] for y in p for x in p])
    anchors = np.array([[-0.006, -0.003464101615], [0.006, -0.003464101615], [0.0, 0.00692820323]])
    base = knots(-L / 2, L / 2, 0.005)
    modes = ["blend", "nearest", "top_two", "overlap"]
    for halfwidth in [0.006, 0.009]:
        axes = [(refined_knots(a[0], halfwidth, 0.002), refined_knots(a[1], halfwidth, 0.002)) for a in anchors]
        models = [Plate(f"feature{i}", x, y, probes) for i, (x, y) in enumerate(axes)]
        mother = Plate(
            "mother",
            np.unique(np.round(np.concatenate([a[0] for a in axes] + [base]), 12)),
            np.unique(np.round(np.concatenate([a[1] for a in axes] + [base]), 12)),
            probes,
        )
        embeddings = [embedding(model, mother) for model in models]
        result.setdefault("libraries", []).append(
            dict(
                halfwidth=halfwidth,
                models=[m.metadata() for m in models],
                mother=mother.metadata(),
                embedding_bytes=sum(t.data.nbytes + t.indices.nbytes + t.indptr.nbytes for t in embeddings),
            )
        )
        contact_sets = [[(x, 0.00037, 0.05, 0.25, -0.1, 0.0024)] for x in [-0.008, -0.003, 0.0, 0.003, 0.008]]
        contact_sets += [
            [(-0.007, -0.002, 0.04, 0.2, -0.1, 0.002), (0.007, 0.001, 0.07, -0.3, 0.1, 0.0025)],
            [(0.0, 0.0, 0.05, 0.2, 0.1, 0.0035)],
            [(0.014, 0.0, 0.05, 0.1, 0.1, 0.003)],
        ]
        for case, contacts in enumerate(contact_sets):
            loads, solutions, tensors = fields([*models, mother], contacts)
            reference = tensors[-1].sum(axis=-1)
            for mode in modes:
                for overlap in [0.0005, 0.002] if mode == "overlap" else [0.002]:
                    total = np.zeros_like(reference)
                    common = np.zeros(mother.m.ndof)
                    grouped = [np.zeros(m.m.ndof) for m in models]
                    fallback, contributors = 0, []
                    for j, contact in enumerate(contacts):
                        w = weights(np.array(contact[:2]), anchors, mode, overlap)
                        coverage = np.max(np.abs(anchors - contact[:2]), axis=1) + contact[-1] <= halfwidth
                        if not coverage[w > 0].all():
                            fallback += 1
                            total += tensors[-1][..., j]
                            common += solutions[-1][:, j]
                            contributors.append(["mother"])
                        else:
                            contributors.append(np.flatnonzero(w > 0).tolist())
                            for i in range(len(models)):
                                grouped[i] += w[i] * loads[i][:, j]
                                total += w[i] * tensors[i][..., j]
                                common += embeddings[i] @ (w[i] * solutions[i][:, j])
                    recovered = np.asarray(mother.S @ common).reshape(total.shape)
                    identity = rel(recovered, total)
                    assert identity < 1e-10
                    grouped_tensor = sum(
                        np.asarray(m.S @ m.solve(load)).reshape(total.shape) for m, load in zip(models, grouped)
                    )
                    if not fallback:
                        assert rel(grouped_tensor, total) < 1e-10
                    metrics = measure(total, reference)
                    result["rows"].append(
                        dict(
                            halfwidth=halfwidth,
                            case=case,
                            mode=mode,
                            overlap=overlap,
                            contributors=contributors,
                            fallback_contacts=fallback,
                            common_mother_relative=identity,
                            metrics=metrics,
                            budget_passed=metrics["tensor_Frobenius_relative"] < 0.1
                            and metrics["peak_vm_relative"] < 0.2,
                        )
                    )
            write_json(args.output / "result.json", result)
        # Cross the three-region junction with smooth physical loads and full raw selector responses.
        for mode in modes:
            pair = []
            for x in [-1e-8, 1e-8]:
                c = (x, 0.0, 0.05, 0.2, -0.1, 0.001)
                _, _, tensors = fields([*models, mother], [c])
                w = weights(np.array(c[:2]), anchors, mode, 0.0005)
                pair.append(
                    (
                        sum(w[i] * tensors[i][..., 0] for i in range(len(models))),
                        tensors[-1][..., 0],
                        np.flatnonzero(w > 0).tolist(),
                    )
                )
            result["switches"].append(
                dict(
                    halfwidth=halfwidth,
                    mode=mode,
                    delta_m=2e-8,
                    selector_jump=rel(pair[1][0], pair[0][0]),
                    physical_change=rel(pair[1][1], pair[0][1]),
                    contributors=[pair[0][2], pair[1][2]],
                )
            )
        write_json(args.output / "result.json", result)
    condensation(args.output)
    result["complete"] = True
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
