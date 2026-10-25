"""GPU common-mother recovery control on a validated nested planar fixture.

CPU reference operators/solutions are fixture inputs, never a native production
preparation path. Timings cover GPU contact grouping and recovery only, excluding
pressure preparation and solves. Native curved-surface mapping is not assumed.
"""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy import sparse
from threadpoolctl import threadpool_limits

import genesis as gs
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.atlas_study.shell.plate_atlas import L, Plate, knots, refined_knots

from .artifacts import runtime_identity, write_json
from .check_atlas_candidates import embedding
from .check_response_candidates import relative, timings


@dataclass
class CSR:
    start: object
    columns: object
    values: object

    @classmethod
    def upload(cls, matrix):
        matrix = matrix.tocsr()
        fields = [
            V(dtype=dtype, shape=array.shape)
            for dtype, array in [(gs.qd_int, matrix.indptr), (gs.qd_int, matrix.indices), (gs.qd_float, matrix.data)]
        ]
        for field, array in zip(fields, [matrix.indptr, matrix.indices, matrix.data]):
            field.from_numpy(array)
        return cls(*fields)


@qd.kernel
def kernel_weights(
    overlap: float,
    halfwidth: float,
    anchors: qd.Tensor,
    contacts: qd.Tensor,
    weights: qd.Tensor,
    active: qd.Tensor,
    mode: qd.template(),
):
    for i_c, i_b in qd.ndrange(contacts.shape[0], weights.shape[2]):
        x, y, radius = contacts[i_c, 0], contacts[i_c, 1], contacts[i_c, 2]
        distance_min = 1.0
        nearest = 0
        for i_m in range(anchors.shape[0]):
            distance = qd.sqrt((x - anchors[i_m, 0]) ** 2 + (y - anchors[i_m, 1]) ** 2)
            if distance < distance_min:
                distance_min, nearest = distance, i_m
        total = 0.0
        covered = True
        for i_m in range(anchors.shape[0]):
            distance = qd.sqrt((x - anchors[i_m, 0]) ** 2 + (y - anchors[i_m, 1]) ** 2)
            value = 0.0
            if qd.static(mode == "nearest"):
                value = gs.qd_float(i_m == nearest)
            elif qd.static(mode == "overlap"):
                t = qd.max(0.0, 1.0 - (distance - distance_min) / overlap)
                value = t * t * (3.0 - 2.0 * t)
            else:
                value = qd.max(0.0, 1.0 - (distance / 0.02) ** 2) ** 2
            weights[i_m, i_c, i_b] = value
            total += value
            if value > 0.0 and qd.max(qd.abs(x - anchors[i_m, 0]), qd.abs(y - anchors[i_m, 1])) + radius > halfwidth:
                covered = False
        for i_m in range(anchors.shape[0]):
            weights[i_m, i_c, i_b] = weights[i_m, i_c, i_b] / total if covered else 0.0
        weights[anchors.shape[0], i_c, i_b] = 0.0 if covered else 1.0
    for i_m, i_b in qd.ndrange(active.shape[0], active.shape[1]):
        has_load = False
        for i_c in range(contacts.shape[0]):
            has_load = has_load or weights[i_m, i_c, i_b] > 0.0
        active[i_m, i_b] = has_load


@qd.kernel
def kernel_group(bank: qd.Tensor, owner: qd.Tensor, weights: qd.Tensor, grouped: qd.Tensor):
    for i, i_b in qd.ndrange(grouped.shape[0], grouped.shape[1]):
        model = owner[i]
        value = 0.0
        for i_c in range(bank.shape[1]):
            # Heterogeneous signs/amplitudes; per-contact location and radius remain fixture-defined.
            scale = 0.75 + 0.5 * ((i_b * 37 + i_c * 11) % 101) / 100.0
            value += weights[model, i_c, i_b] * bank[i, i_c] * scale
        grouped[i, i_b] = value


@qd.kernel
def kernel_product(
    start: qd.Tensor,
    columns: qd.Tensor,
    values: qd.Tensor,
    vector: qd.Tensor,
    output: qd.Tensor,
    owner: qd.Tensor,
    active: qd.Tensor,
    skip: qd.template(),
):
    for row, i_b in qd.ndrange(output.shape[0], output.shape[1]):
        value = 0.0
        for entry in range(start[row], start[row + 1]):
            column = columns[entry]
            enabled = True
            if qd.static(skip):
                model = owner[column]
                enabled = active[model, i_b]
            if enabled:
                value += values[entry] * vector[column, i_b]
        output[row, i_b] = value


@qd.kernel(graph=True)
def kernel_peak(tensor: qd.Tensor, peak: qd.Tensor):
    for i_b in range(peak.shape[0]):
        peak[i_b] = 0.0
    for sample, i_b in qd.ndrange(tensor.shape[0] // 3, tensor.shape[1]):
        a, b, t = tensor[3 * sample, i_b], tensor[3 * sample + 1, i_b], tensor[3 * sample + 2, i_b]
        vm = qd.sqrt(qd.max(0.0, a * a - a * b + b * b + 3.0 * t * t))
        qd.atomic_max(peak[i_b], vm)


def recover(common, skip, transform, independent, mother_operator, grouped, mother_q, output, peak, owner, active):
    if common:
        kernel_product(transform.start, transform.columns, transform.values, grouped, mother_q, owner, active, skip)
        operator, vector = mother_operator, mother_q
    else:
        operator, vector = independent, grouped
    kernel_product(
        operator.start, operator.columns, operator.values, vector, output, owner, active, skip and not common
    )
    kernel_peak(output, peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--library-size", type=int, choices=(3, 5), default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    result = dict(
        identity=runtime_identity(),
        complete=False,
        envs=args.envs,
        library_size=args.library_size,
        scope=__doc__,
        rows=[],
    )
    write_json(args.output / "result.json", result)
    anchors = np.array([[-0.006, -0.003464101615], [0.006, -0.003464101615], [0.0, 0.00692820323]])
    if args.library_size == 5:
        anchors = np.concatenate((anchors, [[-0.007, 0.006], [0.007, 0.006]]))
    axis = np.linspace(-0.0193, 0.0193, 29)
    probes = np.array([[x, y] for y in axis for x in axis])
    width = 0.009
    axes = [(refined_knots(a[0], width, 0.002), refined_knots(a[1], width, 0.002)) for a in anchors]
    models = [Plate(f"feature{i}", x, y, probes) for i, (x, y) in enumerate(axes)]
    base = knots(-L / 2, L / 2, 0.005)
    mother = Plate(
        "mother",
        np.unique(np.round(np.concatenate([a[0] for a in axes] + [base]), 12)),
        np.unique(np.round(np.concatenate([a[1] for a in axes] + [base]), 12)),
        probes,
    )
    transforms = [embedding(m, mother) for m in models]
    maps = [
        sparse.hstack([m.S for m in [*models, mother]], format="csr"),
        sparse.hstack([*transforms, sparse.eye(mother.m.ndof)], format="csr"),
        mother.S,
    ]
    independent, transform, mother_operator = [CSR.upload(matrix) for matrix in maps]
    contacts = [
        (-0.007, -0.002, 0.04, 0.2, -0.1, 0.002),
        (0.007, 0.001, 0.07, -0.3, 0.1, 0.0025),
        (0.0, 0.0, 0.03, 0.1, 0.3, 0.001),
    ]
    bank_host = np.concatenate([m.solve(np.column_stack([m.load(c)[0] for c in contacts])) for m in [*models, mother]])
    owners = np.concatenate([np.full(m.m.ndof, i, dtype=np.int32) for i, m in enumerate([*models, mother])])
    bank = V(dtype=gs.qd_float, shape=bank_host.shape)
    bank.from_numpy(bank_host)
    owner = V(dtype=gs.qd_int, shape=owners.shape)
    owner.from_numpy(owners)
    anchor = V(dtype=gs.qd_float, shape=anchors.shape)
    anchor.from_numpy(anchors)
    points = np.array([[c[0], c[1], c[-1]] for c in contacts])
    contact = V(dtype=gs.qd_float, shape=points.shape)
    contact.from_numpy(points)
    weights = V(dtype=gs.qd_float, shape=(len(models) + 1, len(contacts), args.envs))
    active = V(dtype=gs.qd_bool, shape=(len(models) + 1, args.envs))
    grouped = V(dtype=gs.qd_float, shape=(len(owners), args.envs))
    mother_q = V(dtype=gs.qd_float, shape=(mother.m.ndof, args.envs))
    output = V(dtype=gs.qd_float, shape=(mother.S.shape[0], args.envs))
    peak = V(dtype=gs.qd_float, shape=(args.envs,))
    result["operator_bytes"] = [m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in maps]
    result["mother_workspace_bytes"] = mother.m.ndof * args.envs * 8
    for mode in ["nearest", "overlap", "blend"]:
        selection = partial(kernel_weights, 0.002, width, anchor, contact, weights, active, mode)
        select_times = timings(selection, 5, 30)
        grouping_times = timings(partial(kernel_group, bank, owner, weights, grouped), 5, 30)
        prior = None
        for common, skip in [(False, False), (True, False), (False, True), (True, True)]:
            operation = partial(
                recover,
                common,
                skip,
                transform,
                independent,
                mother_operator,
                grouped,
                mother_q,
                output,
                peak,
                owner,
                active,
            )
            samples = timings(operation, 5, 30)
            field = qd_to_numpy(output, transpose=True)
            if prior is None:
                prior = field
            error = relative(field, prior)
            assert error < 1e-10
            result["rows"].append(
                dict(
                    mode=mode,
                    common_mother=common,
                    skip_inactive=skip,
                    selection_seconds=select_times,
                    grouping_seconds=grouping_times,
                    recovery_seconds=samples,
                    tensor_relative=error,
                    fallback_contacts=float(qd_to_numpy(weights)[-1].sum() / args.envs),
                )
            )
            write_json(args.output / "result.json", result)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
