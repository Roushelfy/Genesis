"""Check fixed-block query coverage against independent physical triangle geometry.

Scientific execution requires Slurm rtx-mid/image 1_26. This query-only control keeps original force, radius, position and
geometry bytes unchanged between serial, cell-warp and block-warp paths. Existing bank/trajectory checks qualify loads,
source-point conservation, original elastic residuals and stress fields under their unchanged budgets separately.
"""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class QueryCase:
    name: str
    centers: tuple
    radii: tuple
    slots: tuple = (0,)
    sides: tuple = (0,)
    anchors: tuple = (0,)
    status: tuple = (0,)
    zero_force: bool = False


def _api():
    from genesis.engine.solvers.rigid.stress.planar_loading import (
        PlanarFallback,
        PlanarLoading,
        kernel_planar_query,
        kernel_planar_query_warp,
    )

    from . import check_planar_footprint, check_planar_loading_gpu, check_planar_primitives
    from .check_planar_loading_gpu import unbatched_array

    api = check_planar_loading_gpu._api()
    primitive = check_planar_primitives._native_api()
    return SimpleNamespace(**locals())


def flat_cases(np):
    """Construct dyadic boundaries and bounded random spheres over two connected flat Q4 parents."""
    span, thickness = 2.0**-6, 2.0**-11
    physical = {
        "vertices": np.array(
            [[-span, -span, 0], [0, -span, 0], [span, -span, 0], [-span, span, 0], [0, span, 0], [span, span, 0]]
        ),
        "quads": np.array([[0, 1, 4, 3], [1, 2, 5, 4]]),
        "directors": np.tile([0.0, 0.0, 1.0], (6, 1)),
        "thickness": thickness,
    }
    plane, radius = thickness / 2, span / 16
    cases = [
        QueryCase("shared-parent-edge", ((0.0, 0.0, plane),), (radius,)),
        QueryCase("outer-only", ((-span / 4, 0.0, plane),), (radius,), sides=(1,)),
        QueryCase("inner-only", ((span / 4, 0.0, -plane),), (radius,), sides=(-1,), anchors=(3,)),
        QueryCase("empty", ((4 * span, 0.0, plane),), (radius,)),
        QueryCase("plane-tangent", ((span / 4, 0.0, plane + radius),), (radius,)),
        QueryCase("plane-tangent-inside", ((span / 4, 0.0, np.nextafter(plane + radius, plane)),), (radius,)),
        QueryCase("plane-tangent-outside", ((span / 4, 0.0, np.nextafter(plane + radius, float("inf"))),), (radius,)),
        QueryCase("edge-tangent", ((span + radius, 0.0, plane),), (radius,), anchors=(1,)),
        QueryCase("edge-tangent-inside", ((np.nextafter(span + radius, span), 0.0, plane),), (radius,), anchors=(1,)),
        QueryCase("all-faces", ((0.0, 0.0, plane),), (4 * span,)),
        QueryCase("zero-radius", ((0.0, 0.0, plane),), (0.0,)),
        QueryCase("zero-force", ((0.0, 0.0, plane),), (radius,), zero_force=True),
        QueryCase("invalid-anchor", ((0.0, 0.0, plane),), (radius,), status=(1,)),
        QueryCase("slot-two", ((0.0, 0.0, plane),), (radius,), slots=(2,)),
        QueryCase(
            "two-sources",
            ((-span / 4, 0.0, plane), (span / 4, 0.0, -plane)),
            (radius, radius),
            slots=(0, 2),
            sides=(0, 0),
            anchors=(0, 3),
            status=(0, 0),
        ),
        QueryCase(
            "two-sources-slot-reversed",
            ((span / 4, 0.0, -plane), (-span / 4, 0.0, plane)),
            (radius, radius),
            slots=(0, 2),
            sides=(0, 0),
            anchors=(3, 0),
            status=(0, 0),
        ),
        QueryCase("release", (), (), slots=(), sides=(), anchors=(), status=()),
    ]
    random = np.random.default_rng(8102732)
    for i_case in range(32):
        center = (random.uniform(-1.2 * span, 1.2 * span), random.uniform(-1.2 * span, 1.2 * span), plane)
        cases.append(
            QueryCase(
                f"random-{i_case}",
                (center,),
                (random.uniform(span / 32, span / 4),),
                slots=(i_case % 3,),
                anchors=(0 if center[0] < 0 else 1,),
            )
        )
    return physical, cases


def physical_triangles(np, physical, subdivisions):
    """Generate an independent bilinear chart and its two triangle fans from physical Q4 vertices."""
    coordinates = np.linspace(-1.0, 1.0, subdivisions + 1)
    xi, eta = np.meshgrid(coordinates, coordinates)
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    shapes = 0.25 * (1 + xi[..., None] * signs[:, 0]) * (1 + eta[..., None] * signs[:, 1])
    result = []
    for side in (1, -1):
        vertices = physical["vertices"] + side * physical["thickness"] / 2 * physical["directors"]
        for quad in physical["quads"]:
            grid = shapes @ vertices[quad]
            a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]
            triangles = np.stack((np.stack((a, b, c), axis=-2), np.stack((a, c, d), axis=-2)), axis=-3)
            if side == -1:
                triangles = triangles[..., (0, 2, 1), :]
            result.append(triangles.reshape(-1, 3, 3))
    return np.stack(result)


def positive_triangles(np, triangles, center, radius):
    """Find positive sphere intersections using an independent three-dimensional closest-point distance."""
    first, second = triangles[..., 1, :] - triangles[..., 0, :], triangles[..., 2, :] - triangles[..., 0, :]
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    offset = center - triangles[..., 0, :]
    aa, ab, bb = np.sum(first * first, axis=-1), np.sum(first * second, axis=-1), np.sum(second * second, axis=-1)
    ax, bx = np.sum(offset * first, axis=-1), np.sum(offset * second, axis=-1)
    determinant = aa * bb - ab * ab
    u, v = (bb * ax - ab * bx) / determinant, (aa * bx - ab * ax) / determinant
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    edge = np.roll(triangles, -1, axis=-2) - triangles
    relative = center - triangles
    parameter = np.clip(np.sum(relative * edge, axis=-1) / np.sum(edge * edge, axis=-1), 0, 1)
    edge_distance = np.min(np.sum((relative - parameter[..., None] * edge) ** 2, axis=-1), axis=-1)
    plane_distance = np.sum(offset * normal, axis=-1) ** 2
    return (radius > 0) & (np.where(inside, plane_distance, edge_distance) < radius * radius)


def upload_cases(api, cases):
    np = api.np
    contacts = api.create_contacts(3, len(cases), 1e-4, hertz_profile=True)
    position = np.zeros((3, len(cases), 3))
    force, normal = np.zeros_like(position), np.zeros_like(position)
    radius, friction = np.zeros(position.shape[:2]), np.full(position.shape[:2], 0.5)
    valid, status, anchors, sides = (
        np.zeros(position.shape[:2], dtype=bool),
        *(np.zeros(position.shape[:2], dtype=api.gs.np_int) for _ in range(3)),
    )
    for i_b, case in enumerate(cases):
        for i_source, i_c in enumerate(case.slots):
            position[i_c, i_b], radius[i_c, i_b], valid[i_c, i_b] = case.centers[i_source], case.radii[i_source], True
            force[i_c, i_b], normal[i_c, i_b] = [0.1, 0.0, -1.0], [0.0, 0.0, -1.0]
            if case.zero_force:
                force[i_c, i_b] = 0
            status[i_c, i_b], anchors[i_c, i_b], sides[i_c, i_b] = (
                case.status[i_source],
                case.anchors[i_source],
                case.sides[i_source],
            )
    contacts.position.from_numpy(position)
    contacts.center.from_numpy(position)
    contacts.force.from_numpy(force)
    contacts.normal.from_numpy(normal)
    contacts.radius.from_numpy(radius)
    contacts.friction.from_numpy(friction)
    contacts.valid.from_numpy(valid)
    contacts.status.from_numpy(status)
    contacts.anchor_face.from_numpy(anchors)
    return contacts, sides


def input_identities(api, contacts, geometry):
    values = {
        "position": contacts.position,
        "center": contacts.center,
        "force": contacts.force,
        "normal": contacts.normal,
        "radius": contacts.radius,
        "friction": contacts.friction,
        "valid": contacts.valid,
        "status": contacts.status,
        "anchor_face": contacts.anchor_face,
        "grid_positions": geometry.grid_positions,
        "grid_shapes": geometry.grid_shapes,
        "triangle_nodes": geometry.triangle_nodes,
        "cell_nodes": geometry.cell_nodes,
        "cell_bounds_low": geometry.cell_bounds_low,
        "cell_bounds_high": geometry.cell_bounds_high,
    }
    return {
        name: hashlib.sha256(api.qd_to_numpy(value, transpose=True).tobytes()).hexdigest()
        for name, value in values.items()
    }


def query_fixture(context, args, writer, checks, name, physical, cases, subdivisions, *, exact_cpu):
    api, primitive, np = context.api, context.primitive, context.api.np
    load_info, surface = context.check_planar_primitives._surface(primitive, physical, 24)
    contacts, sides = upload_cases(api, cases)
    loading = context.PlanarLoading(surface.info, load_info, contacts, subdivisions, tasks_per_env=8192)
    loading.state.side_filter.from_numpy(sides)
    label = f"{name}/k{subdivisions}"
    cells_low = context.unbatched_array(api, loading.info.cell_bounds_low)
    cells_high = context.unbatched_array(api, loading.info.cell_bounds_high)
    axis = subdivisions // 4
    for bound, cells, reduction in (("low", cells_low, np.min), ("high", cells_high, np.max)):
        expected = reduction(cells.reshape(len(cells), axis, 4, axis, 4, 3), axis=(2, 4)).reshape(len(cells), -1, 3)
        actual = context.unbatched_array(
            api, loading.info.cell_block_bounds_low if bound == "low" else loading.info.cell_block_bounds_high
        )
        checks.exact(f"{label}/block_{bound}_exact_child_reduction", actual, expected)
    triangles = physical_triangles(np, physical, subdivisions)
    cpu_low = np.min(triangles.reshape(len(triangles), -1, 2, 3, 3), axis=(2, 3))
    cpu_high = np.max(triangles.reshape(len(triangles), -1, 2, 3, 3), axis=(2, 3))
    checks.close(f"{label}/independent_physical_cell_low", cells_low, cpu_low, rtol=1e-10, atol=1e-12)
    checks.close(f"{label}/independent_physical_cell_high", cells_high, cpu_high, rtol=1e-10, atol=1e-12)
    inputs_before = input_identities(api, contacts, loading.info)
    centers = api.qd_to_numpy(contacts.center, transpose=True)
    radii = api.qd_to_numpy(contacts.radius, transpose=True)
    valid = api.qd_to_numpy(contacts.valid, transpose=True)
    force = api.qd_to_numpy(contacts.force, transpose=True)
    status = api.qd_to_numpy(contacts.status, transpose=True)
    writer.arrays(
        f"{label}/contact-inputs.npz",
        position=api.qd_to_numpy(contacts.position, transpose=True),
        center=centers,
        force=force,
        normal=api.qd_to_numpy(contacts.normal, transpose=True),
        radius=radii,
        friction=api.qd_to_numpy(contacts.friction, transpose=True),
        valid=valid,
        status=status,
        anchor_face=api.qd_to_numpy(contacts.anchor_face, transpose=True),
        side_filter=sides.T,
    )
    loaded = valid & (np.linalg.norm(force, axis=-1) > 1e-30)
    expected_tasks = {}
    if exact_cpu:
        for i_b, case in enumerate(cases):
            for i_c in range(3):
                hit = positive_triangles(np, triangles, centers[i_b, i_c], radii[i_b, i_c])
                if not loaded[i_b, i_c] or status[i_b, i_c] != 0:
                    hit.fill(False)
                if sides[i_c, i_b] != 0:
                    hit[
                        (len(hit) // 2 if sides[i_c, i_b] == 1 else 0) : (
                            len(hit) if sides[i_c, i_b] == 1 else len(hit) // 2
                        )
                    ] = False
                expected_tasks[i_b, i_c] = np.argwhere(hit)
    snapshots = {}
    for mode in ("serial", "cells", "blocks"):
        if mode == "serial":
            context.kernel_planar_query(contacts, loading.info, loading.state)
        else:
            context.kernel_planar_query_warp(contacts, loading.info, loading.state, mode == "blocks")
        api.qd.sync()
        checks.condition(
            f"{label}/{mode}/input_bytes_unchanged", inputs_before == input_identities(api, contacts, loading.info)
        )
        snapshot = {
            field: api.qd_to_numpy(value, transpose=True)
            for field, value in (
                ("eligible", loading.state.eligible),
                ("fallback", loading.state.fallback),
                ("reason", loading.state.reason),
                ("task_count", loading.state.task_count),
                ("overflow", loading.state.overflow),
                ("task_start", loading.state.task_start),
                ("task_end", loading.state.task_end),
                ("tasks", loading.state.tasks),
                ("parent_start", loading.state.parent_start),
                ("parent_end", loading.state.parent_end),
                ("parent_faces", loading.state.parent_faces),
                ("parent_connected", loading.state.parent_connected),
            )
        }
        snapshots[mode] = snapshot
        writer.arrays(f"{label}/{mode}.npz", **snapshot)
        for i_b, case in enumerate(cases):
            for i_c in range(3):
                start, end = snapshot["task_start"][i_b, i_c], snapshot["task_end"][i_b, i_c]
                tasks = snapshot["tasks"][i_b, start:end]
                checks.condition(
                    f"{label}/{mode}/{case.name}/slot{i_c}/contiguous_source_range",
                    not len(tasks) or np.all(tasks[:, 2] == i_c),
                )
                if exact_cpu:
                    expected = expected_tasks[i_b, i_c]
                    checks.condition(
                        f"{label}/{mode}/{case.name}/slot{i_c}/independent_eligibility",
                        snapshot["eligible"][i_b, i_c] == (len(expected) > 0),
                    )
                    actual = tasks[:, :2]
                    actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]
                    checks.exact(
                        f"{label}/{mode}/{case.name}/slot{i_c}/independent_positive_triangles", actual, expected
                    )
        if mode != "serial":
            for field in ("eligible", "fallback", "reason", "task_count", "overflow"):
                checks.exact(f"{label}/{mode}/serial_{field}", snapshot[field], snapshots["serial"][field])
    for i_b, case in enumerate(cases):
        for i_c in range(3):
            for kind, start_name, end_name, data_name in (
                ("triangles", "task_start", "task_end", "tasks"),
                ("parents", "parent_start", "parent_end", "parent_faces"),
            ):
                canonical = []
                for mode in ("serial", "cells", "blocks"):
                    snapshot = snapshots[mode]
                    start, end = snapshot[start_name][i_b, i_c], snapshot[end_name][i_b, i_c]
                    values = snapshot[data_name][i_b, start:end]
                    if kind == "triangles":
                        canonical.append(sorted(map(tuple, values[:, :2])))
                    else:
                        flags = snapshot["parent_connected"][i_b, start:end]
                        canonical.append(sorted(set(zip(values.tolist(), flags.tolist()))))
                checks.condition(f"{label}/{case.name}/slot{i_c}/{kind}_set_cells_blocks", canonical[1] == canonical[2])
                checks.condition(
                    f"{label}/{case.name}/slot{i_c}/{kind}_set_serial_blocks", canonical[0] == canonical[2]
                )
    if name != "flat":
        checks.condition(
            f"{label}/independent_unsupported_footprint_rejected",
            not snapshots["blocks"]["eligible"].any() and snapshots["blocks"]["fallback"].any(),
        )
    small = context.PlanarLoading(
        surface.info, load_info, contacts, subdivisions, geometry=loading.info, tasks_per_env=1
    )
    small.state.side_filter.from_numpy(sides)
    overflow = []
    for mode in ("cells", "blocks"):
        context.kernel_planar_query_warp(contacts, small.info, small.state, mode == "blocks")
        current = {
            field: api.qd_to_numpy(value, transpose=True)
            for field, value in (
                ("eligible", small.state.eligible),
                ("fallback", small.state.fallback),
                ("reason", small.state.reason),
                ("overflow", small.state.overflow),
            )
        }
        overflow.append(current)
        writer.arrays(f"{label}/overflow-{mode}.npz", **current)
        for i_b in np.flatnonzero(current["overflow"]):
            checks.condition(
                f"{label}/{mode}/overflow-env{i_b}/whole_environment_fallback",
                not current["eligible"][i_b].any()
                and current["fallback"][i_b, loaded[i_b]].all()
                and np.all(current["reason"][i_b, loaded[i_b]] == context.PlanarFallback.CAPACITY),
            )
        if name == "flat":
            checks.condition(f"{label}/{mode}/bounded_capacity_really_overflows", current["overflow"].any())
    for field in overflow[0]:
        checks.exact(f"{label}/overflow_cells_blocks/{field}", overflow[0][field], overflow[1][field])
    writer.json(
        f"{label}/inputs.json",
        {
            "cases": [case.name for case in cases],
            "hashes": inputs_before,
            "scope": "Independent physical geometry/query control; no elastic solve or stress-error claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--subdivisions", type=int, nargs="+", choices=(4, 8, 16, 32), default=[4, 8, 16, 32])
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh output under /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    context = _api()
    api, np = context.api, context.api.np
    if not Path(api.gs.__file__).resolve().is_relative_to(args.git_source.resolve()):
        raise ValueError("The imported Genesis package is outside the requested frozen engine source.")
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    writer, checks = api.Writer(args.output, api), api.Checks(api)
    writer.json(
        "start.json",
        {
            "identity": api.runtime_identity(),
            "git_source": str(args.git_source),
            "git_source_revision": subprocess.check_output(
                ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "native_package": api.gs.__file__,
            "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        },
    )
    physical, cases = flat_cases(np)
    writer.arrays("flat-physical.npz", **physical)
    for subdivisions in args.subdivisions:
        query_fixture(context, args, writer, checks, "flat", physical, cases, subdivisions, exact_cpu=True)
    vertices, quads, directors = context.check_planar_footprint._book(np, "closed")
    source = context.check_planar_footprint._source(np, vertices, quads, directors, 0.0005, 0, 1, 0.016, False, 0.001)
    physical = {"vertices": vertices, "quads": quads, "directors": directors, "thickness": 0.0005}
    query_fixture(
        context,
        args,
        writer,
        checks,
        "closed-book-tip",
        physical,
        [QueryCase("disconnected-tip", (source["point"],), (source["radius"],))],
        32,
        exact_cpu=False,
    )
    physical = {
        "vertices": np.array([[-1, -1, 6], [1, -1, -6], [1, 1, 6], [-1, 1, -6]]) * 1e-3,
        "quads": np.array([[0, 1, 2, 3]]),
        "directors": np.tile([0.0, 0.0, 1.0], (4, 1)),
        "thickness": 5e-5,
    }
    point = (0.0008, 0.0008, 0.00384 + physical["thickness"] / 2)
    query_fixture(
        context,
        args,
        writer,
        checks,
        "strong-saddle",
        physical,
        [QueryCase("disconnected-single-parent", (point,), (0.003,))],
        32,
        exact_cpu=False,
    )
    checks.flush(writer)
    passed = all(row["passed"] for row in checks.rows)
    writer.json(
        "result.json",
        {
            "complete": True,
            "passed": passed,
            "checks": len(checks.rows),
            "failed_checks": [row["name"] for row in checks.rows if not row["passed"]],
            "scope": "Query coverage, original input bytes, conservative block bounds, routing/component and complete-environment overflow controls. Load/wrench/residual/full-field budgets remain unchanged in separate existing bank/trajectory qualification.",
        },
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
