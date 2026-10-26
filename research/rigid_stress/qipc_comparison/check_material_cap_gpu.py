"""Check native material caps without constructing any stiffness or factor.

Scientific imports follow the Slurm rtx-mid/image 1_26 guard. Native execution
receives physical geometry/source inputs only. SciPy and the original CPU load
adapter are untimed, independent oracles. Strict chart/element/node identity is
reported separately from independently checked physical assembly equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity, write_json

REPOSITORY = Path(__file__).resolve().parents[3]
STRATEGIES = ("continuous", "nearest", "narrow")


def _api():
    import numpy as np
    import quadrants as qd
    import scipy
    from scipy.special import roots_jacobi
    from threadpoolctl import threadpool_limits

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.atlas_selection import AtlasSelection
    from genesis.engine.solvers.rigid.stress.material_cap import (
        MaterialCap,
        allocate_material_cap_sources,
        build_material_cap_geometry,
    )
    from genesis.utils.array_class import V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    from research.rigid_stress.atlas_study.cpu_repro.fixtures import cases_manifest, make_cases
    from research.rigid_stress.atlas_study.cpu_repro.geometry import _ray_inverse, authored_egg, make_geometry
    from research.rigid_stress.atlas_study.cpu_repro.loads import (
        Source,
        assemble_sources,
        cap_rule,
        force_for_source,
        source_directions,
    )
    from research.rigid_stress.atlas_study.cpu_repro.selection import selection_weights

    return SimpleNamespace(**locals())


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


class Checks:
    def __init__(self, api):
        self.np = api.np
        self.rows = []

    def condition(self, name, passed, *, gating=True, **details):
        self.rows.append(dict(name=name, passed=bool(passed), gating=gating, **details))

    def close(self, name, actual, expected, *, rtol=1e-10, atol=1e-13, columns=False):
        np = self.np
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.condition(name, False, actual_shape=list(actual.shape), expected_shape=list(expected.shape))
            return
        difference = actual - expected
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        delta = (
            np.linalg.norm(difference.reshape((-1, difference.shape[-1])), axis=0)
            if columns
            else np.linalg.norm(difference)
        )
        scale = (
            np.linalg.norm(expected.reshape((-1, expected.shape[-1])), axis=0) if columns else np.linalg.norm(expected)
        )
        finite = finite and bool(np.isfinite(delta).all() and np.isfinite(scale).all())
        self.condition(
            name,
            finite and bool(np.all(delta <= atol + rtol * scale)),
            absolute_l2_max=float(np.max(delta)) if finite else None,
            relative_l2_max=float(np.max(delta / np.maximum(scale, 1e-30))) if finite else None,
            maximum_absolute=float(np.max(np.abs(difference), initial=0.0)) if finite else None,
            rtol=rtol,
            absolute_l2_floor=atol,
            population="each source column" if columns else "complete array",
        )

    def exact(self, name, actual, expected, *, gating=True):
        np = self.np
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.condition(
            name,
            np.array_equal(actual, expected),
            gating=gating,
            comparison="exact",
            differing_entries=int(np.count_nonzero(actual != expected)) if actual.shape == expected.shape else None,
            actual_shape=list(actual.shape),
            expected_shape=list(expected.shape),
        )

    def passed(self):
        return all(row["passed"] for row in self.rows if row["gating"])


class Output:
    def __init__(self, api, directory):
        self.api, self.directory, self.inventory = api, directory, {}

    def arrays(self, relative, **values):
        np = self.api.np
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(value) for key, value in values.items()}
        if any(array.dtype.hasobject for array in arrays.values()):
            raise TypeError("Oracle artifacts must not contain object arrays.")
        np.savez_compressed(path, **arrays)
        self.inventory[relative] = dict(
            **file_id(path),
            arrays={
                key: dict(
                    shape=list(array.shape),
                    dtype=array.dtype.str,
                    sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                )
                for key, array in arrays.items()
            },
        )


def _vector(api, values):
    tensor = api.V_VEC(values.shape[-1], dtype=api.gs.qd_float, shape=values.shape[:-1])
    tensor.from_numpy(values.astype(api.gs.np_float, copy=False))
    return tensor


def _integer(api, values):
    tensor = api.V(dtype=api.gs.qd_int, shape=values.shape)
    tensor.from_numpy(values.astype(api.gs.np_int, copy=False))
    return tensor


def _source_inputs(api, cases):
    sources = api.allocate_material_cap_sources(len(cases["radius"]))
    for field, key, integer in (
        (sources.direction, "direction", False),
        (sources.radius, "radius", False),
        (sources.force, "force", False),
        (sources.normal_force, "normal_force", False),
        (sources.friction, "friction", False),
        (sources.tangent, "tangent", False),
        (sources.cap_tangent, "cap_tangent", False),
        (sources.side, "side", True),
        (sources.environment, "source_environment", True),
        (sources.explicit_force, "has_force", True),
        (sources.has_tangent, "has_tangent", True),
        (sources.has_cap_tangent, "has_cap_tangent", True),
    ):
        field.from_numpy(cases[key].astype(api.gs.np_int if integer else api.gs.np_float, copy=False))
    return sources


def _cpu_sources(api, cases):
    return tuple(
        api.Source(
            direction=cases["direction"][i],
            radius=float(cases["radius"][i]),
            force=cases["force"][i] if cases["has_force"][i] else None,
            normal_force=float(cases["normal_force"][i]),
            friction=float(cases["friction"][i]),
            tangent=cases["tangent"][i] if cases["has_tangent"][i] else None,
            cap_tangent=cases["cap_tangent"][i] if cases["has_cap_tangent"][i] else None,
            side=int(cases["side"][i]),
            name=str(cases["source_names"][i]),
        )
        for i in range(len(cases["radius"]))
    )


def _cpu_loads(api, mother, sources, order):
    np = api.np
    raw = api.assemble_sources(mother, sources, order=order, audit=True)
    sample_directions = np.stack([api.source_directions(source, order)[0] for source in sources])
    location = mother.geometry.locate(sample_directions.reshape((-1, 3)))
    n_sources, n_samples = sample_directions.shape[:2]
    nodes = location.nodes.reshape((n_sources, n_samples, 4))
    shape = location.weights.reshape((n_sources, n_samples, 4))
    positions = location.positions.reshape((n_sources, n_samples, 3))
    forces = np.array([api.force_for_source(mother, source) for source in sources])
    sides = np.array([source.side for source in sources])
    physical = positions + 0.5 * mother.model.h * sides[:, None, None] * np.einsum(
        "sqi,sqij->sqj", shape, mother.geometry.directors[nodes]
    )
    radial, angular, weights = api.cap_rule(order)
    nodal = raw.reshape((-1, 6, n_sources))
    resultant_force = nodal[:, :3].sum(axis=0).T
    resultant_moment = (
        (np.cross(mother.geometry.vertices[:, :, None], nodal[:, :3], axisa=1, axisb=1, axisc=1) + nodal[:, 3:])
        .sum(axis=0)
        .T
    )
    return dict(
        raw=raw,
        source_force=forces,
        cap_directions=sample_directions,
        location_nodes=nodes,
        location_weights=shape,
        location_positions=positions,
        location_normals=location.normals.reshape((n_sources, n_samples, 3)),
        location_macro_ids=location.macro_ids.reshape((n_sources, n_samples)),
        location_element_ids=location.element_ids.reshape((n_sources, n_samples)),
        location_local=location.local.reshape((n_sources, n_samples, 2)),
        physical_positions=physical,
        expected_moment=np.einsum("sqj,q->sj", np.cross(physical, forces[:, None]), weights),
        recovered_force=resultant_force,
        recovered_moment=resultant_moment,
        radial=radial,
        angular=angular,
        cap_weights=weights,
    )


def _native_loads(api, cap, geometry, sources, model_input, cases):
    np = api.np
    state = cap.assemble(geometry, sources)
    read = api.qd_to_numpy
    raw_blocks = read(state.raw_force)
    shape, nodes = read(state.sample_shape), read(state.sample_nodes)
    positions = read(state.sample_positions)
    physical = positions + 0.5 * float(model_input["thickness"]) * cases["side"][:, None, None] * np.einsum(
        "sqi,sqij->sqj", shape, model_input["directors"][nodes]
    )
    return dict(
        raw=raw_blocks.transpose((0, 2, 1)).reshape((-1, len(cases["radius"]))),
        raw_blocks=raw_blocks,
        source_force=read(state.force),
        cap_directions=read(state.sample_directions),
        location_nodes=nodes,
        location_weights=shape,
        location_positions=positions,
        location_macro_ids=read(state.sample_macro),
        location_element_ids=read(state.sample_element),
        location_local=read(state.sample_local),
        physical_positions=physical,
        expected_moment=read(state.expected_moment),
        recovered_force=read(state.resultant_force),
        recovered_moment=read(state.resultant_moment),
        valid=read(state.valid),
        sample_valid=read(state.sample_valid),
        failed_samples=read(state.failed_samples),
        scanned_samples=read(state.scanned_samples),
    )


def _location_checks(api, output, checks, prefix, native, cpu, cpu_geometry):
    np = api.np
    macro_difference = native["location_macro_ids"] != cpu["location_macro_ids"]
    element_difference = native["location_element_ids"] != cpu["location_element_ids"]
    node_difference = np.any(native["location_nodes"] != cpu["location_nodes"], axis=-1)
    for key in ("macro_ids", "element_ids", "nodes"):
        checks.exact(f"{prefix}/strict_location_{key}", native[f"location_{key}"], cpu[f"location_{key}"], gating=False)
    # Match nodal shape functions by physical node ID, including zero-weight
    # corners of either neighboring fine cell. No chart-ID mismatch is hidden.
    same_node = native["location_nodes"][..., :, None] == cpu["location_nodes"][..., None, :]
    cpu_at_native = np.sum(same_node * cpu["location_weights"][..., None, :], axis=-1)
    native_at_cpu = np.sum(same_node * native["location_weights"][..., :, None], axis=-2)
    nodal_error = np.maximum(
        np.max(np.abs(native["location_weights"] - cpu_at_native), axis=-1),
        np.max(np.abs(cpu["location_weights"] - native_at_cpu), axis=-1),
    )
    position_error = np.linalg.norm(native["location_positions"] - cpu["location_positions"], axis=-1)
    physical_error = np.linalg.norm(native["physical_positions"] - cpu["physical_positions"], axis=-1)
    directions = native["cap_directions"].reshape((-1, 3))
    native_macro = native["location_macro_ids"].reshape(-1)
    admissible_local, admissible = api._ray_inverse(
        cpu_geometry.macro_coefficients[native_macro], api.authored_egg(directions)
    )
    local_error = np.linalg.norm(
        admissible_local.reshape(native["location_local"].shape) - native["location_local"], axis=-1
    )
    checks.condition(
        f"{prefix}/native_charts_cpu_admissible", admissible.all(), rejected=int(np.count_nonzero(~admissible))
    )
    for name, error, tolerance, unit in (
        ("midsurface_position", position_error, 1e-12, "m"),
        ("physical_offset_position", physical_error, 1e-12, "m"),
        ("node_mapped_shape", nodal_error, 1e-10, "dimensionless"),
        ("cpu_inverse_native_chart_local", local_error, 2e-10, "dimensionless"),
    ):
        maximum = _number(np.max(error))
        checks.condition(
            f"{prefix}/{name}",
            np.isfinite(error).all() and maximum is not None and maximum <= tolerance,
            maximum=maximum,
            tolerance=tolerance,
            unit=unit,
        )
    output.arrays(
        f"{prefix}/location-differences.npz",
        macro_difference=macro_difference,
        element_difference=element_difference,
        node_difference=node_difference,
        differing_samples=np.argwhere(macro_difference | element_difference | node_difference),
        nodal_shape_error=nodal_error,
        position_error_m=position_error,
        physical_position_error_m=physical_error,
        native_chart_admissible=admissible.reshape(macro_difference.shape),
        native_chart_cpu_local=admissible_local.reshape(native["location_local"].shape),
        native_chart_local_error=local_error,
    )
    return dict(
        macro_id_differing_samples=int(macro_difference.sum()),
        element_id_differing_samples=int(element_difference.sum()),
        node_id_differing_samples=int(node_difference.sum()),
        fine_cell_alias_samples=int((element_difference & ~macro_difference).sum()),
        strict_identity_passed=bool(not (macro_difference.any() or element_difference.any() or node_difference.any())),
        maximum_node_mapped_shape_error=_number(nodal_error.max()),
        maximum_position_error_m=_number(position_error.max()),
        independently_admissible_native_charts=bool(admissible.all()),
        acceptance="IDs are diagnostic and retained verbatim; physical/node-mapped/raw-force checks are separate explicit gates.",
    )


def _rule(api, output, checks, cap, order):
    np = api.np
    roots, weights = api.roots_jacobi(order, 0.5, 0.0)
    weights /= weights.sum()
    radial, angular, cap_weights = api.cap_rule(order)
    native = dict(
        nodes=api.qd_to_numpy(cap.rule.nodes),
        radial=api.qd_to_numpy(cap.rule.radial),
        weights=api.qd_to_numpy(cap.rule.weights),
    )
    output.arrays(
        f"q{order}/rule.npz",
        **native,
        scipy_nodes=roots,
        scipy_weights=weights,
        cpu_cap_radial=radial,
        cpu_cap_angular=angular,
        cpu_cap_weights=cap_weights,
    )
    checks.condition(f"q{order}/rule_valid", bool(api.qd_to_numpy(cap.rule.valid)))
    checks.close(f"q{order}/rule_nodes_vs_scipy", native["nodes"], roots, rtol=2e-13, atol=2e-14)
    checks.close(f"q{order}/rule_weights_vs_scipy", native["weights"], weights, rtol=1e-12, atol=5e-15)
    checks.close(
        f"q{order}/rule_radial_vs_cpu_cap", np.repeat(native["radial"], 4 * order), radial, rtol=2e-13, atol=2e-14
    )
    checks.close(
        f"q{order}/rule_weights_vs_cpu_cap",
        np.repeat(native["weights"] / (4 * order), 4 * order),
        cap_weights,
        rtol=1e-12,
        atol=5e-15,
    )


def _selectors(api, output, checks, prefix, selector, sources, cap, centers, selection, cases, cpu_sources, cpu):
    np = api.np
    rows = []
    for strategy in selection["strategies"]:
        state = selector.run(
            sources,
            cap.state,
            centers,
            strategy=strategy,
            support_angle=selection["support_angle"],
            narrow_fraction=selection["narrow_fraction"],
            maximum_radius=selection["maximum_radius"],
        )
        weights = api.qd_to_numpy(state.weights)
        expected_weights = np.stack(
            [
                api.selection_weights(
                    selection["centers"],
                    source.direction,
                    selection["support_angle"],
                    strategy,
                    selection["narrow_fraction"],
                )
                for source in cpu_sources
            ],
            axis=1,
        )
        grouped = api.qd_to_numpy(state.grouped_force)
        expected_grouped = np.zeros_like(grouped)
        cpu_blocks = cpu["raw"].reshape((-1, 3, len(cpu_sources))).transpose((0, 2, 1))
        for source, environment in enumerate(cases["source_environment"]):
            expected_grouped[:, :, environment] += (
                cpu_blocks[:, source, None, :] * expected_weights[:, source][None, :, None]
            )
        active = api.qd_to_numpy(state.active_groups)
        expected_active = np.zeros_like(active)
        for source, environment in enumerate(cases["source_environment"]):
            expected_active[:, environment] |= (expected_weights[:, source] > 0).astype(expected_active.dtype)
        output.arrays(
            f"{prefix}/{strategy}/selection.npz",
            weights=weights,
            cpu_weights=expected_weights,
            grouped_force=grouped,
            cpu_grouped_force=expected_grouped,
            active_groups=active,
            cpu_active_groups=expected_active,
            source_valid=api.qd_to_numpy(state.source_valid),
            environment_valid=api.qd_to_numpy(state.environment_valid),
            source_contributions=api.qd_to_numpy(state.source_contributions),
        )
        checks.close(f"{prefix}/{strategy}/weights", weights, expected_weights, rtol=1e-10, atol=1e-13)
        checks.close(f"{prefix}/{strategy}/grouped_raw_force", grouped, expected_grouped)
        checks.exact(f"{prefix}/{strategy}/active_groups", active, expected_active)
        checks.condition(f"{prefix}/{strategy}/source_valid", api.qd_to_numpy(state.source_valid).all())
        checks.condition(f"{prefix}/{strategy}/environment_valid", api.qd_to_numpy(state.environment_valid).all())
        group_count, contribution_count = (
            int(api.qd_to_numpy(state.group_count)),
            int(api.qd_to_numpy(state.contribution_count)),
        )
        checks.condition(
            f"{prefix}/{strategy}/counts",
            group_count == int(expected_active.sum()) and contribution_count == int((expected_weights > 0).sum()),
            groups=group_count,
            contributions=contribution_count,
        )
        released = np.diff(cases["case_offsets"]) == 0
        checks.condition(f"{prefix}/{strategy}/release_exact_zero", np.count_nonzero(grouped[:, :, released]) == 0)
        rows.append(dict(strategy=strategy, groups=group_count, contributions=contribution_count))
    return rows


def _inner_controls(api, published):
    np = api.np
    return dict(
        direction=published["direction"][[0, 4]].copy(),
        radius=np.array([0.001, 0.003]),
        force=np.array([[0.0, 0.0, 0.0], [0.07, -0.015, 0.023]]),
        has_force=np.array([False, True]),
        normal_force=np.array([0.1, 0.1]),
        friction=np.array([0.6, 0.3]),
        tangent=np.zeros((2, 3)),
        cap_tangent=np.zeros((2, 3)),
        has_tangent=np.zeros(2, dtype=bool),
        has_cap_tangent=np.zeros(2, dtype=bool),
        side=-np.ones(2, dtype=np.int64),
        source_environment=np.arange(2, dtype=np.int64),
        case_offsets=np.arange(3, dtype=np.int64),
        case_names=np.array(["inner-implicit-friction", "inner-explicit-force"]),
        source_names=np.array(["inner-implicit-friction", "inner-explicit-force"]),
    )


def run(api, args, result):
    np = api.np
    output, checks = Output(api, args.output), Checks(api)
    model = _load(np, args.reference / "inputs/model.npz")
    selection_input = _load(np, args.reference / "inputs/selection.npz")
    published = _load(np, args.reference / "cases/published43.npz")
    checks.condition(
        "published43/population",
        len(published["case_names"]) == 43
        and len(published["radius"]) == 46
        and int(np.count_nonzero(np.diff(published["case_offsets"]))) == 42,
    )
    checks.condition(
        "published43/fixture_identity",
        json.loads((args.reference / "cases/published43.json").read_text()) == api.cases_manifest(api.make_cases()),
    )
    # CPU geometry is reconstructed only as an independent oracle. Its derived
    # coefficients, chart locations, shape values and nodal loads are never uploaded.
    cpu_geometry = api.make_geometry(int(model["ncoarse"]), int(model["subdiv"]))
    checks.close("geometry/frozen_vertices", cpu_geometry.vertices, model["vertices"], rtol=1e-13, atol=1e-15)
    checks.close("geometry/frozen_directors", cpu_geometry.directors, model["directors"], rtol=1e-13, atol=1e-15)
    checks.exact("geometry/frozen_face_tables", cpu_geometry.face_tables, model["face_tables"])
    cpu_mother = SimpleNamespace(
        geometry=cpu_geometry, n_dof=6 * len(model["vertices"]), model=SimpleNamespace(h=float(model["thickness"]))
    )
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    geometry = api.build_material_cap_geometry(
        _vector(api, model["vertices"]),
        _integer(api, model["face_tables"]),
        _vector(api, model["directors"]),
        ncoarse=int(model["ncoarse"]),
        subdiv=int(model["subdiv"]),
    )
    coefficients = api.qd_to_numpy(geometry.macro_coefficients)
    output.arrays(
        "geometry.npz",
        native_macro_coefficients=coefficients,
        cpu_macro_coefficients=cpu_geometry.macro_coefficients,
        vertices=model["vertices"],
        directors=model["directors"],
        face_tables=model["face_tables"],
    )
    checks.close(
        "geometry/qd_macro_coefficients", coefficients, cpu_geometry.macro_coefficients, rtol=1e-13, atol=1e-15
    )
    centers = _vector(api, selection_input["centers"])
    selection = dict(
        centers=selection_input["centers"],
        support_angle=float(selection_input["support_angle"]),
        narrow_fraction=float(selection_input["narrow_fraction"]),
        maximum_radius=float(selection_input["maximum_radius"]),
        strategies=args.strategies,
    )
    banks = [("published43", published)]
    if args.side_controls:
        banks.append(("inner-side-controls", _inner_controls(api, published)))
    result["orders"] = []
    for order in args.orders:
        order_result = dict(order=order, banks=[])
        for bank_name, cases in banks:
            prefix = f"q{order}/{bank_name}"
            sources, cpu_sources = _source_inputs(api, cases), _cpu_sources(api, cases)
            cap = api.MaterialCap(
                order,
                len(cpu_sources),
                len(model["vertices"]),
                ncoarse=int(model["ncoarse"]),
                subdiv=int(model["subdiv"]),
                thickness=float(model["thickness"]),
                audit=True,
            )
            if bank_name == "published43":
                _rule(api, output, checks, cap, order)
            native = _native_loads(api, cap, geometry, sources, model, cases)
            cpu = _cpu_loads(api, cpu_mother, cpu_sources, order)
            output.arrays(f"{prefix}/native-loads.npz", **native)
            output.arrays(f"{prefix}/cpu-loads.npz", **cpu)
            checks.condition(f"{prefix}/all_sources_valid", native["valid"].all())
            checks.condition(f"{prefix}/all_samples_valid", native["sample_valid"].all())
            checks.close(f"{prefix}/raw_force", native["raw"], cpu["raw"], columns=True)
            checks.close(
                f"{prefix}/source_force",
                native["source_force"].T,
                cpu["source_force"].T,
                columns=True,
                rtol=2e-13,
                atol=2e-15,
            )
            checks.close(
                f"{prefix}/cap_directions", native["cap_directions"], cpu["cap_directions"], rtol=1e-12, atol=1e-13
            )
            for key in ("recovered_force", "recovered_moment", "expected_moment"):
                moment = key != "recovered_force"
                checks.close(
                    f"{prefix}/{key}",
                    native[key].T,
                    cpu[key].T,
                    columns=True,
                    rtol=2e-12 if moment else 2e-13,
                    atol=2e-16 if moment else 2e-15,
                )
            checks.close(
                f"{prefix}/physical_resultant_force",
                native["recovered_force"].T,
                native["source_force"].T,
                columns=True,
                rtol=2e-13,
                atol=2e-15,
            )
            checks.close(
                f"{prefix}/physical_resultant_moment",
                native["recovered_moment"].T,
                native["expected_moment"].T,
                columns=True,
                rtol=2e-12,
                atol=2e-16,
            )
            location = _location_checks(api, output, checks, prefix, native, cpu, cpu_geometry)
            if order == 8 and bank_name == "published43":
                frozen = _load(np, args.reference / "oracles/published43/loads.npz")
                for key in (
                    "raw",
                    "source_force",
                    "recovered_force",
                    "recovered_moment",
                    "expected_moment",
                    "cap_directions",
                    "location_positions",
                ):
                    checks.close(f"{prefix}/cpu_vs_frozen_q8/{key}", cpu[key], frozen[key], rtol=1e-12, atol=1e-14)
            selector = api.AtlasSelection(len(cpu_sources), len(cases["case_names"]), 2 * len(model["vertices"]))
            rows = _selectors(
                api, output, checks, prefix, selector, sources, cap, centers, selection, cases, cpu_sources, cpu
            )
            order_result["banks"].append(
                dict(
                    name=bank_name,
                    published=bank_name == "published43",
                    sources=len(cpu_sources),
                    cases=len(cases["case_names"]),
                    location=location,
                    selectors=rows,
                )
            )
            result.update(checks=checks.rows, passed=checks.passed(), artifacts=output.inventory)
            write_json(args.output / "checks.json", checks.rows)
            write_json(args.output / "result.json", result)
        result["orders"].append(order_result)
    api.qd.sync()
    source_files_after = [file_id(Path(row["path"])) for row in result["source_files"]]
    checks.condition("provenance/source_unchanged", source_files_after == result["source_files"])
    result.update(
        complete=True,
        passed=checks.passed(),
        physical_assembly_equivalence_passed=checks.passed(),
        strict_location_identity_passed=all(
            bank["location"]["strict_identity_passed"] for order in result["orders"] for bank in order["banks"]
        ),
        checks=checks.rows,
        artifacts=output.inventory,
        source_files_after=source_files_after,
    )
    write_json(args.output / "checks.json", checks.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", nargs="+", type=int, default=(8, 16, 24))
    parser.add_argument("--strategies", "--selectors", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument(
        "--side-controls",
        action="store_true",
        help="Add separately labelled inner-side implicit/explicit-force controls.",
    )
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid with GENESIS_IMAGE_VER=1_26.")
    if min(args.orders) < 1 or len(set(args.orders)) != len(args.orders):
        parser.error("Use distinct positive quadrature orders.")
    args.reference, args.output = args.reference.expanduser().resolve(), args.output.expanduser().resolve()
    if args.output.is_relative_to(REPOSITORY):
        parser.error("Keep runtime arrays outside the source checkout, under the project's data directory.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory; previous evidence is never overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(args.output / "cache/genesis"))
    result = dict(
        schema="genesis.material-cap-gpu-check.v1",
        complete=False,
        passed=False,
        identity=runtime_identity(),
        worker=file_id(Path(__file__)),
        reference=file_id(args.reference / "manifest.json"),
        scope="Untimed load/rule/selector consistency only; no ShellModel, stiffness, factors, stress solves or throughput claims.",
        native_input_policy="Only physical vertices/directors, integer face tables, physical sources and selector centers/parameters are uploaded. CPU-derived rules, coefficients, locations, shape functions and loads are oracle-only.",
        strict_identity_policy="Strict macro/element/node failures remain non-gating diagnostic rows; independent chart admissibility, physical positions, node-mapped shape and assembled loads all have explicit gates.",
        precision="64",
        orders_requested=list(args.orders),
        strategies=list(args.strategies),
        inner_side_controls=args.side_controls,
        source_files=[
            file_id(REPOSITORY / relative)
            for relative in (
                "genesis/engine/solvers/rigid/stress/material_cap.py",
                "genesis/engine/solvers/rigid/stress/atlas_selection.py",
                "research/rigid_stress/atlas_study/cpu_repro/geometry.py",
                "research/rigid_stress/atlas_study/cpu_repro/loads.py",
                "research/rigid_stress/atlas_study/cpu_repro/selection.py",
                "research/rigid_stress/atlas_study/cpu_repro/fixtures.py",
            )
        ],
        input_files=[
            file_id(args.reference / relative)
            for relative in (
                "inputs/model.npz",
                "inputs/selection.npz",
                "cases/published43.npz",
                "cases/published43.json",
            )
        ],
    )
    write_json(args.output / "result.json", result)
    started = time.perf_counter()
    try:
        api = _api()
        result["versions"] = dict(numpy=api.np.__version__, scipy=api.scipy.__version__)
        with api.threadpool_limits(limits=1):
            run(api, args, result)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds_including_jit_oracle_and_exports"] = time.perf_counter() - started
        write_json(args.output / "result.json", result)
    assert result["passed"], "Material-cap GPU checks failed; retain all diagnostics and arrays."


if __name__ == "__main__":
    main()
