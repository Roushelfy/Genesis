"""Check hinted geometry against the frozen exhaustive root and cache lifecycle."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.curvature import CurvatureStatus, StressCurvature, StressCurvatureInfo
from genesis.engine.solvers.rigid.stress.hertz_regions import (
    StressHertzRegions,
    StressHertzRegionState,
    StressHertzCache,
)
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, create_sources
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.utils import array_class, geom
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_hertz_regions import kernel_sources, array


@qd.func
def func_exhaustive_normal_root(
    i_g: int,
    i_b: int,
    direction: qd.types.vector(3),
    seed: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    """Choose the normal-field root nearest the source and return its curvature.

    A mesh root satisfies the normalized barycentric vertex-normal field exactly.
    Its position remains on the collision triangle. Multiple roots retain the
    source-local choice. Visibility and finite opponent support require separate
    checks before pressure admission.
    """
    rotation = geom.qd_quat_to_R(dyn_state.geoms.quat[i_g, i_b], epsilon)
    local_direction = rotation.transpose() @ direction
    local_seed = rotation.transpose() @ (seed - dyn_state.geoms.pos[i_g, i_b])
    center = qd.Vector.zero(gs.qd_float, 3)
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    status = gs.qd_int(CurvatureStatus.UNSUPPORTED)
    i_selected = -1
    weights = qd.Vector.zero(gs.qd_float, 3)
    residual = gs.qd_float(0.0)
    if dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.SPHERE:
        radius = dyn_info.geoms.data[i_g][0]
        center = radius * local_direction
        tensor = (qd.Matrix.identity(gs.qd_float, 3) - local_direction.outer_product(local_direction)) / radius
        status = CurvatureStatus.ANALYTIC
    elif dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.MESH:
        distance = gs.qd_float(float("inf"))
        for i_f in range(dyn_info.geoms.face_start[i_g], dyn_info.geoms.face_end[i_g]):
            if curvature_info.normal_inverse_valid[i_f]:
                barycentric = curvature_info.normal_inverse[i_f] @ local_direction
                total = barycentric.sum()
                if total > 128.0 * epsilon * barycentric.norm():
                    barycentric /= total
                    if barycentric.min() >= -128.0 * epsilon:
                        barycentric = qd.max(barycentric, 0.0)
                        barycentric /= barycentric.sum()
                        indices = dyn_info.faces.verts_idx[i_f]
                        candidate = qd.Vector.zero(gs.qd_float, 3)
                        for i in qd.static(range(3)):
                            i_v = indices[i]
                            candidate += barycentric[i] * dyn_info.verts.init_pos[i_v]
                        candidate_distance = (candidate - local_seed).norm_sqr()
                        if candidate_distance < distance:
                            distance, center, i_selected, weights = candidate_distance, candidate, i_f, barycentric
        if i_selected >= 0:
            indices = dyn_info.faces.verts_idx[i_selected]
            is_valid = True
            recovered_normal = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(3)):
                i_v = indices[i]
                tensor += weights[i] * curvature_info.vertices_tensor[i_v]
                recovered_normal += weights[i] * dyn_info.verts.init_normal[i_v]
                is_valid = is_valid and curvature_info.vertices_weight[i_v] > 0.0
            residual = (recovered_normal.normalized() - local_direction).norm()
            projection = qd.Matrix.identity(gs.qd_float, 3) - local_direction.outer_product(local_direction)
            tensor = projection @ tensor @ projection
            if is_valid:
                status = CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
    return (
        dyn_state.geoms.pos[i_g, i_b] + rotation @ center,
        rotation @ tensor @ rotation.transpose(),
        i_selected,
        residual,
        status,
    )


@qd.kernel
def kernel_cache_oracle(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    cache: StressHertzCache,
    errors: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(regions.status.shape[0], regions.status.shape[1]):
        errors[i_b, i_c, 0], errors[i_b, i_c, 1], errors[i_b, i_c, 2] = 0.0, 0.0, 0.0
        if regions.status[i_c, i_b] == 0:
            seed = geom.qd_transform_by_trans_quat(
                sources.witness[i_c, i_b], contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
            )
            center, operator, face, residual, status = func_exhaustive_normal_root(
                sources.geom[i_c, i_b], i_b, regions.normal[i_c, i_b], seed, dyn_state, dyn_info, curvature, epsilon
            )
            errors[i_b, i_c, 0] = (regions.center[i_c, i_b] - center).norm()
            errors[i_b, i_c, 1] = gs.qd_float(cache.face[i_c, i_b] != face)
            errors[i_b, i_c, 2] = qd.abs(regions.normal_residual[i_c, i_b] - residual)


@qd.kernel
def kernel_vertex_hints(vertices: qd.types.ndarray(), sources: StressSourceState):
    for i_c, i_b in qd.ndrange(vertices.shape[0], sources.geom.shape[1]):
        sources.vertex[i_c, i_b] = vertices[i_c]


@qd.kernel
def kernel_wrong_hints(hints: qd.types.ndarray(), cache: StressHertzCache):
    for i_c, i_b in qd.ndrange(hints.shape[1], hints.shape[0]):
        cache.face[i_c, i_b] = hints[i_b, i_c]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(), "backend": args.backend})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(morph=gs.morphs.Plane())
    left = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    left.vertices += [-0.035, 0, 0]
    right = left.copy()
    right.vertices += [0.07, 0, 0]
    mesh = trimesh.util.concatenate([left, right])
    path = args.output / "disconnected.obj"
    mesh.export(path)
    target = scene.add_entity(morph=gs.morphs.Mesh(file=path, convexify=False, decimate=False, pos=(0, 0, 0.02)))
    scene.build(n_envs=3)
    solver = scene.rigid_solver
    geometry = target.base_link.geoms[0]
    assert len(target.base_link.geoms) == 1
    vertices = tensor_to_array(target.get_verts())[0]
    selected = np.r_[
        np.flatnonzero(vertices[:, 0] < 0)[np.argsort(vertices[vertices[:, 0] < 0, 2])[:2]],
        np.flatnonzero(vertices[:, 0] > 0)[np.argsort(vertices[vertices[:, 0] > 0, 2])[:2]],
    ]
    points = vertices[selected].copy()
    vertex_ids = (selected + geometry.vert_start).astype(gs.np_int)
    owner = np.full(4, geometry.idx, dtype=gs.np_int)
    forces = np.broadcast_to(
        np.array([[0.01, 0.02, 0.1], [0.02, -0.01, 0.2], [-0.01, 0.01, 0.15], [0.02, 0.01, 0.25]]), (3, 4, 3)
    ).copy()
    contacts = create_contacts(8, 3, 1.0, cooperative=False, cooperative_scatter=False)
    sources = create_sources(8, 3)
    epsilon = float(np.finfo(gs.np_float).eps)
    curvature = StressCurvature(solver.dyn_info, epsilon)
    regions = StressHertzRegions(contacts)
    errors = np.zeros((3, 8, 3))
    reference = None
    rows = []
    baseline_faces = None
    for case in [
        "cold",
        "warm",
        "wrong_component_hint",
        "release",
        "reload",
        "partial_reset",
        "permuted",
        "invalid_hint",
        "bad_pair",
        "recovered",
        "discontinuous_source",
    ]:
        indices = np.array([0, 1, 2, 3])
        if case == "permuted":
            indices = np.array([2, 3, 0, 1])
        if case == "discontinuous_source":
            indices = np.array([2, 3, 2, 3])
        current_force = forces[:, indices].copy()
        if case == "release":
            current_force[1] = 0
        if case == "partial_reset":
            before = array(regions.cache.face)
            regions.reset(np.array([1], dtype=gs.np_int))
            after = array(regions.cache.face)
            np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
            np.testing.assert_array_equal(after[1], -1)
        if case in ("wrong_component_hint", "invalid_hint"):
            hints = baseline_faces[:, [2, 3, 0, 1]].copy()
            if case == "invalid_hint":
                hints.fill(geometry.face_end + 100)
            kernel_wrong_hints(hints, regions.cache)
        kernel_sources(
            owner[indices],
            points[indices],
            current_force,
            np.zeros(3),
            np.array([1.0, 0, 0, 0]),
            contacts,
            sources,
            solver.collider.collider_state,
            target.base_link.idx,
            plane.base_link.idx,
            plane.base_link.geoms[0].idx,
            case == "bad_pair",
            False,
            False,
            False,
        )
        kernel_vertex_hints(vertex_ids[indices], sources)
        regions.update(
            target.base_link.idx,
            contacts,
            sources,
            solver.dyn_state,
            solver.collider.collider_state,
            solver.dyn_info,
            curvature.info,
            epsilon,
        )
        kernel_cache_oracle(
            contacts,
            sources,
            regions.state,
            regions.cache,
            errors,
            solver.dyn_state,
            solver.dyn_info,
            curvature.info,
            epsilon,
        )
        np.testing.assert_allclose(errors[..., [0, 2]], 0, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(errors[..., 1], 0)
        face, hit, kinds = array(regions.cache.face), array(regions.cache.hit), array(regions.cache.hint_kind)
        state = {
            key: array(value)
            for key, value in [
                ("center", regions.state.center),
                ("force", regions.state.force),
                ("moment", regions.state.moment),
                ("source_count", regions.state.source_count),
                ("valid", regions.state.is_valid),
            ]
        }
        if case == "cold":
            reference, baseline_faces = state, face[:, :4].copy()
            assert np.all(face[:, :4] >= 0)
            assert not hit.any()
            assert (kinds[:, :4] == 2).all()
        elif case == "release":
            np.testing.assert_array_equal(face[1], -1)
            np.testing.assert_array_equal(state["force"][1], 0)
            np.testing.assert_array_equal(face[[0, 2], :4], baseline_faces[[0, 2]])
            np.testing.assert_array_equal(face[[0, 2], 4:], -1)
        elif case == "bad_pair":
            assert not state["valid"][1] and face[1, 0] == -1
            np.testing.assert_array_equal(state["force"][1], 0)
        elif case not in ("permuted", "discontinuous_source"):
            for key in reference:
                np.testing.assert_allclose(state[key], reference[key], rtol=0, atol=1e-12)
        if case == "warm":
            assert hit[:, :4].all()
        if case == "partial_reset":
            assert hit[[0, 2], :4].all() and not hit[1, :4].any()
        if case == "discontinuous_source":
            assert (np.count_nonzero(state["source_count"], axis=1) == 1).all()
            np.testing.assert_allclose(state["force"].sum(axis=1), current_force.sum(axis=1), atol=1e-14, rtol=0)
        rows.append(
            {
                "case": case,
                "oracle_maximum_errors": errors.max(axis=(0, 1)).tolist(),
                "hint_kind_counts": np.bincount(kinds[:, :4].ravel(), minlength=4).tolist(),
                "cache_hits": int(hit.sum()),
                "query_evaluations": array(regions.cache.evaluations).sum(axis=(0, 1)).tolist(),
            }
        )
        np.savez_compressed(args.output / f"{case}.npz", **state, cache_face=face, cache_hit=hit, errors=errors)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "cases": rows,
            "mesh": file_id(path),
            "scope": "Synthetic current-plane contacts on one disconnected mesh. Exact root-face and bounded position/residual agreement with the frozen exhaustive normal-root implementation, source vertex hints, cold/warm/wrong-component/invalid-hint fallback, release, pair rejection, source reordering, partial reset and discontinuous source changes. No friction-history or physical pressure claim.",
        },
    )


if __name__ == "__main__":
    main()
