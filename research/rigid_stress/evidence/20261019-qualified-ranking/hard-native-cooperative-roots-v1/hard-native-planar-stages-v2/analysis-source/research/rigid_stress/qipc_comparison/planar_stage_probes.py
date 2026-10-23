"""Read-only stage probes for the native planar contact geometry kernel."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress import hertz_regions
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.curvature import (
    StressCurvatureInfo,
    func_closest_surface,
    func_surface_curvature,
)
from genesis.engine.solvers.rigid.stress.hertz_regions import HertzRegionStatus, StressHertzRegionState
from genesis.engine.solvers.rigid.stress.normal_roots import func_normal_root
from genesis.engine.solvers.rigid.stress.pair_geometry import func_opponent_boundary
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.utils import array_class, geom
from genesis.utils.array_class import DataKind, V, V_MAT, V_VEC

from .check_native_hertz_recovery import array


@dataclass(frozen=True)
class PlanarQueryState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    is_valid: qd.Tensor
    geom: qd.Tensor
    seed: qd.Tensor
    normal: qd.Tensor
    center: qd.Tensor
    operator: qd.Tensor
    residual: qd.Tensor
    error: qd.Tensor
    counts: qd.Tensor


@qd.kernel
def kernel_probe_opposing_face(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressHertzRegionState,
    queries: PlanarQueryState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        queries.is_valid[i_c, i_b] = False
        if regions.status[i_c, i_b] == HertzRegionStatus.ITERATIONS:
            i_own = sources.geom[i_c, i_b]
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            i_other = collider_state.contact_data.geom_a[i_contact, i_b]
            if i_other == i_own:
                i_other = collider_state.contact_data.geom_b[i_contact, i_b]
            own_kind, other_kind = dyn_info.geoms.type[i_own], dyn_info.geoms.type[i_other]
            is_planar = (own_kind == gs.GEOM_TYPE.MESH or own_kind == gs.GEOM_TYPE.SPHERE) and (
                other_kind == gs.GEOM_TYPE.PLANE or other_kind == gs.GEOM_TYPE.BOX
            )
            if is_planar:
                seed = geom.qd_transform_by_trans_quat(
                    sources.witness[i_c, i_b], contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
                )
                local_normal = qd.Vector([dyn_info.geoms.data[i_other][i] for i in qd.static(range(3))]).normalized()
                if other_kind == gs.GEOM_TYPE.BOX:
                    position, i_face = seed, -1
                    if own_kind == gs.GEOM_TYPE.MESH:
                        position, i_face = func_closest_surface(
                            i_own, i_b, seed, dyn_state, dyn_info, curvature, epsilon
                        )
                    else:
                        position, boundary_normal, boundary_gap, i_face, boundary_status = func_opponent_boundary(
                            i_own, i_b, seed, dyn_state, dyn_info, epsilon
                        )
                    own_normal, own_tensor, status = func_surface_curvature(
                        i_own, i_face, i_b, position, dyn_state, dyn_info, curvature, epsilon
                    )
                    opposed = geom.qd_inv_transform_by_quat(-own_normal, dyn_state.geoms.quat[i_other, i_b])
                    i_axis = 0
                    for i in qd.static(range(1, 3)):
                        if qd.abs(opposed[i]) > qd.abs(opposed[i_axis]):
                            i_axis = i
                    local_normal = qd.Vector.zero(gs.qd_float, 3)
                    local_normal[i_axis] = qd.select(opposed[i_axis] >= 0.0, 1.0, -1.0)
                queries.normal[i_c, i_b] = -geom.qd_transform_by_quat(local_normal, dyn_state.geoms.quat[i_other, i_b])
                queries.geom[i_c, i_b], queries.seed[i_c, i_b], queries.is_valid[i_c, i_b] = i_own, seed, True
                qd.atomic_add(queries.counts[0], 1)
                if other_kind == gs.GEOM_TYPE.BOX:
                    qd.atomic_add(queries.counts[1], 1)


@qd.kernel
def kernel_probe_normal_root(
    dyn_state: array_class.DynState,
    queries: PlanarQueryState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(queries.is_valid.shape[0], queries.is_valid.shape[1]):
        if queries.is_valid[i_c, i_b]:
            center, operator, i_face, residual, status = func_normal_root(
                queries.geom[i_c, i_b],
                i_b,
                queries.normal[i_c, i_b],
                queries.seed[i_c, i_b],
                dyn_state,
                dyn_info,
                curvature,
                epsilon,
            )
            queries.center[i_c, i_b], queries.operator[i_c, i_b], queries.residual[i_c, i_b] = (
                center,
                operator,
                residual,
            )


@qd.kernel
def kernel_check_planar_probes(queries: PlanarQueryState, regions: StressHertzRegionState):
    for i_c, i_b in qd.ndrange(queries.is_valid.shape[0], queries.is_valid.shape[1]):
        if queries.is_valid[i_c, i_b]:
            normal, tangent = regions.normal[i_c, i_b], regions.tangent[i_c, i_b]
            frame = qd.Matrix.rows([tangent, normal.cross(tangent)])
            expected = frame @ queries.operator[i_c, i_b] @ frame.transpose()
            qd.atomic_max(queries.error[0], (queries.center[i_c, i_b] - regions.center[i_c, i_b]).norm())
            qd.atomic_max(queries.error[1], (queries.normal[i_c, i_b] - normal).norm())
            qd.atomic_max(
                queries.error[2], (expected - regions.curvature[i_c, i_b]).norm() / qd.max(expected.norm(), 1.0)
            )
            qd.atomic_max(queries.error[3], qd.abs(queries.residual[i_c, i_b] - regions.normal_residual[i_c, i_b]))


class PlanarStageProbe:
    def __init__(self, contacts):
        shape = contacts.valid.shape
        self.original = hertz_regions.kernel_hertz_planar
        self.state = PlanarQueryState(
            is_valid=V(dtype=gs.qd_bool, shape=shape),
            geom=V(dtype=gs.qd_int, shape=shape),
            seed=V_VEC(3, dtype=gs.qd_float, shape=shape),
            normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
            center=V_VEC(3, dtype=gs.qd_float, shape=shape),
            operator=V_MAT(3, 3, dtype=gs.qd_float, shape=shape),
            residual=V(dtype=gs.qd_float, shape=shape),
            error=V(dtype=gs.qd_float, shape=(4,)),
            counts=V(dtype=gs.qd_int, shape=(2,)),
        )
        self.state.error.fill(0)
        self.state.counts.fill(0)

    def __call__(self, dyn_state, collider, contacts, sources, regions, dyn_info, curvature, epsilon):
        kernel_probe_opposing_face(
            dyn_state, collider, contacts, sources, regions, self.state, dyn_info, curvature, epsilon
        )
        kernel_probe_normal_root(dyn_state, self.state, dyn_info, curvature, epsilon)
        self.original(dyn_state, collider, contacts, sources, regions, dyn_info, curvature, epsilon)
        kernel_check_planar_probes(self.state, regions)

    def report(self):
        errors, counts = array(self.state.error), array(self.state.counts)
        np.testing.assert_allclose(errors, 0, atol=1e-12, rtol=0)
        assert counts[0] > 0 and counts[1] > 0
        return {
            "passed": True,
            "maximum_center_normal_curvature_residual_differences": errors.tolist(),
            "planar_queries": int(counts[0]),
            "box_queries": int(counts[1]),
            "scope": "Read-only opposing-face/closest-surface and normal-root probes on actual pre-planar inputs. The original kernel remains the sole producer. Extra kernels and comparison checks are intrusive profiling work; their sum is not primary scene throughput.",
        }
