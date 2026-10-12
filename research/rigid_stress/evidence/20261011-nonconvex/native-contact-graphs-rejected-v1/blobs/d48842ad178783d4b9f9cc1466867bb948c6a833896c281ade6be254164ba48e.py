"""Prototype a native Quadrants graph of the unchanged complete aggregate contact pipeline."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import quadrants as qd

import genesis as gs
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import (
    compatibility,
    cooperative_fit,
    cooperative_scatter,
    fan_refinement,
    footprint,
    intrinsic,
    provenance,
    source_priors,
    source_vertices,
    surface_frontier,
    traction,
)
from genesis.engine.solvers.rigid.stress.compatibility import StressWrenchSearchState
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState
from genesis.engine.solvers.rigid.stress.intrinsic import StressContactPacking, StressWrenchCertificates
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.source_vertices import StressSourceVertexInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionInfo, StressTractionState
from genesis.utils.misc import qd_to_numpy

# Reuse the exact original function bodies rather than copy numerical kernels.
func_load = qd.func(intrinsic.kernel_load_intrinsic_contacts.fn)
func_bind_vertices = qd.func(source_vertices.kernel_bind_source_vertices.fn)
func_ports = qd.func(footprint.kernel_footprint_ports.fn)
func_frontier = qd.func(surface_frontier.kernel_fan_frontier.fn)
func_samples = qd.func(surface_frontier.kernel_fan_frontier_samples.fn)
func_support = qd.func(intrinsic.kernel_bind_traction_support_impl.fn)
func_snapshot = qd.func(source_priors.kernel_snapshot_source_priors.fn)
func_partition = qd.func(source_priors.kernel_partition_source_priors.fn)
func_fit = qd.func(cooperative_fit.kernel_fit_tractions.fn)
func_retry = qd.func(traction.kernel_fit_tractions_impl.fn)
func_disks = qd.func(footprint.kernel_refine_failed_disks.fn)
func_refresh = qd.func(source_priors.kernel_refresh_refined_source_priors.fn)
func_select_fans = qd.func(fan_refinement.kernel_select_failed_fans.fn)
func_refine_fans = qd.func(fan_refinement.kernel_refine_failed_fans.fn)
func_force_certificate = qd.func(compatibility.kernel_certify_continuous_force.fn)
func_search = qd.func(compatibility.kernel_search_wrench_certificate_impl.fn)
func_wrench_certificate = qd.func(compatibility.kernel_certify_continuous_wrench_impl.fn)
func_scatter = qd.func(cooperative_scatter.kernel_scatter_cached_tractions.fn)
func_commit = qd.func(cooperative_scatter.kernel_commit_intrinsic_contacts.fn)
func_body_commit = qd.func(intrinsic.kernel_commit_body_partition.fn)


@qd.kernel(graph=True, fastcache=True)
def kernel_contact_graph(
    epsilon: float,
    projection_limit: float,
    fit_tolerance: float,
    warm_start: qd.template(),
    raw_contacts: StressContactState,
    raw_sources: StressSourceState,
    contacts: StressContactState,
    sources: StressSourceState,
    footprints: StressFootprintState,
    tractions: StressTractionState,
    state: StressState,
    model: StressInfo,
    surface: StressSurfaceInfo,
    paths: StressPathInfo,
    info: StressTractionInfo,
    vertices: StressSourceVertexInfo,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
    search: StressWrenchSearchState,
    errno: qd.Tensor,
):
    func_load(i_start=0, raw_contacts=raw_contacts, raw_sources=raw_sources, contact_state=contacts,
              source_state=sources, packing=packing)
    func_bind_vertices(contacts, sources, vertices, epsilon)
    provenance.func_resolve_sources(projection_limit, epsilon, contacts, sources, model, surface)
    func_ports(contacts, sources, footprints, surface, paths)
    func_frontier(contacts, footprints, tractions, model, surface, paths, info, epsilon, tractions.frontier)
    func_samples(contacts, footprints, tractions, model, surface, paths, info, epsilon,
                 tractions.frontier, tractions.frontier.maximum[None])
    func_support(contacts, sources, footprints, tractions, body_union=True)
    func_snapshot(contacts, sources, tractions)
    func_partition(contacts, sources, tractions)
    func_fit(contacts, sources, tractions, info, epsilon, fit_tolerance, max_evaluations=128, warm_start=warm_start)
    func_retry(contacts, sources, tractions, info, epsilon, fit_tolerance, max_evaluations=128, retry_mode=1)
    func_disks(contacts, sources, footprints, tractions, surface, info, epsilon)
    func_refresh(contacts, tractions)
    func_retry(contacts, sources, tractions, info, epsilon, fit_tolerance, max_evaluations=128, retry_mode=2)
    func_select_fans(contacts, footprints, tractions)
    func_refine_fans(contacts, sources, footprints, tractions, model, surface, paths, info, epsilon)
    func_refresh(contacts, tractions)
    func_retry(contacts, sources, tractions, info, epsilon, fit_tolerance, max_evaluations=128, retry_mode=3)
    func_retry(contacts, sources, tractions, info, epsilon, fit_tolerance, max_evaluations=128, retry_mode=4)
    func_force_certificate(contacts, tractions, surface, epsilon, force_tolerance=1e-8)
    func_search(contacts, sources, tractions, search, info, epsilon, wrench_tolerance=1e-8, boundary_witnesses=False)
    func_wrench_certificate(contacts, sources, tractions, search, surface, epsilon,
                           maximum_depth=8, maximum_nodes=16384, wrench_tolerance=1e-8, accumulate=False)
    func_search(contacts, sources, tractions, search, info, epsilon, wrench_tolerance=1e-8, boundary_witnesses=True)
    func_wrench_certificate(contacts, sources, tractions, search, surface, epsilon,
                           maximum_depth=12, maximum_nodes=65536, wrench_tolerance=1e-8, accumulate=True)
    func_scatter(epsilon, contacts, tractions, state, model, surface, info, errno, reset_force=False)
    func_commit(i_start=0, epsilon=epsilon, raw_contacts=raw_contacts, raw_sources=raw_sources,
                contact_state=contacts, source_state=sources, footprint_state=footprints,
                traction_state=tractions, traction_info=info, packing=packing, certificates=certificates)
    func_body_commit(raw_contacts, packing)


mode = sys.argv.pop(1)
assert mode in ("off", "on")
original_scatter = intrinsic.IntrinsicContacts.scatter


def selected_scatter(contacts, raw_contacts, state, stress_info, surface_info, options, epsilon, errno):
    if (
        mode == "off" or gs.backend != gs.cuda or options.intrinsic_load_partition != "body_source_prior"
        or options.intrinsic_integration != "surface_fan" or not options.cooperative_tractions
        or not options.surface_fan_frontier or contacts.vertices.info is None
    ):
        return original_scatter(contacts, raw_contacts, state, stress_info, surface_info, options, epsilon, errno)
    intrinsic.kernel_pack_intrinsic_contacts(raw_contacts, contacts.packing, contacts.certificates, state)
    n_work = int(qd_to_numpy(contacts.packing.maximum, transpose=True))
    if n_work > contacts.contacts.valid.shape[0]:
        contacts._allocate_workspace(surface_info, options, n_work, raw_contacts.valid.shape[1])
    if n_work > 0:
        kernel_contact_graph(
            epsilon, options.source_projection_limit, intrinsic._TRACTION_FIT_TOLERANCE,
            options.warm_start_tractions, raw_contacts, contacts.raw_sources, contacts.contacts, contacts.sources,
            contacts.footprints, contacts.tractions, state, stress_info, surface_info, contacts.paths.info,
            contacts.traction_info, contacts.vertices.info, contacts.packing, contacts.certificates,
            contacts.wrench_search, errno,
        )
    intrinsic.kernel_capture_contact_failure(
        raw_contacts, contacts.raw_sources, contacts.failure, errno, grouped=True
    )


with patch.object(intrinsic.IntrinsicContacts, "scatter", selected_scatter):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["native_contact_graph"] = mode == "on"
result["native_contact_graph_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
