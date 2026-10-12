"""Ablate Quadrants fastcache and per-kernel graphs on the unchanged native contact bodies."""
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import quadrants as qd

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

mode = sys.argv.pop(1)
assert mode in ("off", "fast", "graph", "both")
bindings = (
    (intrinsic.kernel_pack_intrinsic_contacts, ((intrinsic, "kernel_pack_intrinsic_contacts"),)),
    (intrinsic.kernel_load_intrinsic_contacts, ((intrinsic, "kernel_load_intrinsic_contacts"),)),
    (source_vertices.kernel_bind_source_vertices, ((source_vertices, "kernel_bind_source_vertices"), (intrinsic, "kernel_bind_source_vertices"))),
    (provenance.kernel_resolve_sources, ((provenance, "kernel_resolve_sources"), (intrinsic, "kernel_resolve_sources"))),
    (footprint.kernel_footprint_ports, ((footprint, "kernel_footprint_ports"), (intrinsic, "kernel_footprint_ports"))),
    (surface_frontier.kernel_fan_frontier, ((surface_frontier, "kernel_fan_frontier"),)),
    (surface_frontier.kernel_fan_frontier_samples, ((surface_frontier, "kernel_fan_frontier_samples"),)),
    (intrinsic.kernel_bind_traction_support_impl, ((intrinsic, "kernel_bind_traction_support_impl"),)),
    (source_priors.kernel_snapshot_source_priors, ((source_priors, "kernel_snapshot_source_priors"), (intrinsic, "kernel_snapshot_source_priors"))),
    (source_priors.kernel_partition_source_priors, ((source_priors, "kernel_partition_source_priors"), (intrinsic, "kernel_partition_source_priors"))),
    (cooperative_fit.kernel_fit_tractions, ((cooperative_fit, "kernel_fit_tractions"),)),
    (traction.kernel_fit_tractions_impl, ((traction, "kernel_fit_tractions_impl"),)),
    (footprint.kernel_refine_failed_disks, ((footprint, "kernel_refine_failed_disks"), (intrinsic, "kernel_refine_failed_disks"))),
    (source_priors.kernel_refresh_refined_source_priors, ((source_priors, "kernel_refresh_refined_source_priors"), (intrinsic, "kernel_refresh_refined_source_priors"))),
    (fan_refinement.kernel_select_failed_fans, ((fan_refinement, "kernel_select_failed_fans"), (intrinsic, "kernel_select_failed_fans"))),
    (fan_refinement.kernel_refine_failed_fans, ((fan_refinement, "kernel_refine_failed_fans"), (intrinsic, "kernel_refine_failed_fans"))),
    (compatibility.kernel_certify_continuous_force, ((compatibility, "kernel_certify_continuous_force"), (intrinsic, "kernel_certify_continuous_force"))),
    (compatibility.kernel_search_wrench_certificate_impl, ((compatibility, "kernel_search_wrench_certificate_impl"),)),
    (compatibility.kernel_certify_continuous_wrench_impl, ((compatibility, "kernel_certify_continuous_wrench_impl"),)),
    (cooperative_scatter.kernel_scatter_cached_tractions, ((cooperative_scatter, "kernel_scatter_cached_tractions"),)),
    (cooperative_scatter.kernel_commit_intrinsic_contacts, ((cooperative_scatter, "kernel_commit_intrinsic_contacts"),)),
    (intrinsic.kernel_commit_body_partition, ((intrinsic, "kernel_commit_body_partition"),)),
)
with ExitStack() as stack:
    for original, destinations in bindings:
        selected = original if mode == "off" else qd.kernel(
            original.fn, graph=mode in ("graph", "both"), fastcache=mode in ("fast", "both")
        )
        for module, name in destinations:
            stack.enter_context(patch.object(module, name, selected))
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["native_contact_dispatch"] = mode
result["native_contact_dispatch_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
