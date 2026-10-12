"""Recompute full source-prior cases through the complete graph before unchanged FP64 field checks."""
from pathlib import Path

import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, func_closest_triangle
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState
from genesis.options.rigid_stress import RigidStressOptions
from tests.rigid import test_stress_source_priors as prior_tests

from native_contact_graph_test_v1 import selected_scatter

original_prepare = prior_tests.kernel_prepare_cached_tractions
assets = Path(prior_tests.__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"


@qd.kernel
def kernel_original_sources(
    i_geom: int,
    contacts: StressContactState,
    tractions: StressTractionState,
    raw: StressContactState,
    sources: StressSourceState,
    surface: StressSurfaceInfo,
):
    for i_b in range(raw.valid.shape[1]):
        raw.frame_position[i_b] = contacts.frame_position[i_b]
        raw.frame_quaternion[i_b] = contacts.frame_quaternion[i_b]
    for i_c, i_b in qd.ndrange(raw.valid.shape[0], raw.valid.shape[1]):
        raw.valid[i_c, i_b] = contacts.valid[i_c, i_b]
        raw.position[i_c, i_b] = contacts.position[i_c, i_b]
        raw.force[i_c, i_b] = tractions.priors.force[i_c, i_b]
        raw.radius[i_c, i_b] = contacts.radius[i_c, i_b]
        raw.friction[i_c, i_b] = contacts.friction[i_c, i_b]
        sources.geom[i_c, i_b] = i_geom
        sources.pair[i_c, i_b] = 100 + i_c
        sources.witness[i_c, i_b] = contacts.center[i_c, i_b]
        sources.witness_face[i_c, i_b] = contacts.anchor_face[i_c, i_b]
        position, bary = func_closest_triangle(
            contacts.anchor_face[i_c, i_b], contacts.center[i_c, i_b], surface
        )
        sources.witness_barycentric[i_c, i_b] = bary
        sources.free_moment[i_c, i_b] = tractions.priors.anchor_moment[i_c, i_b] - (
            contacts.position[i_c, i_b] - contacts.center[i_c, i_b]
        ).cross(tractions.priors.force[i_c, i_b])


def prepare_graph(epsilon, contacts, tractions, state, model, surface, info, errno, reset_force=True):
    if gs.backend != gs.cuda:
        return original_prepare(
            epsilon, contacts, tractions, state, model, surface, info, errno, reset_force=reset_force
        )
    n_sources, n_envs = contacts.valid.shape
    options = RigidStressOptions(
        mesh=assets / "elastic.npz", method="direct", contact_model="intrinsic",
        intrinsic_load_partition="body_source_prior", contact_block_size=n_sources,
    )
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(gs.morphs.URDF(
        file=assets / "egg_shell.urdf", align=False, convexify=True, decimate=False, preserve_mesh_precision=True
    ))
    scene.build(n_envs=n_envs)
    raw = create_contacts(n_sources, n_envs, options.contact_radius)
    streamed = IntrinsicContacts(entity.base_link, raw, model, surface, options, None)
    kernel_original_sources(entity.base_link.geoms[0].idx, contacts, tractions, raw, streamed.raw_sources, surface)
    errno.fill(0)
    kernel_begin_step(state)
    selected_scatter(streamed, raw, state, model, surface, options, epsilon, errno)


@pytest.fixture(autouse=True)
def graph_contact_cases(monkeypatch):
    monkeypatch.setattr(IntrinsicContacts, "scatter", selected_scatter)
    monkeypatch.setattr(prior_tests, "kernel_prepare_cached_tractions", prepare_graph)
