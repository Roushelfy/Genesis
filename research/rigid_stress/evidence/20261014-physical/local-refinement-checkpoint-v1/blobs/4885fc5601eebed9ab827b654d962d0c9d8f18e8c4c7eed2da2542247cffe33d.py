"""Nested local tetrahedron bisection around measured reference contact points."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .franka_contact_reference import RigidFingerContactSystem
from .refine_franka_reference import refine_body
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def contact_centres(system, run):
    run = Path(run)
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((run / "results/configuration.json").read_text())
    assert config["body_refinement"] == 0 and config["cold_closure"]
    with np.load(run / "results/reference_geometry.npz") as saved:
        np.testing.assert_array_equal(saved["vertices"], system.body.mesh.p.T)
        np.testing.assert_array_equal(saved["tetrahedra"], system.body.mesh.t.T)
    with np.load(run / f"results/step{config['steps']:04d}.npz") as saved:
        force = saved["contact_force_N"].reshape(-1, 3)[: len(system.body.coordinates)]
    active = np.flatnonzero(np.linalg.norm(force, axis=1) > 1e-8)
    assert len(active) > 0
    return system.body.coordinates[active].copy()


def localized_mesh(mesh, centres, radius, levels, max_cells=60000):
    assert 0 < radius < 0.1 and 0 <= levels <= 24
    centres = np.asarray(centres)
    history = []
    for level in range(levels):
        vertices = mesh.p[:, mesh.t].transpose(2, 1, 0)
        lo, hi = vertices.min(axis=1), vertices.max(axis=1)
        lower_bound = np.maximum(np.maximum(lo[:, None] - centres, centres - hi[:, None]), 0)
        marked = np.flatnonzero(np.any(np.linalg.norm(lower_bound, axis=2) <= radius, axis=1))
        assert len(marked) > 0
        mesh = mesh.refined(marked)
        assert mesh.nelements <= max_cells, (level + 1, mesh.nelements, max_cells)
        history.append({"level": level + 1, "marked_cells": len(marked), "tetrahedra": int(mesh.nelements)})
    return mesh, history


def refine_local_body(system, levels, run, radius):
    seed = refine_body(system, 0)
    centres = contact_centres(seed, run)
    mesh, history = localized_mesh(seed.body.mesh, centres, radius, levels)
    metadata = {
        "seed_contact_run": str(run),
        "reference_centres_m": centres.tolist(),
        "fixed_radius_m": radius,
        "marking": "Cell AABB lower distance to any fixed contact centre is at most radius.",
        "refinement": "scikit-fem conforming marked-tetrahedron bisection; fixed seed polyhedron",
        "history": history,
    }
    if not levels:
        seed.local_refinement_metadata = metadata
        return seed
    original = seed.body
    body = SkfemElasticReference(mesh.p.T, mesh.t.T, original.young, original.poisson, original.density)
    np.testing.assert_allclose(body.total_mass, original.total_mass, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(body.com, original.com, rtol=0, atol=1e-13)
    fixed = seed.fixed_initializer_points_m
    axes = seed.fixed_initializer_components
    distance, nodes = cKDTree(body.coordinates).query(fixed)
    assert distance.max() < 1e-12
    body.gauge_dofs = body.component_dofs[nodes, axes]
    assert len(np.unique(body.gauge_dofs)) == 6
    body.free_dofs = np.setdiff1d(np.arange(body.basis.N), body.gauge_dofs)
    finger_vertices, finger_faces, finger_centres = [], [], []
    for index, indices in enumerate(seed.finger_nodes):
        local = indices[:-1]
        selected = seed.faces[seed.groups[seed.faces[:, 0]] == index + 1]
        assert np.all(np.isin(selected, local))
        finger_vertices.append(seed.rest[local])
        finger_faces.append(selected - local[0])
        finger_centres.append(seed.rest[indices[-1]])
    result = RigidFingerContactSystem(
        body, finger_vertices, finger_faces, seed.axes, seed.masses, finger_centres, seed.dhat, seed.friction
    )
    result.world_coordinate_origin_m = seed.world_coordinate_origin_m.copy()
    result.fixed_initializer_points_m = fixed.copy()
    result.fixed_initializer_components = axes.copy()
    result.local_refinement_metadata = metadata
    return result
