"""QIPC coupler tape-import tests: prestress (rest_geometry), Cloth membrane
options, and wound-roll import (design: docs/adhesion_tape_design.md, A5.2).

The wound-roll tests default to the in-tree asset genesis/assets/qipc/
tape_roll_distance_bond.npz; override with QIPC_TAPE_ASSET (cgq
adhesive_tape_wind --save).
"""

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs
from genesis.utils.misc import get_assets_dir


def _tape_module():
    # Deferred: importing the coupler package compiles quadrants kernels whose
    # type annotations need gs.init() (provided per-test by the autouse fixture).
    from genesis.engine.couplers.qipc_coupler import tape

    return tape


TAPE_ASSET_PATH = os.environ.get("QIPC_TAPE_ASSET", "") or os.path.join(
    get_assets_dir(), "qipc", "tape_roll_distance_bond.npz"
)

needs_tape_asset = pytest.mark.skipif(
    not os.path.exists(TAPE_ASSET_PATH),
    reason=("wound-roll npz not found (in-tree genesis/assets/qipc/tape_roll_distance_bond.npz or QIPC_TAPE_ASSET)"),
)


@needs_tape_asset
@pytest.mark.parametrize(
    ("collar", "expected_count"),
    [(0, 1019), (1, 1007), (2, 995), (3, 983)],
)
def test_bond_cluster_member_triangles_follow_authored_front(collar, expected_count):
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)

    members = tape_mod.bond_cluster_member_triangles(asset, collar)

    assert members.dtype == np.int32
    assert members.shape == (expected_count,)
    assert np.array_equal(members, np.unique(members))
    assert int(members.min()) == 0
    assert int(members.max()) < len(asset.tape_tris)


@needs_tape_asset
def test_bond_cluster_member_triangles_require_authored_bonds():
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)

    with pytest.raises(Exception, match="wind-authored distance bonds"):
        tape_mod.bond_cluster_member_triangles(replace(asset, bond_topos=None), 3)
    with pytest.raises(Exception, match="non-negative integer"):
        tape_mod.bond_cluster_member_triangles(asset, -1)


@needs_tape_asset
def test_tape_bond_cluster_requires_positive_relock_floor():
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)
    scene = SimpleNamespace(
        sim=SimpleNamespace(
            coupler=SimpleNamespace(
                _options=SimpleNamespace(adhesion_bond_lock_floor_ratio=0.0),
            )
        )
    )

    with pytest.raises(Exception, match="adhesion_bond_lock_floor_ratio > 0"):
        tape_mod.add_tape_bond_cluster(
            scene,
            object(),
            object(),
            asset,
            collar=3,
            detach_displacement=5.0 * asset.d_hat,
        )


# ---------------------------------------------------------------------------
# Prestress: bent-arc strip with a flat rest springs open
# ---------------------------------------------------------------------------

ARC_L = 0.2
ARC_W = 0.03
ARC_NX = 20
ARC_NZ = 2


def _arc_strip_verts(flat: bool) -> np.ndarray:
    """Strip of length ARC_L along its arc, width along z.

    flat=True: straight strip in the y-z plane. flat=False: isometrically bent
    into a half circle of radius L/pi around the z axis (pure bending, zero
    membrane strain).
    """
    r = ARC_L / np.pi
    verts = np.empty(((ARC_NX + 1) * (ARC_NZ + 1), 3), dtype=np.float64)
    for i in range(ARC_NX + 1):
        s = ARC_L * i / ARC_NX
        for j in range(ARC_NZ + 1):
            z = ARC_W * j / ARC_NZ - ARC_W / 2
            if flat:
                verts[i * (ARC_NZ + 1) + j] = (0.0, s, z)
            else:
                theta = s / r
                verts[i * (ARC_NZ + 1) + j] = (r * (1.0 - np.cos(theta)), r * np.sin(theta), z)
    return verts


def _strip_tris() -> np.ndarray:
    tris = []
    for i in range(ARC_NX):
        for j in range(ARC_NZ):
            v0 = i * (ARC_NZ + 1) + j
            v1 = (i + 1) * (ARC_NZ + 1) + j
            v2 = (i + 1) * (ARC_NZ + 1) + j + 1
            v3 = i * (ARC_NZ + 1) + j + 1
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])
    return np.array(tris, dtype=np.int32)


def _write_obj(path, verts, faces):
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def _build_arc_scene(tmp_path, show_viewer, with_flat_rest: bool):
    obj_path = str(tmp_path / "arc.obj")
    _write_obj(obj_path, _arc_strip_verts(flat=False), _strip_tris())

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    strip = scene.add_entity(
        morph=gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.3)),
        material=gs.materials.FEM.Cloth(
            E=1e6,
            nu=0.3,
            rho=200.0,
            thickness=5e-4,
            bending_stiffness=1e6,
            membrane="stvk",
            bending_model="hinge",
        ),
    )
    if with_flat_rest:
        # Map the flat rest through the entity's actual vertex order (the mesh
        # loader may permute vertices).
        from genesis.utils.misc import tensor_to_array

        tape_mod = _tape_module()
        actual = tensor_to_array(strip.init_positions).astype(np.float64)
        tape_mod._verify_same_vertex_order(_arc_strip_verts(flat=False), actual)
        scene.sim.coupler.set_fem_rest_positions(strip, _arc_strip_verts(flat=True))
    scene.build()
    return scene, strip


def _chord_length(strip) -> float:
    # Permutation-invariant opening measure: bbox diagonal.
    pos = strip.get_state().pos[0].cpu().numpy()
    return float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))


@pytest.mark.required
def test_prestress_arc_springs_open(tmp_path, show_viewer):
    """A bent strip with a FLAT rest carries stored bending energy and opens."""
    scene, strip = _build_arc_scene(tmp_path, show_viewer, with_flat_rest=True)
    chord0 = _chord_length(strip)
    for _ in range(100):
        scene.step()
    chord = _chord_length(strip)
    assert np.isfinite(strip.get_state().pos[0].cpu().numpy()).all()
    # half-circle chord = 2r = 2L/pi ~ 0.127; the flattening strip approaches L
    assert chord > chord0 + 0.03, f"prestressed arc did not spring open (chord {chord0:.3f} -> {chord:.3f})"


@pytest.mark.required
def test_no_prestress_arc_stays_bent(tmp_path, show_viewer):
    """Without a rest override the bent shape IS the rest shape: no springback."""
    scene, strip = _build_arc_scene(tmp_path, show_viewer, with_flat_rest=False)
    chord0 = _chord_length(strip)
    for _ in range(100):
        scene.step()
    chord = _chord_length(strip)
    assert abs(chord - chord0) < 0.01, f"unstressed arc moved (chord {chord0:.3f} -> {chord:.3f})"


def test_rest_positions_validation(tmp_path, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.FEM.Elastic(E=1e5, nu=0.3, rho=1000.0),
    )
    with pytest.raises(Exception, match="shape"):
        scene.sim.coupler.set_fem_rest_positions(cube, np.zeros((4, 2)))
    # wrong vertex count is caught at build
    scene.sim.coupler.set_fem_rest_positions(cube, np.zeros((3, 3)))
    with pytest.raises(Exception, match="does not match"):
        scene.build()


# ---------------------------------------------------------------------------
# Tape asset import
# ---------------------------------------------------------------------------


@needs_tape_asset
def test_tape_asset_parses():
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)
    n_verts = (asset.nx + 1) * (asset.nz + 1)
    assert asset.tape_positions.shape == (n_verts, 3)
    assert asset.tape_tris.shape == (2 * asset.nx * asset.nz, 3)
    rest = asset.flat_rest_positions()
    assert rest.shape == (n_verts, 3)
    # rest strip has the exact physical dimensions
    assert np.isclose(rest[:, 1].max() - rest[:, 1].min(), asset.tape_length)
    assert np.isclose(rest[:, 2].max() - rest[:, 2].min(), asset.width)
    # the wound coil is wound: much shorter in extent than the strip length
    extent = asset.tape_positions.max(axis=0) - asset.tape_positions.min(axis=0)
    assert extent.max() < 0.8 * asset.tape_length
    opts = tape_mod.recommended_coupler_options(asset)
    assert opts["contact_d_hat"] == asset.d_hat
    # Measured tape solver profile, NOT the wind's SOLVER_CFG: velocity_tol
    # 0.01 (qipc's 0.05 leaves released rolls hovering) and cgq's iteration
    # caps, but never its line_search/max_iter=16 (stalls imported rolls).
    assert opts["solver_newton_velocity_tol"] == 3.8e-3
    assert opts["solver_newton_max_iter"] == 300
    assert opts["solver_linear_max_iter"] == 800
    assert opts["solver_linear_tol_rate"] == 3e-3
    assert "solver_line_search_max_iter" not in opts
    # ... but the opt-in translation helper maps SOLVER_CFG onto option fields
    solver_cfg = asset.params.get("SOLVER_CFG") or {}
    translated = tape_mod.solver_cfg_to_options(solver_cfg)
    if "newton/max_iter" in solver_cfg:
        assert translated["solver_newton_max_iter"] == int(solver_cfg["newton/max_iter"])
    if "line_search/max_iter" in solver_cfg:
        assert translated["solver_line_search_max_iter"] == int(solver_cfg["line_search/max_iter"])
    # and QIPCCouplerOptions accepts both dicts
    gs.options.QIPCCouplerOptions(**opts)
    gs.options.QIPCCouplerOptions(**{**opts, **translated})


ROLL_POS = np.array([0.0, 0.0, 0.2])


def _coil_max_radius(tape) -> float:
    """Max vertex distance from the roll axis (world y through ROLL_POS)."""
    pos = tape.get_state().pos[0].cpu().numpy()
    return float(np.sqrt((pos[:, 0] - ROLL_POS[0]) ** 2 + (pos[:, 2] - ROLL_POS[2]) ** 2).max())


def _build_roll_scene(
    show_viewer,
    *,
    sticky: bool,
    hub_fixed: bool = True,
    prepend_rigid: bool = False,
):
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)
    opts = tape_mod.recommended_coupler_options(asset)
    if not sticky:
        opts.update(
            adhesion_bond_distance_lock=False,
            adhesion_bond_max_bonds=0,
        )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(**opts),
        show_viewer=show_viewer,
    )
    if prepend_rigid:
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.5, 0.5, 0.5),
                size=(0.02, 0.02, 0.02),
                fixed=True,
            ),
            material=gs.materials.Rigid(rho=1000.0),
        )
    adhesion_off = dict(Cn=0.0, Ct=0.0, bonding_rate=0.0, beta0=0.0, enabled=False, distance_lock=False)
    tape, hub = tape_mod.add_tape_roll(
        scene,
        asset,
        pos=tuple(ROLL_POS),
        with_hub=True,
        hub_fixed=hub_fixed,
        tape_tape_adhesion=None if sticky else adhesion_off,
        tape_hub_adhesion=None if sticky else adhesion_off,
    )
    scene.build()
    return scene, tape, hub, asset


def _build_cluster_roll_scene(show_viewer):
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)
    coupler_options = tape_mod.recommended_coupler_options(asset)
    coupler_options["adhesion_bond_lock_floor_ratio"] = 0.5
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(**coupler_options),
        show_viewer=show_viewer,
    )
    tape, hub = tape_mod.add_tape_roll(
        scene,
        asset,
        pos=tuple(ROLL_POS),
        with_hub=True,
        hub_fixed=False,
        hub_qipc_abd_kappa=5e7,
    )
    controller = tape_mod.add_tape_bond_cluster(
        scene,
        tape,
        hub,
        asset,
        collar=3,
        detach_displacement=5.0 * asset.d_hat,
    )
    scene.build()
    controller.initialize()
    return scene, tape, hub, asset, controller


def _hub_world_verts(hub) -> np.ndarray:
    """Hub collision-geometry vertices in world frame (same composition the
    coupler's ABD build and the ground preflight use)."""
    import genesis.utils.geom as gu
    from genesis.utils.misc import tensor_to_array

    out = []
    for link in hub.links:
        p_link = tensor_to_array(link.get_pos()).reshape(3).astype(np.float64)
        R_link = gu.quat_to_R(tensor_to_array(link.get_quat()).reshape(4).astype(np.float64))
        for geom in link.geoms:
            v = geom.init_verts.astype(np.float64, copy=True)
            R_geom = gu.quat_to_R(np.asarray(geom.init_quat, dtype=np.float64))
            v = v @ R_geom.T + np.asarray(geom.init_pos, dtype=np.float64)
            out.append(v @ R_link.T + p_link)
    return np.concatenate(out, axis=0)


@needs_tape_asset
def test_tape_roll_adhesion_follows_asset_params(show_viewer):
    """add_tape_roll sources Phase-1 adhesion + friction from the asset params."""
    tape_mod = _tape_module()
    asset = tape_mod.TapeAsset.from_npz(TAPE_ASSET_PATH)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.QIPCCouplerOptions(**tape_mod.recommended_coupler_options(asset)),
        show_viewer=show_viewer,
    )
    tape_mod.add_tape_roll(scene, asset, pos=(0.0, 0.0, 0.2))
    requests = scene.sim.coupler.adhesion._requests
    assert len(requests) == 2  # tape-tape self + tape-hub
    params = asset.params
    for request in requests:
        assert request.Cn == pytest.approx(float(params.get("CN", 1.0)))
        assert request.eta == pytest.approx(float(params.get("ETA", 100.0)))
        assert request.bonding_rate == pytest.approx(float(params.get("BONDING_RATE", 1.0)))
        assert request.beta0 == 1.0
        assert request.friction == pytest.approx(float(params.get("MU", 0.5)))

    # explicit overrides still win
    scene2 = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        coupler_options=gs.options.QIPCCouplerOptions(**tape_mod.recommended_coupler_options(asset)),
        show_viewer=show_viewer,
    )
    tape_mod.add_tape_roll(scene2, asset, pos=(0.0, 0.0, 0.2), tape_tape_adhesion=dict(Cn=7.0))
    assert scene2.sim.coupler.adhesion._requests[0].Cn == 7.0


@needs_tape_asset
def test_tape_roll_hub_concentric(show_viewer):
    """Regression: the coil and the hub must land concentric.

    add_tape_roll bakes R(euler) @ v + pos into the TAPE mesh because the FEM
    loader pivots the morph rotation about the vertex COM -- with this asset's
    off-center coil COM (free tail) that was a ~7mm shift, embedding the hub
    wall inside the coil. The hub keeps morph placement (COM-centered ring:
    origin-pivot == COM-pivot under either rigid align semantics); a FREE hub
    exercises the align=True auto-reframing path, which cancels baked world
    coordinates and is why the hub is NOT baked.
    """
    scene, tape, hub, asset = _build_roll_scene(show_viewer, sticky=True, hub_fixed=False)

    # The hub's actual collision geometry must be centered at ROLL_POS (the
    # misalignment regression left it at the world origin, 0.55 m away). AABB
    # center: exact for the symmetric ring, robust to convex decomposition.
    hub_verts = _hub_world_verts(hub)
    hub_center = 0.5 * (hub_verts.min(axis=0) + hub_verts.max(axis=0))
    assert np.linalg.norm(hub_center - ROLL_POS) < 2e-3, f"hub center {hub_center} != {ROLL_POS}"

    # And the coil must be concentric with it. The wound asset's innermost turn
    # can sit slightly BELOW the analytic r_out (it was wound against cgq's
    # 48-gon hub, whose faces are inscribed), so the pass-through detector is
    # the bore count plus a wall-embedding bound -- the misaligned import gave
    # min_gap = -7 mm with 119 verts inside the bore.
    pos = tape.get_state().pos[0].cpu().numpy()
    d = pos - hub_center
    axis = np.array([0.0, 1.0, 0.0])  # default euler=(90,0,0): asset +z -> world +y
    axial = d @ axis
    radial = np.linalg.norm(d - np.outer(axial, axis), axis=1)
    in_hub_band = np.abs(axial) < 0.5 * asset.hub_height
    assert in_hub_band.any()
    inside_bore = int(((radial < asset.hub_r_inner) & in_hub_band).sum())
    assert inside_bore == 0, f"{inside_bore} tape verts passed through the hub wall into the bore"
    min_gap = float((radial[in_hub_band] - asset.hub_r_outer).min())
    assert abs(min_gap) < 10 * asset.d_hat, f"innermost turn should seat near the hub (gap {min_gap:.6f} m)"


@needs_tape_asset
def test_tape_hub_mesh_is_exact(show_viewer):
    """The ring hub reaches QIPC unprocessed: wind vertex count, order and mass.

    add_tape_roll passes the hub through with convexify=False. Genesis's default
    rigid processing would convex-decompose the annulus (8 hulls filling the
    bore: 160 verts, mass +43%) and renumber its vertices, which both distorts
    the contact surface and breaks the wind-time ids seed_asset_locks needs.
    """
    tape_mod = _tape_module()
    scene, _tape, hub, asset = _build_roll_scene(show_viewer, sticky=True, hub_fixed=False)
    src_verts, _src_tris = tape_mod.make_ring_hub(asset.hub_r_outer, asset.hub_r_inner, asset.hub_height)

    geoms = [g for link in hub.links for g in link.geoms]
    assert len(geoms) == 1, f"hub should stay a single un-decomposed geom (got {len(geoms)})"
    assert len(geoms[0].init_verts) == len(src_verts), (
        f"hub vertex count {len(geoms[0].init_verts)} != authored {len(src_verts)}"
    )
    # Vertex ORDER is what the wind-saved ids index; a rigid fit that lines the
    # two meshes up vertex-for-vertex proves it survived (float32 geom storage
    # sets the residual floor).
    R, t = tape_mod._kabsch(src_verts, geoms[0].init_verts.astype(np.float64))
    assert float(np.abs(src_verts @ R.T + t - geoms[0].init_verts).max()) < 1e-6

    # ABD body sees exactly those vertices, and the mass is the polyhedral
    # ring's (analytic annulus scaled by the 48-gon's inscribed-area ratio).
    assert int(scene.sim.coupler._scene.affine_body.n_verts) == len(src_verts)
    n_sides = len(src_verts) // 4
    poly_ratio = n_sides / (2.0 * np.pi) * np.sin(2.0 * np.pi / n_sides)
    exact_mass = np.pi * (asset.hub_r_outer**2 - asset.hub_r_inner**2) * asset.hub_height * 1000.0 * poly_ratio
    link_mass = [link for link in hub.links if link.geoms][0].inertial_mass
    assert link_mass == pytest.approx(exact_mass, rel=1e-3)


@needs_tape_asset
def test_tape_asset_locks_seed_automatically_and_survive_reset(show_viewer):
    """Authored locks auto-seed, map shifted hub ids, and survive resets.

    A rigid entity deliberately precedes the tape hub, so source hub ids cannot
    pass through unchanged. The importer must map them into the hub's shifted
    QIPC vertex range while independently rebasing FEM ids.
    """
    tape_mod = _tape_module()
    scene, tape, _hub, asset = _build_roll_scene(
        show_viewer,
        sticky=True,
        prepend_rigid=True,
    )
    assert asset.bond_topos is not None, "lock asset should carry wind-saved bond topologies"
    assert asset.bond_topos_space == "global"
    assert asset.bond_fem_gvo > 0

    adhesion = scene.sim.coupler.adhesion
    fem_global_offset = adhesion.fem_global_vertex_offset()
    hub_vertex_offset = fem_global_offset - asset.bond_fem_gvo
    assert hub_vertex_offset > 0

    source = asset.bond_topos.astype(np.int64)
    source_is_fem = source >= asset.bond_fem_gvo
    expected = np.where(
        source_is_fem,
        source - asset.bond_fem_gvo + fem_global_offset,
        source + hub_vertex_offset,
    ).astype(np.int32)

    def rows(topologies: np.ndarray) -> set[tuple[int, int, int, int]]:
        return {tuple(int(value) for value in row) for row in topologies}

    assert adhesion.get_bond_count() == 454
    assert rows(adhesion.get_bond_topos()) == rows(expected)
    # Backward-compatible helper reports the automatic build result rather than
    # trying to seed the same rows a second time.
    assert tape_mod.seed_asset_locks(scene, tape, asset) == (454, 0)

    for _ in range(2):
        scene.step()
        assert np.isfinite(tape.get_state().pos[0].cpu().numpy()).all()
        scene.reset()
        assert adhesion.get_bond_count() == 454
        assert rows(adhesion.get_bond_topos()) == rows(expected)


@needs_tape_asset
def test_tape_bond_cluster_releases_and_replays_after_reset(show_viewer):
    """A released/moved front vertex melts the collar and reset restores it."""
    tape_mod = _tape_module()
    scene, tape, hub, asset, controller = _build_cluster_roll_scene(show_viewer)
    coupler = scene.sim.coupler
    cluster = controller._cluster

    assert hub.material.qipc_abd_kappa == 5e7
    assert controller.initial_member_count == 983
    assert controller.member_count == 983
    assert cluster.member_count == 983
    assert coupler.adhesion.get_bond_count() == 454
    initial_positions = tape.get_state().pos[0].clone()

    triangles, bonded, adjacency = tape_mod._bond_cluster_certificate(asset)
    member_vertices = np.unique(triangles[controller._member].reshape(-1))
    is_member_vertex = np.zeros(len(bonded), dtype=bool)
    is_member_vertex[member_vertices] = True
    candidate = None
    for vertex in np.flatnonzero(bonded & ~is_member_vertex):
        freed = ~bonded
        freed[vertex] = True
        target = tape_mod._bond_cluster_target(triangles, bonded, freed, adjacency, 3)
        if (controller._member & ~target).any():
            candidate = int(vertex)
            break
    assert candidate is not None

    vertex_range = cluster.fem_vertex_range
    global_vertex = coupler.adhesion.fem_global_vertex_offset() + vertex_range.start + candidate
    fem_position = coupler._scene.finite_element.x[vertex_range.start + candidate]
    fem_position[0] += 100.0 * asset.d_hat
    scene.step()
    released = coupler.adhesion.get_released_bond_topos()
    assert global_vertex in released
    fem_position[0] += 10.0 * asset.d_hat

    bonds_before_melt = {tuple(int(value) for value in row) for row in coupler.adhesion.get_bond_topos()}
    melted = controller.before_step()
    assert melted > 0
    assert controller.member_count == 983 - melted
    assert cluster.member_count == controller.member_count
    assert controller.released_total > 0
    bonds_after_melt = {tuple(int(value) for value in row) for row in coupler.adhesion.get_bond_topos()}
    cleared_bonds = bonds_before_melt - bonds_after_melt
    assert cleared_bonds

    scene.reset()
    controller.reset()
    assert controller.member_count == 983
    assert controller.released_total == 0
    assert controller.melted_total == 0
    assert cluster.member_count == 983
    assert coupler.adhesion.get_bond_count() == 454
    np.testing.assert_allclose(
        tape.get_state().pos[0].cpu().numpy(),
        initial_positions.cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    )


@needs_tape_asset
@pytest.mark.required
def test_tape_roll_import_holds(show_viewer):
    """Imported prestressed coil holds together via beta0=1 adhesion + bonds."""
    scene, tape, _hub, _asset = _build_roll_scene(show_viewer, sticky=True)
    r0 = _coil_max_radius(tape)
    for _ in range(100):
        scene.step()
    r = _coil_max_radius(tape)
    assert np.isfinite(tape.get_state().pos[0].cpu().numpy()).all()
    assert r < 1.15 * r0, f"sticky coil should hold (max radius {r0:.4f} -> {r:.4f})"


@needs_tape_asset
def test_tape_roll_unsticky_springs_open(show_viewer):
    """With adhesion disabled the prestressed coil unrolls (control)."""
    scene, tape, _hub, _asset = _build_roll_scene(show_viewer, sticky=False)
    r0 = _coil_max_radius(tape)
    for _ in range(250):  # unrolling takes O(100s) of steps (cf. cgq wind release phase)
        scene.step()
    r = _coil_max_radius(tape)
    assert r > 1.25 * r0, f"unsticky prestressed coil should spring open (max radius {r0:.4f} -> {r:.4f})"
