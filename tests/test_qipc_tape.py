"""QIPC coupler tape-import tests: prestress (rest_geometry), Cloth membrane
options, and wound-roll import (design: docs/adhesion_tape_design.md, A5.2).

The wound-roll tests consume a real wind asset (cgq adhesive_tape_wind --save);
set QIPC_TAPE_ASSET or place it at ~/workspace/qipc-test/assets/tape_roll.npz.
"""

import os

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs


def _tape_module():
    # Deferred: importing the coupler package compiles quadrants kernels whose
    # type annotations need gs.init() (provided per-test by the autouse fixture).
    from genesis.engine.couplers.qipc_coupler import tape

    return tape

TAPE_ASSET_PATH = os.environ.get("QIPC_TAPE_ASSET", "")

needs_tape_asset = pytest.mark.skipif(
    not (TAPE_ASSET_PATH and os.path.exists(TAPE_ASSET_PATH)),
    reason="set QIPC_TAPE_ASSET to a wound-roll npz (generate with cgq adhesive_tape_wind --save)",
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
    assert tape_mod.recommended_coupler_options(asset)["contact_d_hat"] == asset.d_hat


ROLL_POS = np.array([0.0, 0.0, 0.2])


def _coil_max_radius(tape) -> float:
    """Max vertex distance from the roll axis (world y through ROLL_POS)."""
    pos = tape.get_state().pos[0].cpu().numpy()
    return float(np.sqrt((pos[:, 0] - ROLL_POS[0]) ** 2 + (pos[:, 2] - ROLL_POS[2]) ** 2).max())


def _build_roll_scene(show_viewer, *, sticky: bool):
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
    adhesion_off = dict(Cn=0.0, Ct=0.0, bonding_rate=0.0, beta0=0.0, enabled=False, distance_lock=False)
    tape, hub = tape_mod.add_tape_roll(
        scene,
        asset,
        pos=tuple(ROLL_POS),
        with_hub=True,
        hub_fixed=True,
        tape_tape_adhesion=None if sticky else adhesion_off,
        tape_hub_adhesion=None if sticky else adhesion_off,
    )
    scene.build()
    return scene, tape, hub, asset


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
