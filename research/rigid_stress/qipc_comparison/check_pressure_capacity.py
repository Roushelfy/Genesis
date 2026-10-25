"""Exercise exhaustive shell-pressure fallback beyond the shared face-list capacity."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    create_contacts,
    kernel_anchor,
    kernel_pack_contacts,
    kernel_pressure_correct_warp,
    kernel_pressure_warp,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu.geometry import plate_mesh

from .artifacts import runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    vertices, quads = plate_mesh(16, Lx=0.04)
    asset = args.output / "plate.npz"
    np.savez(asset, vertices=vertices, quads=quads)
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset, backend="shell", method="pcg", output_mode="full", quadrature=16
    )
    model = ShellModel(options)
    surface = StressSurface(16, model.load_info)
    state = model.create_state(2, "full")
    contacts = create_contacts(1, 2, 0.018, hertz_profile=True)
    positions = np.array([[[-0.015, 0.0, 0.00025], [-0.015, 0.0, -0.00025]]])
    forces = np.array([[[0.001, 0.0, -0.01], [0.001, 0.0, 0.01]]])
    contacts.position.from_numpy(positions)
    contacts.force.from_numpy(forces)
    contacts.normal.from_numpy(np.array([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]]))
    contacts.valid.fill(True)
    contacts.friction.fill(0.5)
    kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
    kernel_pack_contacts(contacts)
    kernel_pressure_warp(contacts, surface.info, True)
    initial = qd_to_numpy(contacts.status, transpose=True)
    centers = qd_to_numpy(contacts.center, transpose=True)[:, 0]
    low = qd_to_numpy(surface.info.face_bounds_low, transpose=True)
    high = qd_to_numpy(surface.info.face_bounds_high, transpose=True)
    distance = np.maximum(low[None] - centers[:, None], 0) + np.maximum(centers[:, None] - high[None], 0)
    counts = (np.sum(distance**2, axis=-1) < 0.018**2).sum(axis=1)
    result = dict(
        identity=runtime_identity(), initial_status=initial.tolist(), candidate_faces=counts.tolist(), passed=False
    )
    write_json(args.output / "result.json", result)
    assert (initial == 5).all(), initial
    assert (counts > 128).all(), counts
    kernel_pressure_correct_warp(contacts, surface.info)
    status = qd_to_numpy(contacts.status, transpose=True)
    assert (status == 0).all(), status
    kernel_scatter(contacts, state, model.load_info, surface.info)
    load = qd_to_numpy(state.force, transpose=True).reshape(2, -1, 6)
    net_force = load[:, :, :3].sum(axis=1)
    net_moment = (np.cross(vertices[None], load[:, :, :3]) + load[:, :, 3:]).sum(axis=1)
    wrench_error = max(
        float(np.linalg.norm(net_force - forces[0], axis=1).max() / 0.01),
        float(np.linalg.norm(net_moment - np.cross(positions[0], forces[0]), axis=1).max() / (0.01 * 0.018)),
    )
    assert wrench_error < 1e-8
    np.savez_compressed(args.output / "load.npz", load=load, status=status)
    if args.reference:
        reference = np.load(args.reference / "load.npz")
        error = np.linalg.norm(load - reference["load"]) / np.linalg.norm(reference["load"])
        assert error < 1e-10
        np.testing.assert_array_equal(status, reference["status"])
        result["paired_load_relative"] = float(error)
    result.update(
        passed=True,
        final_status=status.tolist(),
        wrench_error=wrench_error,
        pressure_evaluations=qd_to_numpy(contacts.evaluations, transpose=True).tolist(),
    )
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
