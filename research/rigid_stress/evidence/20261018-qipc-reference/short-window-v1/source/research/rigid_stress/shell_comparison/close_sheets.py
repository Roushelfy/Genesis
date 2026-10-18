"""A compact close-sheet capability check of the original PR, with no rigid collider."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .common import grid_sheet, identity, write_json
from .observation import FullField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "sheet.obj"
    grid_sheet(path, 2, 2, 0.02, 0.02)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=False,
    )
    sheets = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=path,
                pos=(0.0, 0.0, z),
                decimate=False,
            ),
            material=gs.materials.Shell(
                E=1e7,
                thickness=0.001,
            ),
        )
        for z in (-0.0025, 0.0025)
    ]
    scene.build(n_envs=1)
    sheets[0].set_verts_vel([0.0, 0.0, 0.025])
    sheets[1].set_verts_vel([0.0, 0.0, -0.025])
    observation = FullField.allocate(scene.sim.shell_solver)
    centers, peaks = [], []
    for _ in range(160):
        scene.step()
        observation.update(scene.sim.shell_solver)
        centers.append([float(tensor_to_array(sheet.get_verts_pos())[0, :, 2].mean()) for sheet in sheets])
        peaks.append(float(tensor_to_array(observation.peaks)[0]))
    centers = np.array(centers)
    separation = centers[:, 1] - centers[:, 0]
    np.savez_compressed(args.output / "trajectory.npz", centers_z_m=centers, peak_pa=np.array(peaks))
    write_json(
        args.output / "result.json",
        {
            "kind": "close-sheet-capability",
            "identity": identity(),
            "steps": 160,
            "dt_s": 0.001,
            "thickness_m": 0.001,
            "initial_midplane_gap_m": 0.005,
            "approach_speed_per_sheet_m_per_s": 0.025,
            "final_signed_separation_m": float(separation[-1]),
            "midplanes_crossed": bool((separation < 0).any()),
            "maximum_peak_pa": max(peaks),
            "interpretation": "Original PR has rigid-shell coupling, but no shell-shell/self-contact response. This is an unsupported capability, not a failed convergence reference.",
        },
    )


if __name__ == "__main__":
    main()
