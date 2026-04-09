"""
Load and visualize the SHARPA Wave hand (URDF) in Genesis.

Usage:
    python examples/IPC_demo/view_sharpa_hand.py
    python examples/IPC_demo/view_sharpa_hand.py --hand right
    python examples/IPC_demo/view_sharpa_hand.py --hand dual
    python examples/IPC_demo/view_sharpa_hand.py --variant with_wrist
"""

import argparse
from pathlib import Path

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARPA_ROOT = REPO_ROOT / "DemoAssets" / "sharpa-urdf-usd-xml" / "wave_01"

# Map (hand, variant) to URDF path relative to SHARPA_ROOT
URDF_MAP = {
    ("left", "base"): "left_sharpa_wave/left_sharpa_wave.urdf",
    ("left", "with_flange"): "left_sharpa_wave/left_sharpa_wave_with_flange.urdf",
    ("left", "with_wrist"): "left_sharpa_wave/left_sharpa_wave_with_wrist.urdf",
    ("right", "base"): "right_sharpa_wave/right_sharpa_wave.urdf",
    ("right", "with_flange"): "right_sharpa_wave/right_sharpa_wave_with_flange.urdf",
    ("right", "with_wrist"): "right_sharpa_wave/right_sharpa_wave_with_wrist.urdf",
    ("dual", "base"): "dual_sharpa_wave/dual_sharpa_wave.urdf",
    ("dual", "with_flange"): "dual_sharpa_wave/dual_sharpa_wave_with_flange.urdf",
    ("dual", "with_wrist"): "dual_sharpa_wave/dual_sharpa_wave_with_wrist.urdf",
}


def main():
    parser = argparse.ArgumentParser(description="Visualize SHARPA Wave hand in Genesis")
    parser.add_argument("--hand", default="left", choices=["left", "right", "dual"])
    parser.add_argument("--variant", default="base", choices=["base", "with_flange", "with_wrist"])
    args = parser.parse_args()

    urdf_rel = URDF_MAP.get((args.hand, args.variant))
    if urdf_rel is None:
        raise ValueError(f"No URDF for hand={args.hand}, variant={args.variant}")
    urdf_path = SHARPA_ROOT / urdf_rel
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"

    print(f"Loading SHARPA hand: {urdf_path}")

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    hand = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0.0, 0.0, 0.2),
            fixed=True,
        ),
    )

    scene.build()

    print(f"Hand entity: {hand.name}, joints: {hand.n_joints}, dofs: {hand.n_dofs}")

    while True:
        scene.step()


if __name__ == "__main__":
    main()
