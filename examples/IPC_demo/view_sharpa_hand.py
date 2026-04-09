"""
Load and visualize the SHARPA Wave hand in Genesis.

Supports URDF, MJCF (xml), and USD formats. USD has the best materials
(metallic shaders), URDF/XML only have flat colors.

Usage:
    python examples/IPC_demo/view_sharpa_hand.py
    python examples/IPC_demo/view_sharpa_hand.py --format usd
    python examples/IPC_demo/view_sharpa_hand.py --format xml
    python examples/IPC_demo/view_sharpa_hand.py --hand right --variant with_wrist
"""

import argparse
from pathlib import Path

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARPA_ROOT = REPO_ROOT / "DemoAssets" / "sharpa-urdf-usd-xml" / "wave_01"

# Map (hand, variant) to file stems relative to SHARPA_ROOT.
# The actual extension is determined by --format.
FILE_MAP = {
    ("left", "base"): "left_sharpa_wave/left_sharpa_wave",
    ("left", "with_flange"): "left_sharpa_wave/left_sharpa_wave_with_flange",
    ("left", "with_wrist"): "left_sharpa_wave/left_sharpa_wave_with_wrist",
    ("right", "base"): "right_sharpa_wave/right_sharpa_wave",
    ("right", "with_flange"): "right_sharpa_wave/right_sharpa_wave_with_flange",
    ("right", "with_wrist"): "right_sharpa_wave/right_sharpa_wave_with_wrist",
    ("dual", "base"): "dual_sharpa_wave/dual_sharpa_wave",
    ("dual", "with_flange"): "dual_sharpa_wave/dual_sharpa_wave_with_flange",
    ("dual", "with_wrist"): "dual_sharpa_wave/dual_sharpa_wave_with_wrist",
}

FORMAT_EXT = {
    "urdf": ".urdf",
    "xml": ".xml",
    "usd": ".usda",
}


def main():
    parser = argparse.ArgumentParser(description="Visualize SHARPA Wave hand in Genesis")
    parser.add_argument("--hand", default="left", choices=["left", "right", "dual"])
    parser.add_argument("--variant", default="base", choices=["base", "with_flange", "with_wrist"])
    parser.add_argument("--format", default="urdf", choices=["urdf", "xml", "usd"],
                        help="Asset format: urdf, xml (MJCF), or usd (best materials)")
    args = parser.parse_args()

    stem = FILE_MAP.get((args.hand, args.variant))
    if stem is None:
        raise ValueError(f"No asset for hand={args.hand}, variant={args.variant}")

    ext = FORMAT_EXT[args.format]
    asset_path = SHARPA_ROOT / (stem + ext)
    assert asset_path.exists(), f"Asset not found: {asset_path}"

    print(f"Loading SHARPA hand ({args.format}): {asset_path}")

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

    if args.format == "usd":
        entities = scene.add_stage(
            morph=gs.morphs.USD(
                file=str(asset_path),
                pos=(0.0, 0.0, 0.2),
            ),
        )
        scene.build()
        for e in entities:
            print(f"  Entity: {e.name}, joints: {e.n_joints}, dofs: {e.n_dofs}")
    elif args.format == "xml":
        hand = scene.add_entity(
            gs.morphs.MJCF(
                file=str(asset_path),
                pos=(0.0, 0.0, 0.2),
                fixed=True,
            ),
        )
        scene.build()
        print(f"Hand entity: {hand.name}, joints: {hand.n_joints}, dofs: {hand.n_dofs}")
    else:
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=str(asset_path),
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
