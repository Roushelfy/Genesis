"""QIPC coupler: a wound tape roll dropped on the ground.

The smallest scene exercising the tape import path -- ground plane, coil, ring hub, no robot.
--mode picks which wound asset to drop, and the asset's wind-time LOCK flag selects the
adhesion mode through recommended_coupler_options:

    --mode bond   tape_roll_lock.npz: Phase-2 distance bonds, the coil holds itself firmly.
    --mode soft   tape_roll_soft.npz: only the beta-state stickiness holds the coil, so it
                  gradually unrolls once it lands.

Assets ship in genesis/assets/qipc/; see tape_teleop.py for how to regenerate them.
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir

ROLL_XY = (0.55, 0.0)
ROLL_EULER = (90.0, 0.0, 0.0)  # asset hub axis +z -> world +y (roll stands on its tread)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bond", "soft"], default="bond")
    parser.add_argument("--asset", type=str, default=None, help="tape roll npz (default per --mode)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("-v", "--vis", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    gs.init(precision="64", logging_level="info")

    # The tape module annotates its kernels with gs.qd_float, which only exists after gs.init.
    from genesis.engine.couplers.qipc_coupler.tape import TapeAsset, add_tape_roll, recommended_coupler_options

    suffix = {"bond": "lock", "soft": "soft"}[args.mode]
    asset_path = args.asset or os.path.join(get_assets_dir(), "qipc", f"tape_roll_{suffix}.npz")
    asset = TapeAsset.from_npz(asset_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.QIPCCouplerOptions(**recommended_coupler_options(asset)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -0.8, 0.4),
            camera_lookat=(0.55, 0.0, 0.1),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Seat the roll on the ground (cgq drop convention: lowest coil vertex one
    # contact band above the plane). add_tape_roll bakes the transform into the
    # meshes, so this analytic height is exact. Do NOT spawn the roll high and
    # let it fall: this IS the settled pose, so dropping only adds a landing
    # transient. (An airborne locked coil also needs the tighter Newton
    # tolerance recommended_coupler_options now ships -- see tape_lift_drop.py.)
    coil_rot = asset.tape_positions @ gu.quat_to_R(gu.xyz_to_quat(np.array(ROLL_EULER), degrees=True)).T
    roll_z = -float(coil_rot[:, 2].min()) + asset.thick + 0.5 * asset.d_hat
    tape, _hub = add_tape_roll(
        scene,
        asset,
        pos=(ROLL_XY[0], ROLL_XY[1], roll_z),
        euler=ROLL_EULER,
    )

    scene.build()
    gs.logger.info(f"tape mode={args.mode} asset={asset_path} roll_z={roll_z:.4f}")

    for i in range(args.steps):
        scene.step()
        if i % 10 == 0:
            gs.logger.info(f"step {i}: tape min_z={float(tape.get_state().pos[..., 2].min()):.5f}")


if __name__ == "__main__":
    main()
