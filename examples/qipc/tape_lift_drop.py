"""QIPC coupler: faithful port of cgq's adhesive_tape_drop demo.

The roll lies FLAT on the ground (hub axis up, hub free). The tape's free-end
center vertex is driven through cgq's exact timeline via the resident
SoftPositionConstraint (strength ratio 1e3, matching SPC_STRENGTH):

    hold  30 frames   settle under gravity, free end unconstrained
    pull 300 frames   lift the free-end center to LIFT_HEIGHT (cosine ease)
    top  600 frames   sway at the top: first half +/-x, second half +/-y
    drop 500 frames   release the constraint

with dt=0.01 (1430 frames total). In bond mode the wind-saved lock topologies
are re-seeded before the first step (tape-tape rows only: hub-side ids are tied
to the wind scene's hub tessellation and cannot transfer through Genesis's
convex-decomposed hub; the innermost turn re-bonds via beta0=1) and the whole
spool -- tape AND hub -- ratchets up as a unit. In soft mode the outer wrap
peels off instead.

Expected reference behavior (cgq native, tape_roll_lock.npz, RTX PRO 6000):
settle tape_mean 0.010 -> lift to ~0.13 -> sway -> after release the spool
HOVERS at ~0.126 (qipc distance locks resist rigid translation; gravity never
reaches the 0.5 N per-pair release force).

    python examples/qipc/tape_lift_drop.py                    # viewer
    python examples/qipc/tape_lift_drop.py --video out.mp4    # headless render
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import get_assets_dir

ROLL_XY = (0.55, 0.0)

# ---- cgq adhesive_tape_drop timeline (dt=0.01) ----
DT = 0.01
HOLD_FRAMES = 30
PULL_FRAMES = 300
TOP_FRAMES = 600
FREEFALL_FRAMES = 500
TOTAL_FRAMES = HOLD_FRAMES + PULL_FRAMES + TOP_FRAMES + FREEFALL_FRAMES  # 1430
LIFT_HEIGHT = 0.20
SWAY_AMPLITUDE = 0.05
SPC_STRENGTH_RATIO = 1.0e3
_PULL_END = HOLD_FRAMES + PULL_FRAMES
_TOP_END = _PULL_END + TOP_FRAMES


def _smooth_lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * (0.5 - 0.5 * np.cos(np.pi * t))  # cosine ease


def _smooth_sway(amp, local):
    """velocity-continuous 0 -> +amp -> 0 -> -amp -> 0 over local in [0,1]."""
    local = float(np.clip(local, 0.0, 1.0))
    if local < 0.25:
        return _smooth_lerp(0.0, amp, local / 0.25)
    if local < 0.5:
        return _smooth_lerp(amp, 0.0, (local - 0.25) / 0.25)
    if local < 0.75:
        return _smooth_lerp(0.0, -amp, (local - 0.5) / 0.25)
    return _smooth_lerp(-amp, 0.0, (local - 0.75) / 0.25)


def drive_offset(f, settle_only):
    """(dx, dy, lift_z, constrained) for the free-end center vertex at frame f.

    cgq's drive_offset with its y-up world mapped to Genesis z-up: the lift goes
    to +z, the two sway axes to +/-x then +/-y.
    """
    if settle_only or f < HOLD_FRAMES:
        return 0.0, 0.0, 0.0, False
    if f >= _TOP_END:
        return 0.0, 0.0, 0.0, False  # FREEFALL: release
    lift = _smooth_lerp(0.0, LIFT_HEIGHT, (f - HOLD_FRAMES) / max(PULL_FRAMES, 1))
    dx = dy = 0.0
    if f >= _PULL_END and SWAY_AMPLITUDE > 0.0:
        tt = (f - _PULL_END) / max(TOP_FRAMES, 1)
        if tt < 0.5:
            dx = _smooth_sway(SWAY_AMPLITUDE, tt / 0.5)
        else:
            dy = _smooth_sway(SWAY_AMPLITUDE, (tt - 0.5) / 0.5)
    return dx, dy, lift, True


def phase_at(f):
    if f < HOLD_FRAMES:
        return "hold"
    if f < _PULL_END:
        return "pull"
    if f < _TOP_END:
        return "top"
    return "freefall"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bond", "soft"], default="bond")
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    parser.add_argument("--settle-only", action="store_true")
    parser.add_argument("--video", type=str, default=None, help="headless: render to this mp4")
    parser.add_argument("--render-every", type=int, default=5)
    args = parser.parse_args()
    headless = args.video is not None

    gs.init(precision="64", logging_level="info")

    from genesis.engine.couplers.qipc_coupler.tape import (
        TapeAsset,
        add_tape_roll,
        recommended_coupler_options,
        seed_asset_locks,
    )

    suffix = {"bond": "lock", "soft": "soft"}[args.mode]
    asset_path = args.asset or os.path.join(get_assets_dir(), "qipc", f"tape_roll_{suffix}.npz")
    asset = TapeAsset.from_npz(asset_path)
    opts = recommended_coupler_options(asset)
    if args.mode == "bond":
        opts.update(adhesion_bond_distance_lock=True, adhesion_bond_max_bonds=16384)
    else:
        opts.update(adhesion_bond_distance_lock=False, adhesion_bond_max_bonds=0)
    opts.update(fem_constraint_strength=SPC_STRENGTH_RATIO)  # cgq SPC_STRENGTH

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.QIPCCouplerOptions(**opts),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.95, -0.55, 0.35),
            camera_lookat=(ROLL_XY[0], ROLL_XY[1], 0.15),
            camera_fov=40,
        ),
        show_viewer=not headless,
    )
    scene.add_entity(gs.morphs.Plane())
    cam = None
    if headless:
        cam = scene.add_camera(
            res=(960, 720), pos=(0.95, -0.55, 0.35), lookat=(ROLL_XY[0], ROLL_XY[1], 0.15), fov=40, GUI=False
        )

    # Lie flat (hub axis stays +z), seated one clearance band above the ground.
    # The hub (height 20mm, centered) reaches lower than the 19mm-wide coil.
    lowest = min(float(asset.tape_positions[:, 2].min()), -0.5 * asset.hub_height)
    seat_z = -lowest + asset.thick + 0.5 * asset.d_hat
    tape, hub = add_tape_roll(
        scene,
        asset,
        pos=(ROLL_XY[0], ROLL_XY[1], seat_z),
        euler=(0.0, 0.0, 0.0),
        with_hub=True,
        hub_fixed=False,
        tape_surface=gs.surfaces.Plastic(color=(0.85, 0.75, 0.3, 1.0)),
        hub_surface=gs.surfaces.Plastic(color=(0.4, 0.25, 0.15, 1.0)),
    )

    scene.build()

    if args.mode == "bond":
        n_seeded, n_dropped = seed_asset_locks(scene, tape, asset)
        gs.logger.info(f"seeded {n_seeded} wind-saved locks ({n_dropped} hub-side rows re-bond dynamically)")

    # cgq drives a SINGLE free-end center vertex: vid(nx, nz//2).
    free_id = asset.nx * (asset.nz + 1) + asset.nz // 2

    writer = None
    if headless:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args.video, fps=30, quality=8)

    def render(frame):
        if writer is None:
            return
        if frame % args.render_every:
            return
        scene.rigid_solver._func_update_geoms(scene._envs_idx)
        writer.append_data(cam.render(rgb=True)[0])

    start = None
    constrained_prev = False
    for f in range(args.frames):
        dx, dy, lift, constrained = drive_offset(f, args.settle_only)
        if constrained:
            if start is None:
                start = tape.get_state().pos[0][free_id].cpu().numpy().astype(np.float64).copy()
                tape.set_vertex_constraints([free_id], target_poss=start[None], is_soft_constraint=True)
            tape.update_constraint_targets([free_id], (start + np.array([dx, dy, lift]))[None])
        elif constrained_prev:
            tape.remove_vertex_constraints([free_id])
            start = None
        constrained_prev = constrained

        scene.step()
        render(f)

        if f % 20 == 0:
            pos = tape.get_state().pos[0].cpu().numpy()
            hub_z = float(hub.get_pos().reshape(-1)[2])
            gs.logger.info(
                f"f {f:4d} [{phase_at(f):8s}] dx={dx:+.3f} dy={dy:+.3f} lift={lift:.3f} "
                f"tape_mean_z={pos[:, 2].mean():.4f} free_z={pos[free_id, 2]:.4f} hub_z={hub_z:.4f}"
            )

    if writer is not None:
        writer.close()
        gs.logger.info(f"video written to {args.video}")

    pos = tape.get_state().pos[0].cpu().numpy()
    print(
        f"END: tape_mean_z={pos[:, 2].mean():.4f} tape_min_z={pos[:, 2].min():.4f} "
        f"hub_z={float(hub.get_pos().reshape(-1)[2]):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
