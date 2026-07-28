"""QIPC coupler FEM smoke tests (P1/P2), headless with rendered videos.

Usage (inside a GPU container):
    python examples/qipc/fem_smoke.py --test box_drop
    python examples/qipc/fem_smoke.py --test cloth_drape
    python examples/qipc/fem_smoke.py --test cloth_pin
    python examples/qipc/fem_smoke.py --test all
"""

import argparse
import os
import sys

import numpy as np

import genesis as gs

OUT_DIR = os.environ.get("QIPC_FEM_OUT", "/mnt/home/zhaofeng/workspace/qipc-test/output")


def make_grid_cloth_obj(path: str, n: int = 21, size: float = 1.0) -> None:
    """Write a flat n x n grid cloth (xy plane, centered at origin) as OBJ."""
    xs = np.linspace(-size / 2, size / 2, n)
    ys = np.linspace(-size / 2, size / 2, n)
    with open(path, "w") as fh:
        for y in ys:
            for x in xs:
                fh.write(f"v {x:.6f} {y:.6f} 0.0\n")
        for j in range(n - 1):
            for i in range(n - 1):
                v0 = j * n + i + 1
                v1 = j * n + i + 2
                v2 = (j + 1) * n + i + 1
                v3 = (j + 1) * n + i + 2
                fh.write(f"f {v0} {v1} {v3}\n")
                fh.write(f"f {v0} {v3} {v2}\n")


def run_scene(scene, cam, n_steps, video_name, per_step=None, render_every=2, fps=50):
    cam.start_recording()
    for i in range(n_steps):
        if per_step is not None:
            per_step(i)
        scene.step()
        scene.rigid_solver._func_update_geoms(scene._envs_idx)
        if i % render_every == 0:
            cam.render()
    path = os.path.join(OUT_DIR, video_name)
    cam.stop_recording(save_to_filename=path, fps=fps)
    print(f"[video] {path}")


def test_box_drop() -> bool:
    """Volumetric FEM (SNH) box drops onto the ground plane."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.005,
            init_collision_pair_capacity=20000,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.FEM.Elastic(
            E=5e4, nu=0.45, rho=1000.0, friction_mu=0.5, model="stable_neohookean"
        ),
        surface=gs.surfaces.Plastic(color=(0.8, 0.4, 0.2, 1.0)),
    )
    cam = scene.add_camera(res=(960, 720), pos=(0.8, -0.8, 0.5), lookat=(0.0, 0.0, 0.15), fov=40, GUI=False)
    scene.build()

    z_trace = []

    def per_step(i):
        pos = box.get_state().pos[0]
        z_trace.append(float(pos[:, 2].mean()))

    run_scene(scene, cam, 120, "fem_box_drop.mp4", per_step)

    z0, z_mid, z_end = z_trace[0], z_trace[30], z_trace[-1]
    pos_end = box.get_state().pos[0].cpu().numpy()
    vel_mid_probe = box.get_state().vel[0].cpu().numpy()
    ok = True
    # Fell from 0.3, settled resting on the ground (half-height 0.05 above z=0)
    if not (z_mid < z0 - 0.02):
        print(f"FAIL box_drop: not falling (z0={z0:.4f}, z30={z_mid:.4f})")
        ok = False
    if not (0.02 < z_end < 0.09):
        print(f"FAIL box_drop: bad rest height z_end={z_end:.4f}")
        ok = False
    if not np.isfinite(pos_end).all():
        print("FAIL box_drop: NaN positions")
        ok = False
    if not (pos_end[:, 2].min() > -0.01):
        print(f"FAIL box_drop: penetrated ground min_z={pos_end[:, 2].min():.4f}")
        ok = False
    print(f"box_drop: z0={z0:.4f} z30={z_mid:.4f} z_end={z_end:.4f} min_z={pos_end[:, 2].min():.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_cloth_drape() -> bool:
    """Cloth drapes over a fixed rigid box under gravity."""
    obj_path = os.path.join(OUT_DIR, "grid21.obj")
    make_grid_cloth_obj(obj_path, n=21, size=1.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.003,
            init_collision_pair_capacity=40000,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        # Lifted d_hat above the ground: a vertex flush with a halfplane (d=0)
        # trips a kernel assert in QIPC's halfplane query (device trap at init).
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.104), size=(0.2, 0.2, 0.2), fixed=True),
        material=gs.materials.Rigid(rho=500, coup_friction=0.5),
        surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 1.0)),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=obj_path, scale=0.5, pos=(0.0, 0.0, 0.25)),
        material=gs.materials.FEM.Cloth(
            E=1e4, nu=0.3, rho=200.0, thickness=0.001, bending_stiffness=1.0, friction_mu=0.5
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )
    cam = scene.add_camera(res=(960, 720), pos=(0.9, -0.9, 0.6), lookat=(0.0, 0.0, 0.15), fov=40, GUI=False)
    scene.build()

    run_scene(scene, cam, 200, "fem_cloth_drape.mp4")

    pos = cloth.get_state().pos[0].cpu().numpy()
    ok = True
    if not np.isfinite(pos).all():
        print("FAIL cloth_drape: NaN positions")
        ok = False
    # Center of cloth should rest on top of the box (z ~= 0.2), skirt hangs below
    center_idx = np.argmin(np.linalg.norm(pos[:, :2], axis=1))
    z_center = pos[center_idx, 2]
    if not (0.15 < z_center < 0.28):
        print(f"FAIL cloth_drape: center z={z_center:.4f} not resting on box top (0.2)")
        ok = False
    if not (pos[:, 2].min() > -0.02):
        print(f"FAIL cloth_drape: penetrated ground min_z={pos[:, 2].min():.4f}")
        ok = False
    print(f"cloth_drape: center_z={z_center:.4f} min_z={pos[:, 2].min():.4f} max_z={pos[:, 2].max():.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_cloth_pin() -> bool:
    """P2: pin two cloth corners with soft constraints, then drag one corner."""
    obj_path = os.path.join(OUT_DIR, "grid21.obj")
    make_grid_cloth_obj(obj_path, n=21, size=1.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.003,
            init_collision_pair_capacity=20000,
            fem_constraint_strength=1000.0,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=obj_path, scale=0.5, pos=(0.0, 0.0, 0.5)),
        material=gs.materials.FEM.Cloth(
            E=1e4, nu=0.3, rho=200.0, thickness=0.001, bending_stiffness=1.0, friction_mu=0.5
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )
    cam = scene.add_camera(res=(960, 720), pos=(1.1, -1.1, 0.7), lookat=(0.0, 0.0, 0.35), fov=40, GUI=False)
    scene.build()

    # Find the two corners at y = +0.25 edge
    verts0 = cloth.get_state().pos[0].cpu().numpy()
    corner_a = int(np.argmin(np.linalg.norm(verts0[:, :2] - np.array([-0.25, 0.25]), axis=1)))
    corner_b = int(np.argmin(np.linalg.norm(verts0[:, :2] - np.array([0.25, 0.25]), axis=1)))
    target_a = verts0[corner_a].copy()
    target_b = verts0[corner_b].copy()

    cloth.set_vertex_constraints([corner_a, corner_b], is_soft_constraint=True)

    drag_target = target_b.copy()

    def per_step(i):
        # After settling, drag corner_b along a horizontal arc
        if i >= 120:
            t = (i - 120) * 0.01
            drag_target[0] = target_b[0] + 0.25 * np.sin(0.8 * t)
            drag_target[2] = target_b[2] - 0.1 * (1 - np.cos(0.8 * t))
            cloth.update_constraint_targets([corner_b], drag_target[None])

    run_scene(scene, cam, 320, "fem_cloth_pin.mp4", per_step)

    pos = cloth.get_state().pos[0].cpu().numpy()
    ok = True
    if not np.isfinite(pos).all():
        print("FAIL cloth_pin: NaN positions")
        ok = False
    err_a = np.linalg.norm(pos[corner_a] - target_a)
    err_b = np.linalg.norm(pos[corner_b] - drag_target)
    if err_a > 0.05:
        print(f"FAIL cloth_pin: pinned corner drifted err_a={err_a:.4f}")
        ok = False
    if err_b > 0.08:
        print(f"FAIL cloth_pin: dragged corner not tracking err_b={err_b:.4f}")
        ok = False
    # Cloth must sag (free edge below pinned edge)
    if not (pos[:, 2].min() < target_a[2] - 0.2):
        print(f"FAIL cloth_pin: cloth did not sag min_z={pos[:, 2].min():.4f}")
        ok = False
    print(f"cloth_pin: err_a={err_a:.4f} err_b={err_b:.4f} min_z={pos[:, 2].min():.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


TESTS = {
    "box_drop": test_box_drop,
    "cloth_drape": test_cloth_drape,
    "cloth_pin": test_cloth_pin,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all", choices=[*TESTS, "all"])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    gs.init(precision="64", logging_level="info")

    names = list(TESTS) if args.test == "all" else [args.test]
    results = {}
    for name in names:
        print(f"\n===== {name} =====")
        results[name] = TESTS[name]()
        if name != names[-1]:
            gs.destroy()
            gs.init(precision="64", logging_level="info")

    print("\n===== summary =====")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
