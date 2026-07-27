"""QIPC port of examples/IPC_Solver/ipc_objects_falling.py, headless + video.

A tilted cloth falls from above onto a free rigid cube and a soft FEM ball
resting on the ground. Exercises: cloth shell + volumetric FEM + free ABD body
+ ground, all in one contact scene.
"""

import os

import numpy as np

import genesis as gs

OUT_DIR = os.environ.get("QIPC_FEM_OUT", "/mnt/home/zhaofeng/workspace/qipc-test/output")


def make_grid_cloth_obj(path: str, n: int = 21, size: float = 1.0) -> None:
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gs.init(precision="64", logging_level="info")

    obj_path = os.path.join(OUT_DIR, "grid21.obj")
    make_grid_cloth_obj(obj_path, n=21, size=1.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=0.005,
            contact_resistance=1e7,
            init_collision_pair_capacity=60000,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=obj_path,
            scale=1.5,
            pos=(0.0, 0.0, 1.0),
            euler=(120, -30, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
            friction_mu=0.3,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.3), size=(0.2, 0.2, 0.2)),
        material=gs.materials.Rigid(rho=500, coup_friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
    )

    soft_ball = scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.08, pos=(0.25, 0.0, 0.1)),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.3, 0.8)),
    )

    cam = scene.add_camera(
        res=(1280, 720), pos=(2.5, 2.5, 1.5), lookat=(0.0, 0.0, 0.3), fov=35, GUI=False
    )

    scene.build()

    qs = scene.sim._coupler._scene.solver
    cam.start_recording()
    for i in range(250):
        scene.step()
        scene.rigid_solver._func_update_geoms(scene._envs_idx)
        if i % 2 == 0:
            cam.render()
        if i % 25 == 0:
            print(f"[step {i}] newton={qs.newton_iters} t={qs.step_ms:.1f}ms", flush=True)

    video_path = os.path.join(OUT_DIR, "qipc_objects_falling.mp4")
    cam.stop_recording(save_to_filename=video_path, fps=50)
    print(f"[video] {video_path}", flush=True)

    cloth_pos = cloth.get_state().pos[0].cpu().numpy()
    ball_pos = soft_ball.get_state().pos[0].cpu().numpy()
    box_pos = box.get_state().pos[0].cpu().numpy() if hasattr(box.get_state(), "pos") else None
    ok = True
    for name, pos in (("cloth", cloth_pos), ("ball", ball_pos)):
        if not np.isfinite(pos).all():
            print(f"FAIL {name}: NaN")
            ok = False
        if pos[:, 2].min() < -0.02:
            print(f"FAIL {name}: below ground min_z={pos[:, 2].min():.4f}")
            ok = False
    # Cloth should have fallen from z=1.0 and settled below 0.6 (draped over objects)
    if cloth_pos[:, 2].max() > 0.7:
        print(f"WARN cloth still high: max_z={cloth_pos[:, 2].max():.3f}")
    print(
        f"final: cloth z=[{cloth_pos[:, 2].min():.3f},{cloth_pos[:, 2].max():.3f}] "
        f"ball z=[{ball_pos[:, 2].min():.3f},{ball_pos[:, 2].max():.3f}] -> {'OK' if ok else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
