import genesis as gs


def main():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),  # (translation, rotation) strength ratios
            IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
        ),
        vis_options=gs.options.VisOptions(),
        viewer_options=gs.options.ViewerOptions(),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    # IPC will hijack the FEM material and route to IpcSolver

    scene.add_entity(gs.morphs.Plane())
    SCENE_POS = (0.5, 0.5, 1.0)
    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (-0.0, -0.0, 0.3)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e3, nu=0.45, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.5)),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0.0, 0)))), size=(0.2, 0.2, 0.2),euler=(0,0,45)),
        material=gs.materials.FEM.Elastic(E=1.0e3, nu=0.45, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build(n_envs=0, env_spacing=(1.0, 1.0))
    for i_step in range(10000):
        print(f"Step {i_step}")
        scene.step()


if __name__ == "__main__":
    main()
