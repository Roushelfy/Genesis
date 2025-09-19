import genesis as gs
import logging

def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG)
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

    # Both FEM and Rigid bodies will be added to IPC for unified contact simulation
    # FEM bodies use StableNeoHookean constitution, Rigid bodies use ABD constitution

    scene.add_entity(gs.morphs.Plane())
    SCENE_POS = (0.0, 0.0, 0.0)

    # FEM entities (added to IPC as deformable bodies)
    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (-0.0, -0.0, 0.8)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.5)),
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0.0, 0.5)))), size=(0.2, 0.2, 0.2)),
        material=gs.materials.FEM.Elastic(E=1.0e6, nu=0.45, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    # Rigid bodies (added to both Genesis rigid solver AND IPC as ABD objects)
    # This enables contact between rigid bodies and FEM bodies through IPC

    # Rigid cube  - above the FEM cube
    # rigid_cube = scene.add_entity(
    #     morph=gs.morphs.Box(
    #         pos=tuple(map(sum, zip(SCENE_POS, (0, 0, 1.1)))),
    #         size=(0.08, 0.08, 0.08),
    #         euler=(0, 0, 0)
    #     ),
    #     material=gs.materials.Rigid(rho=800.0, friction=0.3),
    #     surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.2, 0.8)),
    # )

   # a rigid go2 dog
    rigid_dog = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0, 1.0),
            scale=0.5,
        ),
    )

    scene.build()

    # Show IPC GUI for debugging
    print("Scene built successfully!")
    print("Launching IPC debug GUI...")

    for i_step in range(1000):
        if i_step % 10 == 0:
            gs.logger.debug(f"Step {i_step}: Scene time = {scene.t * scene.dt:.3f}s")
            # Also print ABD transforms for debugging
            if hasattr(scene.rigid_solver, '_abd_affines'):
                for geom_idx, env_dict in scene.rigid_solver._abd_affines.items():
                    for env_idx, transform in env_dict.items():
                        print(f"  ABD Geom {geom_idx} Env {env_idx} Transform:\n{transform}")
        scene.step()


if __name__ == "__main__":
    main()