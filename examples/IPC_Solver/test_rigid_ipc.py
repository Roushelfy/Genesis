import genesis as gs


def main():
    """Test rigid body IPC simulation - rigid bodies only"""
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, -9.8)),
        rigid_options=gs.options.RigidOptions(
            use_IPC=True,
        ),
        vis_options=gs.options.VisOptions(),
        viewer_options=gs.options.ViewerOptions(),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    SCENE_POS = (0.5, 0.5, 1.0)

    # Rigid bodies will be added to both Genesis rigid solver AND IPC as ABD objects
    # This enables contact between rigid bodies through IPC

    # Rigid cube 1 - falling from height
    rigid_cube1 = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(map(sum, zip(SCENE_POS, (-0.2, 0, 1.0)))),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 15)
        ),
        material=gs.materials.Rigid(rho=800.0, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
    )

    # Rigid cube 2 - falling from height
    rigid_cube2 = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(map(sum, zip(SCENE_POS, (0.2, 0, 0.8)))),
            size=(0.08, 0.08, 0.08),
            euler=(0, 0, -15)
        ),
        material=gs.materials.Rigid(rho=600.0, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.8)),
    )

    # Rigid sphere - falling from height
    rigid_sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=tuple(map(sum, zip(SCENE_POS, (0, 0.3, 1.2)))),
            radius=0.06
        ),
        material=gs.materials.Rigid(rho=1000.0, friction=0.2),
        surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.8, 0.8)),
    )

    # Static obstacle - fixed rigid body
    obstacle = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(map(sum, zip(SCENE_POS, (0, 0, 0.5)))),
            size=(0.3, 0.05, 0.1),
            euler=(0, 15, 0)
        ),
        material=gs.materials.Rigid(rho=1000.0, friction=0.5),
        surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.2, 0.8)),
    )

    print("Building scene with rigid body IPC...")
    scene.build(n_envs=1, env_spacing=(1.0, 1.0))
    print("Scene built successfully!")

    print("Starting rigid body IPC simulation...")
    for i_step in range(1000):
        if i_step % 100 == 0:
            print(f"Step {i_step}")
        scene.step()

    print("Simulation completed!")


if __name__ == "__main__":
    main()