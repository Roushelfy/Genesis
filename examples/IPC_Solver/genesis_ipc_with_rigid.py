import genesis as gs
import logging

def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(10, 10),  # (translation, rotation) strength ratios
            IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
        ),
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
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (-0.0, -0.0, 0.2)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=1000.0, model="stable_neohookean")
    )

    # cube = scene.add_entity(
    #     morph=gs.morphs.Box(pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0.0, 0.6)))), size=(0.2, 0.2, 0.2)),
    #     material=gs.materials.FEM.Elastic(E=1.0e6, nu=0.45, rho=1000.0, model="stable_neohookean"),
    #     surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    # )

    # Rigid bodies (added to both Genesis rigid solver AND IPC as ABD objects)
    # This enables contact between rigid bodies and FEM bodies through IPC
    rigid_cube1 = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0, 0.4)))),
            size=(0.1, 0.1, 0.1),
            euler=(0, 15, 15)
        ),
        material=gs.materials.Rigid(rho=500, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
    )

   # a rigid go2 dog
    # rigid_dog = scene.add_entity(
    #     gs.morphs.URDF(
    #         file="urdf/go2/urdf/go2.urdf",
    #         pos=(0, 0, 0.4),
    #         scale=0.5,
    #     ),
    # )

    scene.build(n_envs=1)

    # Show IPC GUI for debugging
    print("Scene built successfully!")
    print("Launching IPC debug GUI...")



    for i_step in range(10000):
        if i_step % 100 == 0:
            print(f"Step {i_step}: Scene time = {scene.t * scene.dt:.3f}s")
        scene.step()
    # link_idx = [1]
    # rotation_direction = 1
    # for i in range(1000):
    #     cube_pos = scene.sim.rigid_solver.get_links_pos(link_idx)
    #     cube_pos[:, :, 2] -= 1
    #     force = -100 * cube_pos
    #     scene.sim.rigid_solver.apply_links_external_force(force=force, links_idx=link_idx)

    #     torque = [[[0, 0, rotation_direction * 5]]]
    #     scene.sim.rigid_solver.apply_links_external_torque(torque=torque, links_idx=link_idx)

    #     scene.step()

    #     if (i + 50) % 100 == 0:
    #         rotation_direction *= -1


if __name__ == "__main__":
    main()