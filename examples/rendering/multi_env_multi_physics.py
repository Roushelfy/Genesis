import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
            particle_size=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            visualize_mpm_boundary=True,
            visualize_sph_boundary=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Plane(),
    )

    cube = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Box(
            pos=(0.5, 0.5, 0.2),
            size=(0.2, 0.2, 0.2),
            euler=(30, 40, 0),
            fixed=True,
        ),
    )

    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=1.0,
            pos=(0.5, 0.5, 0.5),
            euler=(180.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )

    worm = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/worm/worm.obj",
            pos=(0.3, 0.3, 0.001),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=gs.materials.MPM.Muscle(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            model="neohooken",
            n_groups=4,
        ),
    )
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    ########################## build ##########################
    scene.build(n_envs=4, env_spacing=(2.0, 2.0))

    horizon = 500

    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
