import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("-t", "--seconds", type=float, default=0.3, help="Duration of the press, in s")
    parser.add_argument("--dt", type=float, default=1e-2, help="Control timestep, in s")
    parser.add_argument("--substeps", type=int, default=10, help="Number of substeps per control step")
    parser.add_argument("--precision", type=str, default="32", help="Floating-point precision, 32 or 64")
    parser.add_argument("--pcg-tolerance", type=float, default=1e-4, help="Relative tolerance of the linear solves")
    parser.add_argument(
        "--pcg-velocity-tolerance", type=float, default=1e-4, help="Velocity tolerance of the linear solves, in m/s"
    )
    parser.add_argument("--contact-stiffness", type=float, default=25.0, help="Stiffness ratio of the plate contacts")
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.seconds = 0.05

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision=args.precision, logging_level="warning")

    # A low-resolution hollow egg: an icosphere whose cross-section tapers from the blunt to the pointed end
    EGG_LENGTH, EGG_WIDTH, THICKNESS = 0.06, 0.045, 4e-4
    sphere = trimesh.creation.icosphere(subdivisions=2)
    egg_verts = sphere.vertices * 0.5 * np.array([EGG_LENGTH, EGG_WIDTH, EGG_WIDTH])
    egg_verts[:, 1:] *= 1.0 - 0.12 * sphere.vertices[:, :1]
    os.makedirs("out", exist_ok=True)
    egg_path = os.path.join("out", "shell_egg_press_egg.obj")
    trimesh.Trimesh(egg_verts, sphere.faces, process=False).export(egg_path)

    # Two plates on sliders squeeze the egg without gravity, starting 0.1 mm away from the outer surface of its shell
    plates_gap = np.abs(egg_verts[:, 1]).max() + 0.5 * THICKNESS + 1e-4
    mjcf = ET.Element("mujoco", model="press")
    worldbody = ET.SubElement(mjcf, "worldbody")
    for name, side in (("left", -1.0), ("right", 1.0)):
        body = ET.SubElement(worldbody, "body", name=name, pos=f"0 {side * (plates_gap + 0.005)} 0")
        ET.SubElement(body, "joint", name=f"{name}_slide", type="slide", axis="0 1 0")
        ET.SubElement(body, "geom", type="box", size="0.04 0.005 0.04", density="2000")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0.0, 0.0, 0.0),
        ),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=args.pcg_tolerance,
            pcg_velocity_tolerance=args.pcg_velocity_tolerance,
            contact_stiffness=args.contact_stiffness,
        ),
        show_viewer=False,
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=egg_path,
        ),
        material=gs.materials.Shell(
            rho=2e4,
            E=1e10,
            nu=0.3,
            thickness=THICKNESS,
            damping=1e-4,
            tensile_strength=1e7,
            fracture=False,
        ),
    )
    plates = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mjcf, encoding="unicode"),
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
        ),
    )
    scene.build()
    plates.set_dofs_kp([2e5, 2e5])
    plates.set_dofs_kv([2e3, 2e3])
    links_idx = [plates.get_link("left").idx_local, plates.get_link("right").idx_local]

    # The plates close at 2 mm/s each. The compression counts from the first contact, the force is the mean of the
    # forces on the two plates.
    compressions, forces, damages = [], [], []
    for i_step in range(int(args.seconds / args.dt)):
        target = 2e-3 * (i_step + 1) * args.dt
        plates.control_dofs_position([target, -target])
        scene.step()
        slides = tensor_to_array(plates.get_dofs_position())
        plates_force = tensor_to_array(plates.get_links_net_contact_force())[links_idx, 1]
        compressions.append(slides[0] - slides[1] - 2e-4)
        forces.append(0.5 * (plates_force[1] - plates_force[0]))
        damages.append(tensor_to_array(egg.get_peak_damage()))
        if i_step % 5 == 4:
            print(
                f"compression {1e6 * compressions[-1]:8.2f} um, force {forces[-1]:7.3f} N, "
                f"peak damage index {damages[-1]:.4f}"
            )

    compressions, forces, damages = np.array(compressions), np.array(forces), np.array(damages)
    print(
        f"force at 100 um {np.interp(1e-4, compressions, forces):.3f} N, "
        f"at 200 um {np.interp(2e-4, compressions, forces):.3f} N"
    )
    print(f"peak damage index at 100 um {np.interp(1e-4, compressions, damages):.4f}")
    if damages.max() >= 1.0:
        i_failure = np.argmax(damages >= 1.0)
        compression_failure = np.interp(
            1.0, damages[i_failure - 1 : i_failure + 1], compressions[i_failure - 1 : i_failure + 1]
        )
        print(f"first failure at {1e6 * compression_failure:.2f} um, face {tensor_to_array(egg.get_failure_face())}")


if __name__ == "__main__":
    main()
