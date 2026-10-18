"""Read the actual PR22 first-fan thickness/contact geometry without advancing physics."""

import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from ..shell_comparison.application import ShellEgg
from ..shell_comparison.rollout import make_parser
from .artifacts import file_id, runtime_identity, write_json


def main():
    args = make_parser().parse_args()
    assert args.model == "shell" and args.envs == 1
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    workload = ShellEgg(args)
    topology = workload.egg.topology
    # The pinned coupler selects the FIRST fan corner, rather than the union
    # or average of all incident face thicknesses. Read its exact source data.
    contact_face = topology.fans_corner[topology.verts_fan_start] // 3
    thickness = qd_to_numpy(workload.scene.sim.shell_solver.shell_state.faces_thickness, transpose=True)[0]
    softness = qd_to_numpy(workload.scene.rigid_solver.dyn_info.geoms.coup_softness, transpose=True)
    np.testing.assert_allclose(softness, args.coupling_softness, rtol=1e-12, atol=0)
    np.savez_compressed(
        args.output / "layout.npz",
        vertices_m=workload.egg.init_verts - [0.65, 0, 0.031],
        faces=workload.egg.init_faces,
        collision_vertex_face=contact_face,
        face_thickness_m=thickness,
        face_mass_kg=topology.faces_mass,
        rigid_geometry_coupling_softness_m=softness,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-PR22-contact-geometry-layout",
            "identity": runtime_identity(),
            "layout": file_id(args.output / "layout.npz"),
            "configured_softness_m": args.coupling_softness,
            "influence_0_1_cutoff_beyond_half_thickness_m": (args.coupling_softness * np.log(10)).item(),
            "wall_input": workload.wall_input.description() if workload.wall_input is not None else None,
            "scope": "Actual constructed PR22 public topology, first incident fan face, thickness and rigid-geometry softness. No physical step or fitted force/stress. The velocity-projection law remains different from an IPC barrier even when influence cutoff lengths match.",
        },
    )


if __name__ == "__main__":
    main()
