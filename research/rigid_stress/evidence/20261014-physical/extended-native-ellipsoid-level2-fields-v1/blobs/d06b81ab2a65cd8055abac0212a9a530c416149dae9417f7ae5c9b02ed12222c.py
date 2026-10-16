"""Record installed IPC collision constructors and force-magnitude methods before extending a reference."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    classes = [
        ipctk.VertexVertexNormalCollision,
        ipctk.EdgeVertexNormalCollision,
        ipctk.EdgeEdgeNormalCollision,
        ipctk.FaceVertexNormalCollision,
    ]
    result = {cls.__name__: cls.__init__.__doc__ for cls in classes}
    result["force_magnitude"] = ipctk.BarrierPotential.force_magnitude.__doc__
    result["compute_distance"] = ipctk.EdgeEdgeNormalCollision.compute_distance.__doc__
    observations = []
    for run in args.run:
        closed_source(run)
        static, _config = restore_static(run)
        paths = [run / "results/closure_state.npz", *sorted((run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))]
        for path in paths:
            with np.load(path) as saved:
                u = saved["displacement"].copy()
            normal, _current = static.normal_collisions(u)
            positions = static.contact_node_map @ (np.array(static.rest, dtype=np.longdouble) + u.reshape(-1, 3))
            rows = []
            for index in range(len(normal)):
                collision = normal[index]
                ids = np.array(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
                ids = ids[ids >= 0]
                centered = np.array(positions[ids] - positions[ids[0]], dtype=float).reshape(-1)
                groups = static.contact_surface_groups[ids]
                gradient = static.barrier.gradient(collision, centered).reshape(-1, 3)
                magnitude = static.barrier.force_magnitude(collision.compute_distance(centered), collision.dmin)
                resultant = np.linalg.norm(gradient[groups == 0].sum(axis=0)) / abs(collision.weight)
                rows.append(
                    {
                        "type": type(collision).__name__,
                        "vertex_ids": ids.tolist(),
                        "groups": groups.tolist(),
                        "weight": float(collision.weight),
                        "squared_distance_m2": float(collision.compute_distance(centered)),
                        "unmollified_friction_magnitude": float(magnitude),
                        "normal_body_resultant_over_weight": float(resultant),
                    }
                )
                del collision
            observations.append(
                {
                    "source_run": str(run),
                    "state": str(path),
                    "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "stencils": rows,
                }
            )
    result["actual_state_observations"] = observations
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
