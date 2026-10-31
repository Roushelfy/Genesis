"""Inspect captured failure inputs on a compute node without repeating the scene."""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.rigid.stress_mesh_asset import load_surface
from research.rigid_stress.qipc_comparison.artifacts import write_json


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.root.rglob("failure-state.npz"):
        meta = json.loads((path.parent / "progress.json").read_text())
        collision = load_surface(Path(meta["collision"]["path"]))
        with np.load(path) as arrays:
            valid = arrays["stress.0.contacts.valid"][0].astype(bool)
            positions = arrays["stress.0.contacts.position"][0][valid]
            nearest = np.argmin(np.linalg.norm(positions[:, None] - collision.vertices[None], axis=-1), axis=1)
            tensor = arrays["stress.curvature.vertices_tensor"][-len(collision.vertices) :]
            row = {
                "source": str(path),
                "failure": json.loads((path.parent / "failure.json").read_text()),
                "contacts": {
                    name: arrays["stress.0.contacts." + name][0][valid].tolist()
                    for name in (
                        "position",
                        "normal",
                        "force",
                        "radius",
                        "radius_status",
                        "status",
                        "force_error",
                        "moment_error",
                    )
                },
                "nearest_collision_vertices": nearest.tolist(),
                "nearest_vertex_positions": collision.vertices[nearest].tolist(),
                "nearest_vertex_curvature_eigenvalues": np.linalg.eigvalsh(tensor[nearest]).tolist(),
                "outer_vertex_count": len(collision.vertices) // 2,
            }
            row["other_states"] = {
                name: arrays[name].tolist() for name in arrays.files if name.startswith("stress.0.state.")
            }
            rows.append(row)
    write_json(args.output / "inspection.json", finite_json(rows))


if __name__ == "__main__":
    main()
