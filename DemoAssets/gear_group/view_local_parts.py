"""
Preview all raw OBJ parts in Polyscope (no transform applied).
Vertices are as-is from the USD, which already encode assembly positions.

Usage:  python view_local_parts.py
"""

from pathlib import Path
import json
import numpy as np
import polyscope as ps

PARTS_DIR = Path(__file__).parent / "assets"
JSON_PATH = PARTS_DIR / "transforms.json"

COLORS = [
    (0.85, 0.32, 0.30),
    (0.30, 0.68, 0.85),
    (0.40, 0.82, 0.45),
    (0.92, 0.72, 0.20),
    (0.65, 0.40, 0.85),
    (0.90, 0.55, 0.25),
    (0.35, 0.80, 0.75),
    (0.80, 0.45, 0.65),
    (0.55, 0.55, 0.55),
]


def load_obj(filepath: Path):
    verts, faces = [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    part_names = [k for k in meta if k != "_meta"]

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)

    print(f"Loading {len(part_names)} parts ...\n")
    print(f"{'#':<4} {'Name':<20} {'Verts':>8} {'Tris':>8}")
    print("-" * 44)

    for i, name in enumerate(part_names):
        obj_file = PARTS_DIR / meta[name]["obj"]
        verts, faces = load_obj(obj_file)
        color = COLORS[i % len(COLORS)]

        sm = ps.register_surface_mesh(name, verts, faces)
        sm.set_color(color)
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)

        print(f"{i:<4} {name:<20} {len(verts):>8} {len(faces):>8}")

    ps.reset_camera_to_home_view()
    print("\nPolyscope window open. Close to exit.")
    ps.show()


if __name__ == "__main__":
    main()
