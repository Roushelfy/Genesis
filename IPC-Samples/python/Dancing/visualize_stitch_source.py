"""Visualise the original meshes (DemoAssets/kimono) used to generate stitches,
with stitch lines drawn as vertex-to-projection-point connections.

Usage::

    uv run python visualize_stitch_source.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polyscope as ps

REPO_ROOT = Path(__file__).resolve().parents[3]
KIMONO_SRC_DIR = REPO_ROOT / "DemoAssets" / "kimono"
STITCH_DIR = REPO_ROOT / "IPC-Samples" / "python" / "Wearing" / "results" / "kimono_v0"


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def closest_on_tri(pt: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    ab = v1 - v0
    ac = v2 - v0
    ap = pt - v0
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return v0.copy()
    bp = pt - v1
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return v1.copy()
    cp = pt - v2
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return v2.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        w = d1 / (d1 - d3)
        return v0 + w * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return v0 + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return v1 + w * (v2 - v1)
    denom = 1.0 / (va + vb + vc)
    sv = vb * denom
    sw = vc * denom
    return v0 + sv * ab + sw * ac


def main() -> None:
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in sorted(KIMONO_SRC_DIR.glob("*.obj")):
        v, f = parse_obj(p)
        meshes[p.stem] = (v, f)
        print(f"[mesh] {p.stem}: {v.shape[0]} verts, {f.shape[0]} faces")

    ps.init()
    ps.set_up_dir("z_up")

    for name, (v, f) in meshes.items():
        sm = ps.register_surface_mesh(name, v, f)
        sm.set_transparency(0.5)

    for stitch_path in sorted(STITCH_DIR.glob("stitch_*.json")):
        cfg = json.loads(stitch_path.read_text(encoding="utf-8"))
        mesh_a_name = cfg["mesh_a"]
        mesh_b_name = cfg["mesh_b"]
        pairs = np.asarray(cfg["pairs"], dtype=np.int32)
        if pairs.size == 0:
            continue
        if mesh_a_name not in meshes or mesh_b_name not in meshes:
            print(f"[stitch] skip {stitch_path.name}: mesh not found")
            continue

        verts_a = meshes[mesh_a_name][0]
        verts_b, faces_b = meshes[mesh_b_name]
        n = pairs.shape[0]
        nodes = np.zeros((2 * n, 3), dtype=np.float64)
        edges = np.zeros((n, 2), dtype=np.int32)
        dists = np.zeros(n, dtype=np.float64)

        for i in range(n):
            vid_a = pairs[i, 0]
            tid_b = pairs[i, 1]
            pt = verts_a[vid_a]
            tri = faces_b[tid_b]
            proj = closest_on_tri(pt, verts_b[tri[0]], verts_b[tri[1]], verts_b[tri[2]])
            nodes[2 * i] = pt
            nodes[2 * i + 1] = proj
            edges[i] = [2 * i, 2 * i + 1]
            dists[i] = np.linalg.norm(pt - proj)

        net_name = f"stitch_{mesh_a_name}_{mesh_b_name}"
        cn = ps.register_curve_network(net_name, nodes, edges, radius=0.0008)
        edge_dist = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            edge_dist[2 * i] = dists[i]
            edge_dist[2 * i + 1] = dists[i]
        cn.add_scalar_quantity("distance", edge_dist, defined_on="nodes", enabled=True)
        print(
            f"[stitch] {net_name}: {n} pairs, dist min={dists.min():.6f} max={dists.max():.6f} mean={dists.mean():.6f}"
        )

    ps.show()


if __name__ == "__main__":
    main()
