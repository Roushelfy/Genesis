"""Generate stitch pairs using the wearing init-shape_fixed meshes (simulation initial state).

Uses uipc's closest_vertex_triangle_pairs for robust and fast computation.

Usage::

    uv run python generate_stitch_at_frame0.py                # compute + GUI
    uv run python generate_stitch_at_frame0.py --no-gui        # compute + save only
    uv run python generate_stitch_at_frame0.py --distance 0.01 # custom distance
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polyscope as ps

from uipc.geometry import SimplicialComplexIO, closest_vertex_triangle_pairs

REPO_ROOT = Path(__file__).resolve().parents[3]
KIMONO_DIR = REPO_ROOT / "DemoAssets" / "kimono"

STITCH_PAIRS: list[tuple[str, str]] = [
    ("kimono_inner_lower", "kimono_inner_upper"),
    ("kimono_belt", "kimono_inner_upper"),
]

DEFAULT_MAX_DISTANCE = 0.002


def _cloth_piece_original_name(piece_name: str) -> str:
    s = re.sub(r"^wearing_", "", piece_name)
    s = re.sub(r"_\d+$", "", s)
    return s


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def load_wearing_meshes_uipc():
    """Load wearing init-shape_fixed meshes as UIPC SimplicialComplex, keyed by original cloth name."""
    io = SimplicialComplexIO()
    meshes = {}
    for p in sorted(KIMONO_DIR.glob("wearing_*_init-shape_fixed.obj")):
        stem = p.stem
        suffix = "_init-shape_fixed"
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        orig = _cloth_piece_original_name(base)
        sc = io.read(str(p))
        meshes[orig] = sc
        print(f"[mesh] {orig}: {p.name}")
    return meshes


def load_wearing_meshes_numpy():
    """Load wearing init-shape_fixed meshes as numpy arrays for Polyscope visualisation."""
    meshes = {}
    for p in sorted(KIMONO_DIR.glob("wearing_*_init-shape_fixed.obj")):
        stem = p.stem
        suffix = "_init-shape_fixed"
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        orig = _cloth_piece_original_name(base)
        v, f = _parse_obj(p)
        meshes[orig] = (v, f)
    return meshes


def save_stitch_json(mesh_a, mesh_b, pair_ids, max_distance):
    out_path = KIMONO_DIR / f"stitch_{mesh_a}_{mesh_b}.json"
    payload = {
        "mesh_a": mesh_a,
        "mesh_b": mesh_b,
        "mesh_a_obj": f"{mesh_a}.obj",
        "mesh_b_obj": f"{mesh_b}.obj",
        "max_distance": max_distance,
        "pairs": pair_ids.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[save] {out_path}  ({pair_ids.shape[0]} pairs)")


def _closest_on_tri(pt, v0, v1, v2):
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
        return v0 + d1 / (d1 - d3) * ab
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return v0 + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return v1 + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (v2 - v1)
    denom = 1.0 / (va + vb + vc)
    return v0 + (vb * denom) * ab + (vc * denom) * ac


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--distance", type=float, default=DEFAULT_MAX_DISTANCE)
    return p.parse_args()


def main():
    args = parse_args()
    max_dist = args.distance

    sc_meshes = load_wearing_meshes_uipc()

    all_pairs: dict[tuple[str, str], np.ndarray] = {}
    for name_a, name_b in STITCH_PAIRS:
        if name_a not in sc_meshes or name_b not in sc_meshes:
            print(f"[stitch] skip {name_a}<->{name_b}: mesh not found")
            continue
        print(f"[stitch] computing {name_a} <-> {name_b} (max_dist={max_dist}) ...")
        pairs_geo = closest_vertex_triangle_pairs(sc_meshes[name_a], sc_meshes[name_b], max_distance=max_dist)
        topo = pairs_geo.to_json()["instances"]["topo"]["attribute"]["values"]
        pairs_np = np.array([[v[0][0], v[1][0]] for v in topo], dtype=np.int32)
        all_pairs[(name_a, name_b)] = pairs_np
        print(f"[stitch] {name_a} <-> {name_b}: {pairs_np.shape[0]} pairs")
        save_stitch_json(name_a, name_b, pairs_np, max_dist)

    if args.no_gui:
        return

    np_meshes = load_wearing_meshes_numpy()

    ps.init()
    ps.set_up_dir("z_up")
    for name, (v, f) in np_meshes.items():
        sm = ps.register_surface_mesh(name, v, f)
        sm.set_transparency(0.5)

    for (na, nb), pairs_np in all_pairs.items():
        if pairs_np.shape[0] == 0:
            continue
        va, _ = np_meshes[na]
        vb, fb = np_meshes[nb]
        n = pairs_np.shape[0]
        nodes = np.zeros((2 * n, 3), dtype=np.float64)
        edges = np.zeros((n, 2), dtype=np.int32)
        dists = np.zeros(n, dtype=np.float64)
        for i in range(n):
            vid = pairs_np[i, 0]
            tid = pairs_np[i, 1]
            pt = va[vid]
            tri = fb[tid]
            proj = _closest_on_tri(pt, vb[tri[0]], vb[tri[1]], vb[tri[2]])
            nodes[2 * i] = pt
            nodes[2 * i + 1] = proj
            edges[i] = [2 * i, 2 * i + 1]
            dists[i] = np.linalg.norm(pt - proj)
        net = ps.register_curve_network(f"stitch_{na}_{nb}", nodes, edges, radius=0.0008)
        node_dist = np.zeros(2 * n)
        for i in range(n):
            node_dist[2 * i] = dists[i]
            node_dist[2 * i + 1] = dists[i]
        net.add_scalar_quantity("distance", node_dist, defined_on="nodes", enabled=True)
        print(f"[vis] stitch_{na}_{nb}: dist min={dists.min():.6f} max={dists.max():.6f}")

    ps.show()


if __name__ == "__main__":
    main()
