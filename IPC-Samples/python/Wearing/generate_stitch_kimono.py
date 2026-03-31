"""
Generate vertex-triangle stitch pairs between kimono cloth pieces.

Edit STITCH_PAIRS below to control which cloth combinations are stitched.
The script finds closest vertex-triangle correspondences within MAX_STITCH_DISTANCE,
saves them to JSON, and visualises the pairs in Polyscope as line segments.

Run:
    python generate_stitch_kimono.py            # GUI mode
    python generate_stitch_kimono.py --no-gui    # compute + save only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

STITCH_PAIRS: list[tuple[str, str]] = [
    ("kimono_belt", "kimono_inner_lower"),
    ("kimono_belt", "kimono_inner_upper"),
]

MAX_STITCH_DISTANCE: float = 0.01  # metres

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
KIMONO_DIR = REPO_ROOT / "DemoAssets" / "kimono_v0"
OUTPUT_DIR = THIS_DIR / "results" / "kimono_v0"

# ---------------------------------------------------------------------------
# Mesh loading (pure numpy – no uipc dependency for the generator)
# ---------------------------------------------------------------------------


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3 float64, faces Mx3 int32) from a .obj file."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_kimono_meshes(
    kimono_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load every .obj under *kimono_dir*.

    Returns ``{stem_name: (vertices_Nx3, faces_Mx3)}``.
    """
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in sorted(kimono_dir.glob("*.obj")):
        verts, faces = _parse_obj(p)
        meshes[p.stem] = (verts, faces)
    if not meshes:
        raise FileNotFoundError(f"No .obj files found in {kimono_dir}")
    return meshes


# ---------------------------------------------------------------------------
# Closest-point-on-triangle (vectorised numpy)
# ---------------------------------------------------------------------------


def _closest_points_on_triangles(
    points: np.ndarray,
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
) -> np.ndarray:
    """For each row in *points* return the closest point on the corresponding triangle.

    All inputs are (N, 3).  Implements Ericson, *Real-Time Collision Detection* §5.1.5.
    """
    ab = tri_v1 - tri_v0
    ac = tri_v2 - tri_v0
    ap = points - tri_v0

    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)

    bp = points - tri_v1
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)

    cp = points - tri_v2
    d5 = np.sum(ab * cp, axis=1)
    d6 = np.sum(ac * cp, axis=1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    n = len(points)
    closest = np.empty((n, 3), dtype=np.float64)
    assigned = np.zeros(n, dtype=bool)

    # Region A (vertex v0)
    mask = (~assigned) & (d1 <= 0) & (d2 <= 0)
    closest[mask] = tri_v0[mask]
    assigned |= mask

    # Region B (vertex v1)
    mask = (~assigned) & (d3 >= 0) & (d4 <= d3)
    closest[mask] = tri_v1[mask]
    assigned |= mask

    # Region C (vertex v2)
    mask = (~assigned) & (d6 >= 0) & (d5 <= d6)
    closest[mask] = tri_v2[mask]
    assigned |= mask

    # Region AB (edge v0-v1)
    mask = (~assigned) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    w = np.where(mask, d1 / np.maximum(d1 - d3, 1e-30), 0.0)
    closest[mask] = (tri_v0 + w[:, None] * ab)[mask]
    assigned |= mask

    # Region AC (edge v0-v2)
    mask = (~assigned) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = np.where(mask, d2 / np.maximum(d2 - d6, 1e-30), 0.0)
    closest[mask] = (tri_v0 + w[:, None] * ac)[mask]
    assigned |= mask

    # Region BC (edge v1-v2)
    mask = (~assigned) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w = np.where(mask, (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30), 0.0)
    closest[mask] = (tri_v1 + w[:, None] * (tri_v2 - tri_v1))[mask]
    assigned |= mask

    # Interior of triangle
    mask = ~assigned
    denom = np.where(mask, 1.0 / np.maximum(va + vb + vc, 1e-30), 0.0)
    sv = vb * denom
    sw = vc * denom
    closest[mask] = (tri_v0 + sv[:, None] * ab + sw[:, None] * ac)[mask]

    return closest


def compute_vt_pairs(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    faces_b: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find closest vertex-triangle pairs between mesh A vertices and mesh B triangles.

    Returns
    -------
    pair_ids : (K, 2) int32 – ``[vertex_id_in_A, triangle_id_in_B]``
    distances : (K,) float64
    closest_pts : (K, 3) float64 – closest point on each matched triangle
    """
    n_verts = verts_a.shape[0]
    n_tris = faces_b.shape[0]
    if n_verts == 0 or n_tris == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    tv0 = verts_b[faces_b[:, 0]]
    tv1 = verts_b[faces_b[:, 1]]
    tv2 = verts_b[faces_b[:, 2]]

    tri_centers = (tv0 + tv1 + tv2) / 3.0
    tri_radii = np.max(
        np.stack(
            [
                np.linalg.norm(tv0 - tri_centers, axis=1),
                np.linalg.norm(tv1 - tri_centers, axis=1),
                np.linalg.norm(tv2 - tri_centers, axis=1),
            ],
            axis=0,
        ),
        axis=0,
    )

    best_tid = np.full(n_verts, -1, dtype=np.int32)
    best_dist = np.full(n_verts, np.inf, dtype=np.float64)
    best_cp = np.zeros((n_verts, 3), dtype=np.float64)

    batch_size = max(1, min(n_tris, 512))
    for t_start in range(0, n_tris, batch_size):
        t_end = min(t_start + batch_size, n_tris)
        t_idx = np.arange(t_start, t_end)

        for vi in range(n_verts):
            pt = verts_a[vi]
            rough_dist = np.linalg.norm(tri_centers[t_idx] - pt, axis=1) - tri_radii[t_idx]
            candidate_mask = rough_dist < min(max_distance, best_dist[vi])
            if not np.any(candidate_mask):
                continue
            cand_idx = t_idx[candidate_mask]
            pts_rep = np.tile(pt, (len(cand_idx), 1))
            cps = _closest_points_on_triangles(
                pts_rep,
                tv0[cand_idx],
                tv1[cand_idx],
                tv2[cand_idx],
            )
            dists = np.linalg.norm(cps - pts_rep, axis=1)
            local_best = np.argmin(dists)
            if dists[local_best] < best_dist[vi]:
                best_dist[vi] = dists[local_best]
                best_tid[vi] = cand_idx[local_best]
                best_cp[vi] = cps[local_best]

    keep = (best_tid >= 0) & (best_dist <= max_distance)
    vid_keep = np.nonzero(keep)[0].astype(np.int32)
    pair_ids = np.stack([vid_keep, best_tid[keep]], axis=1)
    return pair_ids, best_dist[keep], best_cp[keep]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def save_stitch_json(
    mesh_a_name: str,
    mesh_b_name: str,
    pair_ids: np.ndarray,
    max_distance: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"stitch_{mesh_a_name}_{mesh_b_name}.json"
    payload = {
        "mesh_a": mesh_a_name,
        "mesh_b": mesh_b_name,
        "mesh_a_obj": f"{mesh_a_name}.obj",
        "mesh_b_obj": f"{mesh_b_name}.obj",
        "max_distance": max_distance,
        "pairs": pair_ids.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Pair computation driver
# ---------------------------------------------------------------------------


def compute_all_pairs(
    meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    stitch_pairs: list[tuple[str, str]],
    max_distance: float,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``{(name_a, name_b): (pair_ids, distances, closest_pts, verts_a, verts_b)}``."""
    results: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name_a, name_b in stitch_pairs:
        if name_a not in meshes:
            raise KeyError(f"Mesh '{name_a}' not found. Available: {list(meshes.keys())}")
        if name_b not in meshes:
            raise KeyError(f"Mesh '{name_b}' not found. Available: {list(meshes.keys())}")
        verts_a, _ = meshes[name_a]
        verts_b, faces_b = meshes[name_b]
        pair_ids, dists, cps = compute_vt_pairs(verts_a, verts_b, faces_b, max_distance)
        results[(name_a, name_b)] = (pair_ids, dists, cps, verts_a, verts_b)
    return results


# ---------------------------------------------------------------------------
# Polyscope visualisation
# ---------------------------------------------------------------------------


def _register_meshes(meshes: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for name, (verts, faces) in meshes.items():
        sm = ps.register_surface_mesh(name, verts, faces)
        sm.set_transparency(0.5)


def _register_stitch_lines(
    all_results: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, object]:
    nets: dict[str, object] = {}
    for (name_a, name_b), (pair_ids, _dists, cps, verts_a, _verts_b) in all_results.items():
        net_name = f"stitch_{name_a}_{name_b}"
        n_pairs = pair_ids.shape[0]
        if n_pairs == 0:
            continue
        nodes = np.zeros((2 * n_pairs, 3), dtype=np.float64)
        edges = np.zeros((n_pairs, 2), dtype=np.int32)
        for i in range(n_pairs):
            vid = pair_ids[i, 0]
            nodes[2 * i] = verts_a[vid]
            nodes[2 * i + 1] = cps[i]
            edges[i] = [2 * i, 2 * i + 1]
        nets[net_name] = ps.register_curve_network(net_name, nodes, edges, radius=0.0005)
    return nets


def run_gui(
    meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    stitch_pairs: list[tuple[str, str]],
    max_distance: float,
    output_dir: Path,
) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    _register_meshes(meshes)

    state: dict = {
        "max_distance": max_distance,
        "results": compute_all_pairs(meshes, stitch_pairs, max_distance),
    }
    nets = _register_stitch_lines(state["results"])

    def on_update() -> None:
        changed, new_val = imgui.SliderFloat("Max Distance", state["max_distance"], 0.0001, 0.1)
        if changed:
            state["max_distance"] = new_val

        if imgui.Button("Recompute Pairs"):
            for net_name in nets:
                ps.remove_curve_network(net_name)
            nets.clear()
            state["results"] = compute_all_pairs(meshes, stitch_pairs, state["max_distance"])
            nets.update(_register_stitch_lines(state["results"]))

        if imgui.Button("Save All Stitch JSON"):
            for (na, nb), (pids, _d, _c, _va, _vb) in state["results"].items():
                p = save_stitch_json(na, nb, pids, state["max_distance"], output_dir)
                print(f"[save] {p}  ({pids.shape[0]} pairs)")

        imgui.Separator()
        for (na, nb), (pids, _d, _c, _va, _vb) in state["results"].items():
            imgui.Text(f"{na} <-> {nb}: {pids.shape[0]} pairs")

    ps.set_user_callback(on_update)
    ps.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate kimono stitch VT pairs.")
    parser.add_argument("--no-gui", action="store_true", help="Compute and save without GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meshes = load_kimono_meshes(KIMONO_DIR)
    print(f"Loaded {len(meshes)} meshes: {list(meshes.keys())}")

    if args.no_gui:
        results = compute_all_pairs(meshes, STITCH_PAIRS, MAX_STITCH_DISTANCE)
        for (na, nb), (pids, _d, _c, _va, _vb) in results.items():
            p = save_stitch_json(na, nb, pids, MAX_STITCH_DISTANCE, OUTPUT_DIR)
            print(f"[save] {p}  ({pids.shape[0]} pairs)")
        return

    run_gui(meshes, STITCH_PAIRS, MAX_STITCH_DISTANCE, OUTPUT_DIR)


if __name__ == "__main__":
    main()
