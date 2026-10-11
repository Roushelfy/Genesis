import json
from pathlib import Path

import numpy as np

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
output = root / "source-edge-v1"
output.mkdir(exist_ok=True)
with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as saved:
    vertices, faces = saved["vertices"], saved["surface_triangles"]
records = []
with np.load(root / "online-mesh64-v2/results/failure/snapshot-113.npz") as saved:
    for env, contact in zip(*np.nonzero(saved["raw_status"] == 12)):
        witness = saved["witness"][env, contact]
        distances, anchors, barycentric = [], [], []
        for triangle in vertices[faces]:
            origin = triangle[0]
            edges = (triangle[1:] - origin).T
            local = np.linalg.solve(edges.T @ edges, edges.T @ (witness - origin))
            bary = np.r_[1 - local.sum(), local]
            point = origin + edges @ local
            if min(bary) < 0:
                distance = np.inf
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    edge = triangle[b] - triangle[a]
                    t = np.clip(edge @ (witness - triangle[a]) / (edge @ edge), 0, 1)
                    candidate = triangle[a] + t * edge
                    current = np.linalg.norm(candidate - witness)
                    if current < distance:
                        distance, point = current, candidate
                        bary = np.zeros(3)
                        bary[a], bary[b] = 1 - t, t
            distances.append(np.linalg.norm(point - witness))
            anchors.append(point)
            barycentric.append(bary)
        distances, anchors, barycentric = map(np.array, (distances, anchors, barycentric))
        nearest = np.argmin(distances)
        tolerance = 32 * np.finfo(float).eps * max(np.linalg.norm(witness), 1e-4)
        candidates = np.nonzero(distances <= distances.min() + tolerance)[0]
        records.append({
            "env": int(env), "contact": int(contact), "witness": witness.tolist(),
            "position": saved["position"][env, contact].tolist(), "method": int(saved["method"][env, contact]),
            "force": saved["force"][env, contact].tolist(), "radius": float(saved["radius"][env, contact]),
            "mu": float(saved["friction"][env, contact]), "geom": int(saved["geom"][env, contact]),
            "pair": int(saved["pair"][env, contact]), "nearest": int(nearest), "tolerance": tolerance,
            "candidates": [{"face": int(face), "vertices": faces[face].tolist(), "barycentric": barycentric[face].tolist(), "distance": float(distances[face]), "anchor": anchors[face].tolist(), "separation": float(np.linalg.norm(anchors[face] - anchors[nearest]))} for face in candidates],
        })
with np.load(root / "online-mesh64-v2/results/source-vertices.npz") as saved:
    positions = saved["positions"]
    offset = saved["offsets"]
    geometry = {"maximum_offset": float(offset.max()), "status_counts": np.unique(saved["status"], return_counts=True)[1].tolist()}
(output / "audit.json").write_text(json.dumps({"contacts": records, "geometry": geometry}, indent=2) + "\n")
for record in records:
    print(json.dumps(record), flush=True)
print(json.dumps(geometry), flush=True)
