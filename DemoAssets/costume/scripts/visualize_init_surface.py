"""Visualise the global surface at frame-0 (post world.init) with stitch lines.

Usage::

    uv run python visualize_init_surface.py --dataset 0
    uv run python visualize_init_surface.py --dataset 0 --no-lower
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polyscope as ps

from g1_kimono_batch import (
    build_player,
    build_scene,
    discover_kimono_pieces,
    get_paths,
    load_stitch_configs,
    apply_stitches,
    parse_args,
    _cloth_piece_original_name,
    variant_name,
)
from uipc import SceneIO, view
from uipc.core import World


def _parse_obj(path: str | Path):
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


def main() -> None:
    args = parse_args()
    skip_pieces: set[str] = {"kimono_inner_lower"} if args.no_lower else set()
    output_dir, urdf_path, npz_path, warmup_joint_json, kimono_dir, vname = get_paths(args.dataset, args.no_lower)

    pieces = discover_kimono_pieces(kimono_dir, skip_pieces)
    print(f"[vis] {len(pieces)} cloth pieces: {[p.name for p in pieces]}")

    engine, world, scene, cloth_contact, ground_contact, cloth_slots = build_scene(
        output_dir, pieces, substep_override=args.substep
    )

    stitch_configs = load_stitch_configs(kimono_dir)
    if stitch_configs:
        apply_stitches(scene, stitch_configs, cloth_slots)

    player = build_player(scene, urdf_path, npz_path, warmup_joint_json, cloth_contact, ground_contact)

    root_tf = player.current_root_transform
    for _piece, cloth_geo_slot, cloth_rest_geo_slot in cloth_slots:
        view(cloth_geo_slot.geometry().transforms())[0] = root_tf
        view(cloth_rest_geo_slot.geometry().transforms())[0] = root_tf

    world.init(scene)
    world.retrieve()

    sio = SceneIO(scene)
    surface_path = output_dir / "surface_init.obj"
    sio.write_surface(str(surface_path))
    print(f"[vis] wrote global surface -> {surface_path}")

    # --- Polyscope ---
    ps.init()
    ps.set_up_dir("z_up")

    surf_v, surf_f = _parse_obj(surface_path)
    ps.register_surface_mesh("global_surface", surf_v, surf_f, transparency=0.4)

    for piece, geo_slot, _rest in cloth_slots:
        pos = np.asarray(view(geo_slot.geometry().positions()), copy=True).squeeze()
        orig = _cloth_piece_original_name(piece.name)
        ps.register_point_cloud(f"cloth_{orig}", pos, radius=0.001)

    for cfg in stitch_configs:
        mesh_a_name = cfg["mesh_a"]
        mesh_b_name = cfg["mesh_b"]
        pairs = np.asarray(cfg["pairs"], dtype=np.int32)
        if pairs.size == 0:
            continue

        slot_a = slot_b = None
        for piece, geo_slot, _ in cloth_slots:
            orig = _cloth_piece_original_name(piece.name)
            if orig == mesh_a_name:
                slot_a = geo_slot
            if orig == mesh_b_name:
                slot_b = geo_slot
        if slot_a is None or slot_b is None:
            print(f"[vis] skipping stitch {mesh_a_name}<->{mesh_b_name} (mesh not loaded)")
            continue

        pos_a = np.asarray(view(slot_a.geometry().positions()), copy=True).squeeze()
        pos_b = np.asarray(view(slot_b.geometry().positions()), copy=True).squeeze()
        faces_b_geo = np.asarray(view(slot_b.geometry().triangles()), copy=True).squeeze()

        n = pairs.shape[0]
        nodes = np.zeros((2 * n, 3), dtype=np.float64)
        edges = np.zeros((n, 2), dtype=np.int32)
        for i in range(n):
            vid_a = pairs[i, 0]
            tid_b = pairs[i, 1]
            nodes[2 * i] = pos_a[vid_a]
            tri = faces_b_geo[tid_b]
            nodes[2 * i + 1] = (pos_b[tri[0]] + pos_b[tri[1]] + pos_b[tri[2]]) / 3.0
            edges[i] = [2 * i, 2 * i + 1]

        net_name = f"stitch_{mesh_a_name}_{mesh_b_name}"
        ps.register_curve_network(net_name, nodes, edges, radius=0.0008)
        print(f"[vis] registered stitch lines: {net_name} ({n} pairs)")

    ps.show()


if __name__ == "__main__":
    main()
