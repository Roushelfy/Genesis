"""No-GUI version of debug_stitch_gui.py for headless debugging.

Runs 1 frame so CUDA printf output (F, position comparisons) is visible in terminal.

Usage::

    python debug_stitch_nogui.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uipc import Logger, Timer, view
from uipc.constitution import (
    ElasticModuli,
    SoftVertexTriangleStitch,
    Empty,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, label_surface

KEEP_PIECES = {"kimono_inner_lower", "kimono_inner_upper"}

REPO_ROOT = Path(__file__).resolve().parents[3]
KIMONO_DIR = REPO_ROOT / "IPC-Samples" / "python" / "Wearing" / "results" / "kimono_v0"
OUTPUT_DIR = REPO_ROOT / "IPC-Samples" / "output" / "python" / "Dancing" / "debug_stitch"


@dataclass
class ClothPiece:
    name: str
    init_obj: Path
    rest_obj: Path


def _cloth_piece_original_name(piece_name: str) -> str:
    s = re.sub(r"^wearing_", "", piece_name)
    s = re.sub(r"_\d+$", "", s)
    return s


def discover_pieces() -> list[ClothPiece]:
    pieces: list[ClothPiece] = []
    for init_path in sorted(KIMONO_DIR.glob("wearing_*_init-shape_fixed.obj")):
        stem = init_path.stem
        suffix = "_init-shape_fixed"
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        orig_name = _cloth_piece_original_name(base)
        if orig_name not in KEEP_PIECES:
            continue
        rest_path = KIMONO_DIR / f"{base}_rest-shape.obj"
        if not rest_path.exists():
            raise FileNotFoundError(f"Missing rest-shape: {rest_path.name}")
        pieces.append(ClothPiece(name=base, init_obj=init_path, rest_obj=rest_path))
    print(f"[debug] loaded {len(pieces)} pieces: {[p.name for p in pieces]}")
    return pieces


def main() -> None:
    Logger.set_level(Logger.Level.Info)
    Timer.enable_all()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = Engine("cuda", str(OUTPUT_DIR))
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 0.5
    config["linear_system"]["tol_rate"] = 1e-8
    config["newton"]["max_iter"] = 256
    config["extras"]["debug"]["dump_surface"] = True
    scene = Scene(config)
    scene.animator().substep(20)

    empty = Empty()
    cloth_io = SimplicialComplexIO()

    pieces = discover_pieces()
    cloth_slots: list[tuple[ClothPiece, object, object]] = []

    for idx, piece in enumerate(pieces):
        cloth_mesh = cloth_io.read(str(piece.init_obj))
        rest_source = cloth_io.read(str(piece.rest_obj))
        label_surface(cloth_mesh)
        empty.apply_to(cloth_mesh)

        rest_mesh = cloth_mesh.copy()
        rest_pos = np.asarray(view(rest_source.positions()), copy=True)
        view(rest_mesh.positions())[:] = rest_pos

        cloth_object = scene.objects().create(f"cloth_{idx}_{piece.name}")
        cloth_geo_slot, cloth_rest_geo_slot = cloth_object.geometries().create(cloth_mesh, rest_mesh)
        cloth_slots.append((piece, cloth_geo_slot, cloth_rest_geo_slot))

    # --- stitch ---
    slot_by_orig: dict[str, tuple[object, object]] = {}
    for piece, gs, rs in cloth_slots:
        slot_by_orig[_cloth_piece_original_name(piece.name)] = (gs, rs)

    stitch_file = KIMONO_DIR / "stitch_kimono_inner_lower_kimono_inner_upper.json"
    if stitch_file.exists():
        cfg = json.loads(stitch_file.read_text(encoding="utf-8"))
        stitch_pairs = np.asarray(cfg["pairs"], dtype=np.int32)
        mesh_a, mesh_b = cfg["mesh_a"], cfg["mesh_b"]
        if mesh_a in slot_by_orig and mesh_b in slot_by_orig:
            svts = SoftVertexTriangleStitch()
            stitch_obj = scene.objects().create("stitch")
            geo_a, rest_a = slot_by_orig[mesh_a]
            geo_b, rest_b = slot_by_orig[mesh_b]
            stitch_geo = svts.create_geometry(
                (geo_a, geo_b),
                (rest_a, rest_b),
                stitch_pairs,
                ElasticModuli.youngs_poisson(1e4, 0.498),
                min_separate_distance=0.00001,
            )
            stitch_obj.geometries().create(stitch_geo)
            print(f"[stitch] applied {mesh_a} <-> {mesh_b}: {stitch_pairs.shape[0]} pairs")
    else:
        print(f"[stitch] file not found: {stitch_file}")

    world.init(scene)
    world.retrieve()

    print("\n[debug] === Running 1 frame ===")
    world.advance()
    world.retrieve()
    print(f"[debug] frame = {world.frame()}")
    print("[debug] === Done ===")


if __name__ == "__main__":
    main()
