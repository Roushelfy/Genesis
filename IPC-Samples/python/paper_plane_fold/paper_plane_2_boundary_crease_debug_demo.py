"""
Paper-plane boundary-to-boundary crease debug demo.

This example:
- loads a boundary-crease paper asset such as `paper_plane_2_fine.obj`, `paper_plane_3_coarse.obj`, `paper_plane_4_coarse.obj`, or `paper_plane_5_coarse.obj`,
- detects all interior straight crease chains whose two endpoints lie on the outer boundary,
- assigns stable crease and endpoint labels,
- keeps the step diagnostic-only by oscillating the detected crease chains.

Run:
    python python/examples/paper_plane_2_boundary_crease_debug_demo.py
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from uipc import Logger, Engine, World, Scene, SceneIO, Animation, view, builtin
    from uipc.geometry import SimplicialComplex, SimplicialComplexIO, label_surface
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
from asset_dir import AssetDir


DEFAULT_ASSET_FILENAME = "paper_plane_2_fine.obj"
MESH_PARTITION_SIZE = 16

SHELL_THICKNESS = 8.0e-4
SHELL_DENSITY = 110.0
SHELL_YOUNG = 2.5e6
SHELL_POISSON = 0.33
SHELL_BENDING_STIFFNESS = 1.2e4
SHELL_YIELD_THRESHOLD = 0.001
SHELL_HARDENING_MODULUS = 0.0
CONSTRAINT_STRENGTH = 80.0

OSCILLATION_CYCLES = 2.0
OSCILLATION_AMPLITUDE = 0.015
SETTLE_FRAMES = 12
OSCILLATION_FRAMES = 144
HOLD_FRAMES = 24
TOTAL_FRAMES = SETTLE_FRAMES + OSCILLATION_FRAMES + HOLD_FRAMES

BOUNDARY_TOL = 1.0e-6
LINE_SIGNATURE_DECIMALS = 6
MIN_LINE_SIGNATURE_DECIMALS = 3
FAMILY_ANGLE_TOL = 5.0
MIN_CREASE_VERTS = 5
MIN_CREASE_LENGTH_RATIO = 0.08
EXPECTED_CREASE_COUNT = 14


@dataclass(frozen=True)
class BoundaryCreaseChain:
    index: int
    crease_id: str
    crease_name: str
    family: str
    angle_deg: float
    vertex_ids: tuple[int, ...]
    endpoint_a: int
    endpoint_b: int
    endpoint_a_id: str
    endpoint_b_id: str
    endpoint_a_name: str
    endpoint_b_name: str
    endpoint_a_xyz: tuple[float, float, float]
    endpoint_b_xyz: tuple[float, float, float]
    endpoint_a_side: str
    endpoint_b_side: str
    length: float


def axis_name(axis: int) -> str:
    return "xyz"[axis]


def unit_2d(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-12:
        raise ValueError("zero-length direction")
    return vec / norm


def resolve_origami_asset_path(asset_filename: str | None = None) -> str:
    asset_filename = asset_filename or DEFAULT_ASSET_FILENAME
    if os.path.isabs(asset_filename):
        return asset_filename
    if os.path.exists(asset_filename):
        return asset_filename
    return os.path.join(AssetDir.asset_path(), "sim_data", "trimesh", asset_filename)


def load_paper_plane_mesh(asset_filename: str | None = None) -> SimplicialComplex:
    mesh_path = resolve_origami_asset_path(asset_filename)
    io = SimplicialComplexIO()
    mesh = io.read(mesh_path)
    label_surface(mesh)
    return mesh


def build_topology(mesh: SimplicialComplex) -> dict[str, object]:
    positions = np.array(view(mesh.positions()), copy=True).reshape(-1, 3)
    triangles = np.array(mesh.triangles().topo().view(), copy=True).reshape(-1, 3)

    adjacency: dict[int, set[int]] = defaultdict(set)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((u, v)))
            edge_counts[edge] += 1
            adjacency[u].add(v)
            adjacency[v].add(u)

    edges = np.array(sorted(edge_counts.keys()), dtype=np.int32)
    boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
    boundary_vertices = sorted({vid for edge in boundary_edges for vid in edge})

    boundary_adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        boundary_adj[a].append(b)
        boundary_adj[b].append(a)

    return {
        "positions": positions,
        "triangles": triangles,
        "edges": edges,
        "adjacency": adjacency,
        "boundary_edges": boundary_edges,
        "boundary_vertices": boundary_vertices,
        "boundary_adj": boundary_adj,
    }


def largest_boundary_component(boundary_adj: dict[int, list[int]]) -> list[int]:
    seen: set[int] = set()
    best_component: list[int] = []

    for start in boundary_adj:
        if start in seen:
            continue
        component: list[int] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for nb in boundary_adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(component) > len(best_component):
            best_component = component

    if not best_component:
        raise AssertionError("failed to find a boundary component")

    return best_component


def order_boundary_loop(component: list[int], boundary_adj: dict[int, list[int]]) -> list[int]:
    start = min(component)
    prev = -1
    cur = start
    ordered = [start]

    while True:
        candidates = [nb for nb in boundary_adj[cur] if nb != prev]
        if not candidates:
            raise AssertionError("boundary loop walk terminated early")

        nxt = candidates[0]
        if nxt == start:
            break

        ordered.append(nxt)
        prev, cur = cur, nxt
        if len(ordered) > len(component):
            raise AssertionError("boundary loop walk exceeded component size")

    if len(ordered) != len(component):
        raise AssertionError("boundary loop ordering missed vertices")

    return ordered


def detect_planar_frame(
    positions: np.ndarray,
    boundary_loop: list[int],
) -> tuple[tuple[int, int], int, tuple[float, float, float, float]]:
    boundary_positions = positions[boundary_loop]
    spans = boundary_positions.max(axis=0) - boundary_positions.min(axis=0)
    normal_axis = int(np.argmin(spans))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    plane_axes.sort(key=lambda axis: spans[axis])
    plane_axes = (plane_axes[0], plane_axes[1])

    boundary_planar = boundary_positions[:, plane_axes]
    bbox = (
        float(boundary_planar[:, 0].min()),
        float(boundary_planar[:, 0].max()),
        float(boundary_planar[:, 1].min()),
        float(boundary_planar[:, 1].max()),
    )
    return plane_axes, normal_axis, bbox


def classify_boundary_side(planar_pos: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
    xmin, xmax, ymin, ymax = bbox
    x = float(planar_pos[0])
    y = float(planar_pos[1])
    if abs(y - ymin) <= BOUNDARY_TOL:
        return "bottom"
    if abs(y - ymax) <= BOUNDARY_TOL:
        return "top"
    if abs(x - xmin) <= BOUNDARY_TOL:
        return "left"
    if abs(x - xmax) <= BOUNDARY_TOL:
        return "right"
    return "interior"


def line_signature(
    pa: np.ndarray,
    pb: np.ndarray,
    decimals: int | None = None,
) -> tuple[tuple[float, ...], np.ndarray, np.ndarray, float]:
    decimals = LINE_SIGNATURE_DECIMALS if decimals is None else int(decimals)
    direction = unit_2d(pb - pa)
    if direction[0] < -1.0e-12 or (abs(direction[0]) <= 1.0e-12 and direction[1] < 0.0):
        direction = -direction

    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    offset = float(np.dot(normal, pa))
    if offset < -1.0e-9:
        normal = -normal
        offset = -offset

    signature = tuple(
        float(value)
        for value in np.round(
            np.concatenate([direction, normal, np.array([offset], dtype=np.float64)]),
            decimals,
        )
    )
    return signature, direction, normal, offset


def collect_crease_candidates(
    topo: dict[str, object],
    plane_positions: np.ndarray,
    boundary_edges: set[tuple[int, int]],
    boundary_vertices: set[int],
    bbox: tuple[float, float, float, float],
    min_crease_length: float,
    signature_decimals: int,
) -> list[dict[str, object]]:
    line_edges: dict[tuple[float, ...], set[tuple[int, int]]] = defaultdict(set)
    line_meta: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, float]] = {}
    for edge in topo["edges"]:
        a = int(edge[0])
        b = int(edge[1])
        edge_key = tuple(sorted((a, b)))
        if edge_key in boundary_edges:
            continue

        signature, direction, normal, offset = line_signature(
            plane_positions[a],
            plane_positions[b],
            decimals=signature_decimals,
        )
        line_edges[signature].add(edge_key)
        if signature not in line_meta:
            line_meta[signature] = (direction, normal, offset)

    candidates: list[dict[str, object]] = []
    for signature, edges in line_edges.items():
        direction, normal, offset = line_meta[signature]

        line_vertices = sorted({vid for edge in edges for vid in edge})
        if len(line_vertices) < MIN_CREASE_VERTS:
            continue

        boundary_line_vertices = [
            vid
            for vid in line_vertices
            if vid in boundary_vertices
            and classify_boundary_side(plane_positions[vid], bbox) != "interior"
        ]
        if len(boundary_line_vertices) != 2:
            continue

        projections = {
            vid: float(np.dot(plane_positions[vid], direction))
            for vid in line_vertices
        }
        ordered_vertices = tuple(sorted(line_vertices, key=lambda vid: projections[vid]))

        endpoint_a = int(boundary_line_vertices[0])
        endpoint_b = int(boundary_line_vertices[1])
        if projections[endpoint_a] > projections[endpoint_b]:
            endpoint_a, endpoint_b = endpoint_b, endpoint_a

        endpoint_a_side = classify_boundary_side(plane_positions[endpoint_a], bbox)
        endpoint_b_side = classify_boundary_side(plane_positions[endpoint_b], bbox)
        if endpoint_a_side == endpoint_b_side:
            continue

        length = float(np.linalg.norm(plane_positions[endpoint_b] - plane_positions[endpoint_a]))
        if length < min_crease_length:
            continue

        angle_deg = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        family = family_from_angle(angle_deg)
        midpoint = (plane_positions[endpoint_a] + plane_positions[endpoint_b]) * 0.5

        candidates.append(
            {
                "signature": signature,
                "direction": direction,
                "normal": normal,
                "offset": offset,
                "angle_deg": angle_deg,
                "family": family,
                "vertex_ids": ordered_vertices,
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "endpoint_a_side": endpoint_a_side,
                "endpoint_b_side": endpoint_b_side,
                "pair_code": f"{side_code(endpoint_a_side)}{side_code(endpoint_b_side)}",
                "midpoint": midpoint,
                "length": length,
            }
        )
    return candidates


def connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []

    for start in adjacency:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    return components


def order_component_path(
    component: list[int],
    adjacency: dict[int, set[int]],
    plane_positions: np.ndarray,
    direction: np.ndarray,
) -> tuple[int, ...] | None:
    component_set = set(component)
    local_degree = {
        vid: sum(1 for nb in adjacency[vid] if nb in component_set)
        for vid in component
    }
    endpoints = [vid for vid, degree in local_degree.items() if degree == 1]
    if len(endpoints) != 2:
        return None

    projections = {vid: float(np.dot(plane_positions[vid], direction)) for vid in endpoints}
    start = min(endpoints, key=lambda vid: projections[vid])
    target = max(endpoints, key=lambda vid: projections[vid])

    ordered = [start]
    prev = -1
    current = start
    while current != target:
        candidates = [nb for nb in adjacency[current] if nb in component_set and nb != prev]
        if len(candidates) != 1:
            return None
        nxt = candidates[0]
        ordered.append(nxt)
        prev, current = current, nxt

        if len(ordered) > len(component):
            return None

    if len(ordered) != len(component):
        return None

    return tuple(ordered)


def family_from_angle(angle_deg: float) -> str:
    if abs(angle_deg) <= FAMILY_ANGLE_TOL:
        return "H"
    if abs(abs(angle_deg) - 90.0) <= FAMILY_ANGLE_TOL:
        return "V"
    return "DP" if angle_deg > 0.0 else "DN"


def side_code(side: str) -> str:
    codes = {
        "left": "L",
        "right": "R",
        "top": "T",
        "bottom": "B",
    }
    if side not in codes:
        raise AssertionError(f"unsupported boundary side {side}")
    return codes[side]


def group_member_sort_key(candidate: dict[str, object]) -> tuple[float, ...]:
    family = candidate["family"]
    midpoint = candidate["midpoint"]
    angle_deg = float(candidate["angle_deg"])

    if family == "V":
        return (float(midpoint[0]),)
    if family == "H":
        return (-float(midpoint[1]),)
    return (-float(midpoint[1]), float(midpoint[0]), abs(angle_deg))


def overall_chain_sort_key(candidate: dict[str, object]) -> tuple[float, ...]:
    family_bucket = {"V": 0, "H": 1, "DN": 2, "DP": 3}
    family = candidate["family"]
    midpoint = candidate["midpoint"]
    angle_deg = float(candidate["angle_deg"])

    if family == "V":
        local = (float(midpoint[0]),)
    elif family == "H":
        local = (-float(midpoint[1]),)
    else:
        local = (-float(midpoint[1]), float(midpoint[0]), abs(angle_deg))

    return (family_bucket[family], *local)


def assign_chain_labels(
    candidates: list[dict[str, object]],
    positions: np.ndarray,
) -> list[BoundaryCreaseChain]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["pair_code"], candidate["family"])].append(candidate)

    for group_candidates in grouped.values():
        group_candidates.sort(key=group_member_sort_key)

    for (pair_code, family), group_candidates in grouped.items():
        for group_index, candidate in enumerate(group_candidates, start=1):
            candidate["crease_name"] = f"{pair_code}_{family}{group_index}"

    ordered = sorted(candidates, key=overall_chain_sort_key)
    chains: list[BoundaryCreaseChain] = []
    for chain_index, candidate in enumerate(ordered):
        crease_id = f"C{chain_index + 1:02d}"
        endpoint_a_id = f"P{chain_index + 1:02d}A"
        endpoint_b_id = f"P{chain_index + 1:02d}B"
        crease_name = candidate["crease_name"]
        endpoint_a_name = f"{crease_name}_A"
        endpoint_b_name = f"{crease_name}_B"
        endpoint_a = int(candidate["endpoint_a"])
        endpoint_b = int(candidate["endpoint_b"])

        chains.append(
            BoundaryCreaseChain(
                index=chain_index,
                crease_id=crease_id,
                crease_name=crease_name,
                family=str(candidate["family"]),
                angle_deg=float(candidate["angle_deg"]),
                vertex_ids=tuple(int(v) for v in candidate["vertex_ids"]),
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                endpoint_a_id=endpoint_a_id,
                endpoint_b_id=endpoint_b_id,
                endpoint_a_name=endpoint_a_name,
                endpoint_b_name=endpoint_b_name,
                endpoint_a_xyz=tuple(float(v) for v in positions[endpoint_a]),
                endpoint_b_xyz=tuple(float(v) for v in positions[endpoint_b]),
                endpoint_a_side=str(candidate["endpoint_a_side"]),
                endpoint_b_side=str(candidate["endpoint_b_side"]),
                length=float(candidate["length"]),
            )
        )

    return chains


def detect_boundary_creases(
    mesh: SimplicialComplex,
    asset_filename: str | None = None,
) -> dict[str, object]:
    topo = build_topology(mesh)
    positions = topo["positions"]
    boundary_edges = topo["boundary_edges"]
    boundary_vertices = set(topo["boundary_vertices"])
    boundary_loop = order_boundary_loop(
        largest_boundary_component(topo["boundary_adj"]),
        topo["boundary_adj"],
    )

    plane_axes, normal_axis, bbox = detect_planar_frame(positions, boundary_loop)
    plane_axis_names = tuple(axis_name(axis) for axis in plane_axes)
    normal_axis_name = axis_name(normal_axis)
    plane_positions = positions[:, plane_axes]
    bbox_diag = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2]])))
    min_crease_length = max(MIN_CREASE_LENGTH_RATIO * bbox_diag, 1.0e-3)

    candidates: list[dict[str, object]] = []
    used_signature_decimals = LINE_SIGNATURE_DECIMALS
    best_count = -1
    for signature_decimals in range(LINE_SIGNATURE_DECIMALS, MIN_LINE_SIGNATURE_DECIMALS - 1, -1):
        trial_candidates = collect_crease_candidates(
            topo,
            plane_positions,
            boundary_edges,
            boundary_vertices,
            bbox,
            min_crease_length,
            signature_decimals,
        )
        if len(trial_candidates) > best_count:
            candidates = trial_candidates
            used_signature_decimals = signature_decimals
            best_count = len(trial_candidates)
        if len(trial_candidates) == EXPECTED_CREASE_COUNT:
            candidates = trial_candidates
            used_signature_decimals = signature_decimals
            break

    # if len(candidates) != EXPECTED_CREASE_COUNT:
    #     raise AssertionError(
    #         f"expected {EXPECTED_CREASE_COUNT} boundary-to-boundary creases, got {len(candidates)}"
    #     )

    chains = assign_chain_labels(candidates, positions)

    crease_lookup: dict[str, BoundaryCreaseChain] = {}
    endpoint_lookup: dict[str, int] = {}
    point_lookup: dict[str, int] = {}
    for chain in chains:
        crease_lookup[chain.crease_id] = chain
        crease_lookup[chain.crease_name] = chain
        endpoint_lookup[chain.endpoint_a_id] = chain.endpoint_a
        endpoint_lookup[chain.endpoint_b_id] = chain.endpoint_b
        endpoint_lookup[chain.endpoint_a_name] = chain.endpoint_a
        endpoint_lookup[chain.endpoint_b_name] = chain.endpoint_b
        point_lookup[chain.endpoint_a_id] = chain.endpoint_a
        point_lookup[chain.endpoint_b_id] = chain.endpoint_b
        point_lookup[chain.endpoint_a_name] = chain.endpoint_a
        point_lookup[chain.endpoint_b_name] = chain.endpoint_b

    print(
        f"[boundary_crease_debug] detected {len(chains)} boundary-to-boundary crease chains "
        f"on {asset_filename or DEFAULT_ASSET_FILENAME}"
    )
    if used_signature_decimals != LINE_SIGNATURE_DECIMALS:
        print(
            f"[boundary_crease_debug] line signature decimals relaxed "
            f"from {LINE_SIGNATURE_DECIMALS} to {used_signature_decimals}"
        )
    print(
        "[boundary_crease_debug] alias rule: "
        "BT_V* = bottom-to-top verticals, LR_H* = left-to-right horizontals, "
        "DP/DN = positive/negative diagonals, suffix A/B follows the chain direction"
    )
    for chain in chains:
        ax, ay, az = chain.endpoint_a_xyz
        bx, by, bz = chain.endpoint_b_xyz
        print(
            f"[boundary_crease_debug] {chain.crease_id}/{chain.crease_name} "
            f"family={chain.family:<2s} angle={chain.angle_deg:+7.3f} deg "
            f"verts={len(chain.vertex_ids):3d} len={chain.length:.4f} "
            f"{chain.endpoint_a_id}/{chain.endpoint_a_name}=v{chain.endpoint_a:<5d} "
            f"{chain.endpoint_a_side:<6s} ({ax:+.6f}, {ay:+.6f}, {az:+.6f}) -> "
            f"{chain.endpoint_b_id}/{chain.endpoint_b_name}=v{chain.endpoint_b:<5d} "
            f"{chain.endpoint_b_side:<6s} ({bx:+.6f}, {by:+.6f}, {bz:+.6f})"
        )

    return {
        **topo,
        "asset_filename": asset_filename or DEFAULT_ASSET_FILENAME,
        "boundary_loop": boundary_loop,
        "bbox": bbox,
        "plane_axes": plane_axes,
        "plane_axis_names": plane_axis_names,
        "normal_axis": normal_axis,
        "normal_axis_name": normal_axis_name,
        "line_signature_decimals": used_signature_decimals,
        "chains": chains,
        "crease_lookup": crease_lookup,
        "endpoint_lookup": endpoint_lookup,
        "point_lookup": point_lookup,
    }


def build_curve_overlay(rest_positions: np.ndarray, chains: list[BoundaryCreaseChain]) -> dict[str, np.ndarray]:
    unique_vertices: list[int] = []
    vertex_to_overlay: dict[int, int] = {}
    overlay_edges: list[tuple[int, int]] = []
    overlay_edge_keys: set[tuple[int, int]] = set()

    for chain in chains:
        for vid in chain.vertex_ids:
            if vid not in vertex_to_overlay:
                vertex_to_overlay[vid] = len(unique_vertices)
                unique_vertices.append(vid)
        for a, b in zip(chain.vertex_ids[:-1], chain.vertex_ids[1:]):
            edge_key = tuple(sorted((a, b)))
            if edge_key in overlay_edge_keys:
                continue
            overlay_edge_keys.add(edge_key)
            overlay_edges.append((vertex_to_overlay[a], vertex_to_overlay[b]))

    overlay_positions = rest_positions[np.array(unique_vertices, dtype=np.int32)]
    return {
        "vertex_ids": np.array(unique_vertices, dtype=np.int32),
        "positions": overlay_positions,
        "edges": np.array(overlay_edges, dtype=np.int32),
    }


def build_vertex_phase_map(chains: list[BoundaryCreaseChain], vertex_count: int) -> np.ndarray:
    phase = np.zeros(vertex_count, dtype=np.float64)
    assigned = np.zeros(vertex_count, dtype=bool)

    for chain in chains:
        chain_phase = 0.0 if (chain.index % 2 == 0) else math.pi
        for vid in chain.vertex_ids:
            if assigned[vid]:
                continue
            phase[vid] = chain_phase
            assigned[vid] = True

    return phase


def oscillation_scale(frame: int) -> float:
    if frame < SETTLE_FRAMES:
        return 0.0
    if frame >= SETTLE_FRAMES + OSCILLATION_FRAMES:
        return 0.0

    alpha = (frame - SETTLE_FRAMES) / max(OSCILLATION_FRAMES - 1, 1)
    return math.sin(math.pi * alpha) ** 2


def default_label_map_path(asset_filename: str | None = None) -> Path:
    workspace = Path(AssetDir.output_path(__file__))
    asset_path = Path(resolve_origami_asset_path(asset_filename))
    return workspace / f"{asset_path.stem}_boundary_crease_label_map.png"


def export_label_map(
    diagnostics: dict[str, object],
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Label-map export requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    output = Path(output_path) if output_path is not None else default_label_map_path(
        diagnostics["asset_filename"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = diagnostics["positions"]
    plane_axes = diagnostics["plane_axes"]
    boundary_loop = diagnostics["boundary_loop"]
    chains = diagnostics["chains"]
    plane_positions = positions[:, plane_axes]

    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_axes([0.06, 0.08, 0.58, 0.84])
    legend_ax = fig.add_axes([0.68, 0.08, 0.29, 0.84])
    legend_ax.axis("off")

    boundary_ids = np.array(boundary_loop + [boundary_loop[0]], dtype=np.int32)
    boundary_poly = plane_positions[boundary_ids]
    ax.plot(boundary_poly[:, 0], boundary_poly[:, 1], color="#202020", linewidth=2.0, zorder=1)

    cmap = plt.get_cmap("tab20")
    for chain in chains:
        color = cmap(chain.index % 20)
        chain_pos = plane_positions[np.array(chain.vertex_ids, dtype=np.int32)]
        ax.plot(chain_pos[:, 0], chain_pos[:, 1], color=color, linewidth=2.2, zorder=2)
        ax.scatter(
            [chain_pos[0, 0], chain_pos[-1, 0]],
            [chain_pos[0, 1], chain_pos[-1, 1]],
            color=color,
            s=24.0,
            zorder=3,
        )

        mid = chain_pos[len(chain_pos) // 2]
        ax.text(
            float(mid[0]),
            float(mid[1]),
            chain.crease_name,
            fontsize=8.0,
            ha="center",
            va="center",
            color="#111111",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.8},
            zorder=4,
        )

        for endpoint_pos, endpoint_id, endpoint_name in (
            (chain_pos[0], chain.endpoint_a_id, chain.endpoint_a_name),
            (chain_pos[-1], chain.endpoint_b_id, chain.endpoint_b_name),
        ):
            ax.text(
                float(endpoint_pos[0]),
                float(endpoint_pos[1]),
                endpoint_id,
                fontsize=7.5,
                ha="left",
                va="bottom",
                color=color,
                zorder=5,
            )

    xmin, xmax, ymin, ymax = diagnostics["bbox"]
    xpad = 0.08 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal")
    asset_stem = Path(resolve_origami_asset_path(diagnostics["asset_filename"])).stem
    ax.set_title(f"{asset_stem} boundary-to-boundary crease labels", fontsize=13.0)
    ax.set_xlabel(f"{axis_name(plane_axes[0])}")
    ax.set_ylabel(f"{axis_name(plane_axes[1])}")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    legend_ax.text(0.0, 0.985, f"Asset: {diagnostics['asset_filename']}", fontsize=10.0, va="top")
    legend_ax.text(0.0, 0.955, f"Detected creases: {len(chains)}", fontsize=10.0, va="top")
    legend_ax.text(
        0.0,
        0.925,
        "Alias rule:\nBT_V* verticals\nLR_H* horizontals\nDP/DN diagonals",
        fontsize=9.5,
        va="top",
    )

    y = 0.84
    row_h = 0.055
    for chain in chains:
        color = cmap(chain.index % 20)
        ax0, ay0, az0 = chain.endpoint_a_xyz
        bx0, by0, bz0 = chain.endpoint_b_xyz
        legend_ax.text(
            0.0,
            y,
            (
                f"{chain.crease_id}/{chain.crease_name}\n"
                f"{chain.endpoint_a_id}={chain.endpoint_a_name}=v{chain.endpoint_a} "
                f"({chain.endpoint_a_side})\n"
                f"{chain.endpoint_b_id}={chain.endpoint_b_name}=v{chain.endpoint_b} "
                f"({chain.endpoint_b_side})"
            ),
            fontsize=8.6,
            va="top",
            color=color,
        )
        y -= row_h
        if y < 0.04:
            break

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def build_demo(asset_filename: str | None = None) -> dict[str, object]:
    try:
        from uipc.constitution import (
            NeoHookeanShell,
            PlasticDiscreteShellBending,
            SoftPositionConstraint,
            ElasticModuli2D,
        )
        from uipc.geometry import mesh_partition
    except ImportError as exc:
        raise SystemExit(
            "This example requires newer libuipc Python bindings with "
            "`NeoHookeanShell`, `PlasticDiscreteShellBending`, "
            "`SoftPositionConstraint`, `ElasticModuli2D`, and `mesh_partition`."
        ) from exc

    Logger.set_level(Logger.Level.Warn)

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["line_search"]["max_iter"] = 12
    config["linear_system"]["tol_rate"] = 1.0e-3
    scene = Scene(config)

    plane_object = scene.objects().create("paper_plane_boundary")

    shell = NeoHookeanShell()
    plastic_bending = PlasticDiscreteShellBending()
    position_constraint = SoftPositionConstraint()

    plane = load_paper_plane_mesh(asset_filename)
    diagnostics = detect_boundary_creases(plane, asset_filename)

    mesh_partition(plane, MESH_PARTITION_SIZE)
    moduli = ElasticModuli2D.youngs_poisson(SHELL_YOUNG, SHELL_POISSON)
    shell.apply_to(plane, moduli, SHELL_DENSITY, SHELL_THICKNESS)
    plastic_bending.apply_to(
        plane,
        SHELL_BENDING_STIFFNESS,
        SHELL_YIELD_THRESHOLD,
        SHELL_HARDENING_MODULUS,
    )
    position_constraint.apply_to(plane, CONSTRAINT_STRENGTH)

    rest_positions = np.array(view(plane.positions()), copy=True).reshape(-1, 3)
    plane_slot = plane_object.geometries().create(plane)[0]

    chain_vertex_ids = np.unique(
        np.array([vid for chain in diagnostics["chains"] for vid in chain.vertex_ids], dtype=np.int32)
    )
    phase_map = build_vertex_phase_map(diagnostics["chains"], rest_positions.shape[0])
    overlay = build_curve_overlay(rest_positions, diagnostics["chains"])
    normal_axis = diagnostics["normal_axis"]

    def animate_plane(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        is_constrained = view(geo.vertices().find(builtin.is_constrained))
        aim_position = view(geo.vertices().find(builtin.aim_position))

        frame = max(info.frame() - 1, 0)
        alpha = 0.0
        if SETTLE_FRAMES <= frame < SETTLE_FRAMES + OSCILLATION_FRAMES:
            alpha = (frame - SETTLE_FRAMES) / max(OSCILLATION_FRAMES - 1, 1)

        targets = np.array(rest_positions, copy=True)
        scale = oscillation_scale(frame)
        base_phase = 2.0 * math.pi * OSCILLATION_CYCLES * alpha
        if scale > 0.0:
            offsets = OSCILLATION_AMPLITUDE * scale * np.sin(base_phase + phase_map)
            targets[chain_vertex_ids, normal_axis] += offsets[chain_vertex_ids]

        is_constrained[:] = 1
        aim_position[:] = targets.reshape(-1, 3, 1)

    scene.animator().insert(plane_object, animate_plane)
    world.init(scene)

    return {
        "engine": engine,
        "world": world,
        "scene": scene,
        "scene_io": SceneIO(scene),
        "plane_slot": plane_slot,
        "rest_positions": rest_positions,
        "diagnostics": diagnostics,
        "chain_vertex_ids": chain_vertex_ids,
        "overlay": overlay,
    }


def run_demo(asset_filename: str | None = None):
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This example requires `polyscope`. Install it with `pip install polyscope`."
        ) from exc

    state = build_demo(asset_filename)
    world = state["world"]
    scene_io = state["scene_io"]
    plane_slot = state["plane_slot"]
    diagnostics = state["diagnostics"]
    overlay = state["overlay"]

    ps.init()
    ps.set_ground_plane_mode("none")

    surface = scene_io.simplicial_surface()
    mesh = ps.register_surface_mesh(
        "paper_plane_boundary_creases",
        surface.positions().view().reshape(-1, 3),
        surface.triangles().topo().view().reshape(-1, 3),
    )
    mesh.set_edge_width(1.0)

    curve = ps.register_curve_network(
        "paper_plane_boundary_crease_chains",
        overlay["positions"],
        overlay["edges"],
    )
    curve.set_radius(0.0035, relative=False)
    curve.set_color((0.95, 0.45, 0.12))

    ui_state = {"run": True}

    def current_positions() -> np.ndarray:
        return np.array(view(plane_slot.geometry().positions()), copy=True).reshape(-1, 3)

    def update_visuals():
        merged = scene_io.simplicial_surface()
        mesh.update_vertex_positions(merged.positions().view().reshape(-1, 3))
        plane_positions = current_positions()
        curve.update_node_positions(plane_positions[overlay["vertex_ids"]])

    def advance_one_frame():
        if world.frame() >= TOTAL_FRAMES:
            ui_state["run"] = False
            return

        world.advance()
        if not world.is_valid():
            ui_state["run"] = False
            return

        world.retrieve()
        update_visuals()

        if world.frame() >= TOTAL_FRAMES:
            ui_state["run"] = False

    def on_update():
        if psim.Button("run / pause"):
            ui_state["run"] = not ui_state["run"]

        psim.SameLine()
        if psim.Button("step"):
            advance_one_frame()

        if ui_state["run"]:
            advance_one_frame()

        plane_axes = diagnostics["plane_axis_names"]

        psim.Separator()
        psim.Text("Diagnostic only: detected boundary-to-boundary crease vertices oscillate twice")
        psim.Text(f"Frame: {world.frame()} / {TOTAL_FRAMES}")
        psim.Text(f"Asset: {diagnostics['asset_filename']}")
        psim.Text(
            f"Plane axes: {plane_axes[0]}, {plane_axes[1]}  normal: {diagnostics['normal_axis_name']}"
        )
        psim.Text(f"Oscillation amplitude: {OSCILLATION_AMPLITUDE:.4f}")
        psim.Text(f"Detected creases: {len(diagnostics['chains'])}")
        psim.Text(
            "Alias rule: BT_V* verticals, LR_H* horizontals, DP/DN positive/negative diagonals"
        )
        psim.Separator()
        psim.Text("Crease | Alias | Family | Angle | Vertices | A endpoint | B endpoint")
        for chain in diagnostics["chains"]:
            ax, ay, az = chain.endpoint_a_xyz
            bx, by, bz = chain.endpoint_b_xyz
            psim.Text(
                f"{chain.crease_id:<3s} | {chain.crease_name:<8s} | {chain.family:<2s} | "
                f"{chain.angle_deg:+7.2f} | {len(chain.vertex_ids):3d} | "
                f"{chain.endpoint_a_name:<12s}=v{chain.endpoint_a:<5d} {chain.endpoint_a_side:<6s} "
                f"({ax:+.4f}, {ay:+.4f}, {az:+.4f}) | "
                f"{chain.endpoint_b_name:<12s}=v{chain.endpoint_b:<5d} {chain.endpoint_b_side:<6s} "
                f"({bx:+.4f}, {by:+.4f}, {bz:+.4f})"
            )

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    run_demo(sys.argv[1] if len(sys.argv) > 1 else None)
