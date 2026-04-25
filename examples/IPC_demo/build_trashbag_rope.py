"""
Drawstring trash bag: channel with TWO opposite gaps + circular rope.

The tunnel has two openings 180 degrees apart on the rim.
The rope forms a full circle inside the tunnel, with two tails
exiting from the two opposite gaps.
"""

import numpy as np
from collections import Counter, defaultdict

INPUT = "/home/zhehuan/Desktop/hz/gs-core/data/IPC_demo/ipc_trashbag/Trashbag_coarse.obj"
OUTPUT_BAG = "/home/zhehuan/Desktop/hz/gs-core/data/IPC_demo/ipc_trashbag/Trashbag_rope.obj"
OUTPUT_ROPE1 = "/home/zhehuan/Desktop/hz/gs-core/data/IPC_demo/ipc_trashbag/rope1.obj"
OUTPUT_ROPE2 = "/home/zhehuan/Desktop/hz/gs-core/data/IPC_demo/ipc_trashbag/rope2.obj"
OUTPUT_GLB = "/home/zhehuan/Desktop/hz/gs-core/data/IPC_demo/ipc_trashbag/Trashbag_rope.glb"

CHANNEL_RADIUS = 0.010
ARC_SEGMENTS = 5
GAP_COUNT = 5  # vertices per gap
ROPE_OFFSET = CHANNEL_RADIUS * 0.4
Z_OFFSET = 0.012  # vertical separation between two ropes (> 4 * rope_thickness=0.0025)
ARC_BULGE = CHANNEL_RADIUS * 10.0
N_ARC = 8


def read_obj(path):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                faces.append([int(p.split("/")[0]) for p in parts[1:]])
    return np.array(vertices, dtype=np.float64), faces


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    print(f"Wrote {path}  ({len(vertices)} verts, {len(faces)} faces)")


def write_rope_obj(path, points):
    with open(path, "w") as f:
        for v in points:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for i in range(1, len(points)):
            f.write(f"l {i} {i + 1}\n")
    print(f"Wrote {path}  ({len(points)} verts, {len(points) - 1} edges)")


def _trace_boundary_loop(faces):
    """Find a single boundary loop from face list. Returns list of 1-based vertex indices."""
    edge_count = Counter()
    for face in faces:
        n = len(face)
        for i in range(n):
            e = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_count[e] += 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    start = boundary_edges[0][0]
    loop = [start]
    visited = {start}
    current = start
    while True:
        nxt = [n for n in adj[current] if n not in visited]
        if not nxt:
            break
        current = nxt[0]
        loop.append(current)
        visited.add(current)
    return loop


def find_boundary_loop(vertices, faces):
    """Find top boundary loop."""
    return _trace_boundary_loop(faces)


def find_bottom_ring(vertices, faces, strip_loop, match_to=None):
    """Find bottom ring by removing strip_loop faces and tracing the new boundary.

    Args:
        vertices: mesh vertices array
        faces: face list to search
        strip_loop: vertex loop to remove (faces touching these are stripped)
        match_to: vertex loop to match result ordering to (default: strip_loop)

    Returns (matched_ring, inner_faces):
        matched_ring: list of 1-based vertex indices matched to match_to order.
            If match_to has more verts than the ring, some ring verts are reused.
        inner_faces: remaining faces after stripping
    """
    from scipy.spatial import cKDTree

    if match_to is None:
        match_to = strip_loop

    loop_set = set(strip_loop)
    inner_faces = [f for f in faces if not any(v in loop_set for v in f)]
    ring = _trace_boundary_loop(inner_faces)
    print(f"Bottom ring: {len(ring)} vertices (from mesh topology)")

    ref_pos = vertices[np.array(match_to) - 1]
    ring_pos = vertices[np.array(ring) - 1]
    tree = cKDTree(ring_pos[:, [0, 2]])
    _, match_idx = tree.query(ref_pos[:, [0, 2]])
    return ring, match_idx, inner_faces


def recenter_bottom(vertices, faces, power=3.0, strength=1.0):
    """Translate the bag's bottom to sit under the mouth axis, preserving local shape.

    Computes the XZ offset between the bottom centroid and the mouth axis,
    then applies a smooth height-weighted translation to remove the lean.
    weight = (1 - t)^power * strength, where t is normalized height (0=bottom, 1=top).
    This shifts vertices without collapsing their spread — the bag stays round.
    """
    out = vertices.copy()
    y = vertices[:, 1]
    y_min, y_max = y.min(), y.max()
    t = (y - y_min) / (y_max - y_min)
    w = (1.0 - t) ** power * strength

    # Mouth axis = XZ centroid of top 5%
    top_mask = t > 0.95
    axis_x = vertices[top_mask, 0].mean()
    axis_z = vertices[top_mask, 2].mean()

    # Bottom centroid = XZ centroid of bottom 5%
    bot_mask = t < 0.05
    bot_cx = vertices[bot_mask, 0].mean()
    bot_cz = vertices[bot_mask, 2].mean()

    # Offset to remove (lean direction)
    offset_x = bot_cx - axis_x
    offset_z = bot_cz - axis_z

    # Translate — shift XZ by -w * offset (preserves local shape)
    out[:, 0] = vertices[:, 0] - w * offset_x * strength
    out[:, 2] = vertices[:, 2] - w * offset_z * strength

    disp = np.linalg.norm(out - vertices, axis=1)
    print(f"Recenter bottom: offset=({offset_x:.4f}, {offset_z:.4f}), max_disp={disp.max():.6f}")
    return out


def round_bottom(vertices, faces, push_dist=0.04, push_power=5.0, smooth_iters=30, smooth_lambda=0.5, smooth_power=3.0):
    """Round the bag bottom by pushing vertices outward, then Laplacian-smoothing.

    Step 1 — Radial expansion: push each vertex outward from the bag axis by
    push_dist * (1-t)^push_power.  This widens the pinched bottom.

    Step 2 — Laplacian smooth XZ only with weight (1-t)^smooth_power to even
    out irregularities from the expansion.

    Y positions are never touched — bag height and face sizes are preserved.
    All weights are continuous (no threshold), so no discontinuities.
    """
    from collections import defaultdict as ddict

    out = vertices.copy()
    y = vertices[:, 1]
    y_min, y_max = y.min(), y.max()
    t = (y - y_min) / (y_max - y_min)

    # Bag axis (XZ centroid of top region)
    top_mask = t > 0.95
    axis_x = vertices[top_mask, 0].mean()
    axis_z = vertices[top_mask, 2].mean()

    # ── Step 1: radial outward push ──
    for i in range(len(vertices)):
        dx = out[i, 0] - axis_x
        dz = out[i, 2] - axis_z
        r = np.sqrt(dx * dx + dz * dz)
        if r < 1e-10:
            continue
        push = push_dist * (1.0 - t[i]) ** push_power
        out[i, 0] += dx / r * push
        out[i, 2] += dz / r * push

    # ── Step 2: Laplacian smooth XZ ──
    adj = ddict(set)
    for face in faces:
        idxs = [v - 1 for v in face]
        for a in idxs:
            for b in idxs:
                if a != b:
                    adj[a].add(b)

    smooth_w = (1.0 - t) ** smooth_power

    for _iteration in range(smooth_iters):
        new_pos = out.copy()
        for i in range(len(out)):
            if smooth_w[i] < 1e-8 or not adj[i]:
                continue
            neighbors = list(adj[i])
            avg = out[neighbors].mean(axis=0)
            w = smooth_lambda * smooth_w[i]
            new_pos[i, 0] = out[i, 0] + w * (avg[0] - out[i, 0])
            new_pos[i, 2] = out[i, 2] + w * (avg[2] - out[i, 2])
        out = new_pos

    disp = np.linalg.norm(out - vertices, axis=1)
    print(f"Round bottom: push={push_dist}, max_disp={disp.max():.6f}, affected={int((disp > 1e-4).sum())} verts")
    return out


def main():
    vertices, faces = read_obj(INPUT)
    print(f"Original: {len(vertices)} vertices, {len(faces)} faces")

    # Fix asymmetric bottom — smooth recentering before building channel/ropes
    vertices = recenter_bottom(vertices, faces)
    # Laplacian smooth XZ only — spreads vertices more evenly, reduces creases
    vertices = round_bottom(vertices, faces)

    boundary_loop = find_boundary_loop(vertices, faces)
    N = len(boundary_loop)
    print(f"Boundary loop: {N} vertices")

    loop_pos = vertices[np.array(boundary_loop) - 1]
    centroid = loop_pos.mean(axis=0)

    # Outward direction (radial in XZ)
    outward = loop_pos - centroid
    outward[:, 1] = 0
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-8)

    up_dir = np.zeros_like(outward)
    up_dir[:, 1] = 1.0

    # ── Four small holes at rope entry/exit points ──
    # Two locations 180° apart, each with an entry and exit hole (1 vertex wide)
    gap1_center = 0
    gap2_center = N // 2
    # Entry/exit are GAP_COUNT apart at each location
    hole1_entry = gap1_center
    hole1_exit = (gap1_center + GAP_COUNT) % N
    hole2_entry = gap2_center
    hole2_exit = (gap2_center + GAP_COUNT) % N
    hole1_gaps = {hole1_entry, hole1_exit - 1}
    hole2_gaps = {hole2_entry, hole2_exit - 1}

    print(f"Hole 1: entry={hole1_entry}, exit={hole1_exit}")
    print(f"Hole 2: entry={hole2_entry}, exit={hole2_exit}")
    # ══════════════════════════════════════════════
    # BUILD CHANNEL MESH
    # ══════════════════════════════════════════════

    bottom_ring1, _, inner_faces1 = find_bottom_ring(vertices, faces, boundary_loop)
    bottom_ring2, match_idx, _ = find_bottom_ring(vertices, inner_faces1, bottom_ring1, match_to=boundary_loop)

    top_y = vertices[np.array(boundary_loop) - 1, 1].mean()
    mid_y = vertices[np.array(bottom_ring1) - 1, 1].mean()
    bot_y = vertices[np.array(bottom_ring2) - 1, 1].mean()
    print(f"Ring Y (top->bottom): boundary={top_y:.4f}, ring1={mid_y:.4f}, ring2={bot_y:.4f}")

    # Compute channel width from distance between boundary_loop and second bottom_ring2
    actual_width = top_y - bot_y
    print(f"Channel width: {actual_width:.4f}")

    mid_pos = loop_pos.copy()
    mid_pos[:, 1] -= actual_width / 2.0

    n_rings = ARC_SEGMENTS + 1
    thetas = np.linspace(np.pi / 2, -np.pi / 2, n_rings)

    new_verts = list(vertices)
    ring_indices = []

    for i in range(N):
        row = []
        for k in range(n_rings - 1):
            if k == 0:
                # Top: reuse boundary loop vertex
                row.append(boundary_loop[i])
            else:
                # Middle arc points: new vertices
                theta = thetas[k]
                pos = (
                    mid_pos[i]
                    + CHANNEL_RADIUS * np.cos(theta) * outward[i]
                    + (actual_width / 2) * np.sin(theta) * up_dir[i]
                )
                new_verts.append(pos)
                row.append(len(new_verts))
        ring_indices.append(row)

    new_verts = np.array(new_verts)

    # Build arc faces (skip both gaps, no end caps)
    tunnel_faces = []
    n_mid = ARC_SEGMENTS / 2
    for i in range(N):
        j = (i + 1) % N
        for k in range(n_rings - 2):
            if i in hole1_gaps and n_mid - 2 <= k <= n_mid:
                continue
            if i in hole2_gaps and n_mid - 1 <= k <= n_mid + 1:
                continue
            v0 = ring_indices[i][k]
            v1 = ring_indices[i][k + 1]
            v2 = ring_indices[j][k + 1]
            v3 = ring_indices[j][k]
            tunnel_faces.append([v0, v3, v2])
            tunnel_faces.append([v0, v2, v1])

        # Fan-triangulate the bottom row connecting last arc ring to bottom_ring2
        k = n_rings - 2
        v0 = ring_indices[i][k]
        v3 = ring_indices[j][k]
        N_br = len(bottom_ring2)
        mi = match_idx[i]
        mj = match_idx[j]
        # Fan-triangulate from mi to mj (modular, forward direction)
        n_steps = (mj - mi) % N_br
        assert n_steps <= N_br // 2, f"Backwards step at i={i}, mi={mi}, mj={mj}, n_steps={n_steps}"
        vi = mi
        for _ in range(n_steps):
            vi_next = (vi + 1) % N_br
            tunnel_faces.append([v0, bottom_ring2[vi_next], bottom_ring2[vi]])
            vi = vi_next
        tunnel_faces.append([v0, v3, bottom_ring2[mj]])

    # Remove degenerate
    tunnel_faces = [f for f in tunnel_faces if len(set(f)) >= 3]
    print(f"Tunnel faces: {len(tunnel_faces)}")

    all_faces = faces + tunnel_faces
    write_obj(OUTPUT_BAG, new_verts, all_faces)

    # ══════════════════════════════════════════════
    # BUILD ROPE (full circle + 2 tails)
    # ══════════════════════════════════════════════
    # One continuous rope forming a full circle inside the channel.
    # At gap1: two tails exit (rope starts and ends here).
    # At gap2: rope crosses the gap with a smooth outward arc (no U-turn).
    #
    # Path: tail1 → enter gap1 → half circle → smooth arc across gap2
    #       → other half circle → exit gap1 → tail2

    # ══════════════════════════════════════════════
    # BUILD TWO ROPE LOOPS
    # ══════════════════════════════════════════════
    # Rope 1 (higher): exits at hole1 (gap1), runs through channel past hole2
    # Rope 2 (lower):  exits at hole2 (gap2), runs through channel past hole1
    # Both are closed loops with a Z offset to avoid collision.

    # Average outward direction at each gap (between entry and exit holes)
    gap1_outward = (outward[hole1_entry] + outward[hole1_exit]) / 2
    gap1_outward /= np.linalg.norm(gap1_outward)
    gap2_outward = (outward[hole2_entry] + outward[hole2_exit]) / 2
    gap2_outward /= np.linalg.norm(gap2_outward)

    def bezier_arc(p0, p1, bulge_dir, bulge_dist, n_pts):
        """Quadratic Bezier arc from p0 to p1, bulging along bulge_dir."""
        ctrl = (p0 + p1) / 2 + bulge_dir * bulge_dist
        pts = []
        for t in range(n_pts + 1):
            s = t / n_pts
            pt = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s**2 * p1
            pts.append(pt)
        return pts

    def collect_channel_segment(start_idx, end_idx, points):
        """Collect channel points from start_idx to end_idx (inclusive), walking forward."""
        seg = []
        i = start_idx
        while True:
            seg.append(points[i])
            if i == end_idx:
                break
            i = (i + 1) % N
        return seg

    def make_rope_loop(circle_pts, exit_hole_entry, exit_hole_exit, exit_outward):
        """Build a closed rope loop that exits at one hole and stays inside everywhere else.

        The rope runs the full circle inside the channel, except at the exit hole
        where it arcs outside via a Bezier curve.
        """
        # Segment from after exit_hole_exit to before exit_hole_entry (the long way around,
        # staying inside the channel past the other hole)
        end = (exit_hole_entry) % N
        end_next = (exit_hole_entry + 1) % N
        start = (exit_hole_exit) % N
        start_prev = (exit_hole_exit - 1) % N
        inside_seg = collect_channel_segment(start, end, circle_pts)

        # Bezier arc outside the channel at the exit hole
        # Arc from end of inside segment to start of inside segment (across the hole)
        outside_arc = bezier_arc(
            (circle_pts[end] * 2 + circle_pts[end_next]) / 3,
            (circle_pts[start] * 2 + circle_pts[start_prev]) / 3,
            exit_outward,
            ARC_BULGE,
            N_ARC,
        )

        # Close the loop: inside_seg → outside_arc (skip duplicate endpoints) → back to start
        loop = inside_seg[1:-1] + outside_arc
        return np.array(loop)

    # Base circle points (at ROPE_OFFSET from channel center)
    circle_points = np.array([mid_pos[i] + outward[i] * ROPE_OFFSET for i in range(N)])

    # Rope 1: slightly higher, exits at hole1
    circle_pts_1 = circle_points.copy()
    circle_pts_1[:, 1] += Z_OFFSET / 2  # Y-up: shift up
    rope1 = make_rope_loop(circle_pts_1, hole1_entry, hole1_exit, gap1_outward)

    # Rope 2: slightly lower, exits at hole2
    circle_pts_2 = circle_points.copy()
    circle_pts_2[:, 1] -= Z_OFFSET / 2  # Y-up: shift down
    rope2 = make_rope_loop(circle_pts_2, hole2_entry, hole2_exit, gap2_outward)

    # Write each rope as a separate OBJ
    for rope, output_path, name in [
        (rope1, OUTPUT_ROPE1, "Rope 1"),
        (rope2, OUTPUT_ROPE2, "Rope 2"),
    ]:
        n = len(rope)
        with open(output_path, "w") as f:
            for v in rope:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for i in range(1, n):
                f.write(f"l {i} {i + 1}\n")
            f.write(f"l {n} 1\n")  # close the loop
        print(f"{name}: {n} verts, wrote {output_path}")

    # ══════════════════════════════════════════════
    # EXPORT GLB with PBR materials
    # ══════════════════════════════════════════════
    import trimesh

    N_ORIG_FACES = len(faces)
    bag_face_arr = np.array(all_faces[:N_ORIG_FACES]) - 1  # OBJ 1-indexed → 0-indexed
    channel_face_arr = np.array(all_faces[N_ORIG_FACES:]) - 1

    def compact_mesh(verts, face_indices):
        """Extract only used vertices and remap face indices."""
        used = np.unique(face_indices.flatten())
        remap = np.full(len(verts), -1, dtype=int)
        remap[used] = np.arange(len(used))
        return verts[used], remap[face_indices]

    bag_v, bag_f = compact_mesh(new_verts, bag_face_arr)
    ch_v, ch_f = compact_mesh(new_verts, channel_face_arr)

    bag_mesh = trimesh.Trimesh(vertices=bag_v, faces=bag_f, process=False)
    bag_mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="plastic_bag",
            baseColorFactor=[240, 240, 245, 128],
            metallicFactor=0.0,
            roughnessFactor=0.01,
            alphaMode="BLEND",
            doubleSided=True,
            ior=2.0,
        )
    )

    channel_mesh = trimesh.Trimesh(vertices=ch_v, faces=ch_f, process=False)
    channel_mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="channel",
            baseColorFactor=[255, 200, 40, 255],
            metallicFactor=0.1,
            roughnessFactor=0.5,
            doubleSided=True,
        )
    )

    scene = trimesh.Scene()
    scene.add_geometry(bag_mesh, node_name="bag", geom_name="bag")
    scene.add_geometry(channel_mesh, node_name="channel", geom_name="channel")
    scene.export(OUTPUT_GLB)

    print("\nGLB export:")
    print(f"  Bag: {len(bag_v)} verts, {len(bag_f)} faces")
    print(f"  Channel: {len(ch_v)} verts, {len(ch_f)} faces")
    print(f"  Wrote {OUTPUT_GLB}")


if __name__ == "__main__":
    main()
