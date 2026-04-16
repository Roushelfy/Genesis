"""
Compare original (_v0) and decimated collision meshes side by side.

Usage:
    python compare_mesh.py <mesh_path>

Example:
    python compare_mesh.py hands/sharpa/left/meshes/wrist_collision.STL
    python compare_mesh.py hands/sharpa/right/meshes/right_hand_C_MC.STL
"""

import os
import sys
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_v0_path(mesh_path):
    base, ext = os.path.splitext(mesh_path)
    return f"{base}_v0{ext}"


def plot_mesh(ax, mesh, title, color):
    verts = mesh.vertices
    faces = mesh.faces
    polys = verts[faces]
    collection = Poly3DCollection(
        polys, alpha=0.6, edgecolor="k", linewidth=0.15, facecolor=color
    )
    ax.add_collection3d(collection)

    margin = 0.05 * mesh.extents.max()
    ax.set_xlim(verts[:, 0].min() - margin, verts[:, 0].max() + margin)
    ax.set_ylim(verts[:, 1].min() - margin, verts[:, 1].max() + margin)
    ax.set_zlim(verts[:, 2].min() - margin, verts[:, 2].max() + margin)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=135)


def compare(relative_mesh_path):
    full_path = os.path.join(BASE_DIR, relative_mesh_path.replace("/", os.sep))
    v0_path = get_v0_path(full_path)

    if not os.path.exists(v0_path):
        print(f"No _v0 backup found for {relative_mesh_path}")
        print(f"  Expected: {v0_path}")
        return
    if not os.path.exists(full_path):
        print(f"Decimated mesh not found: {full_path}")
        return

    v0 = trimesh.load(v0_path)
    dec = trimesh.load(full_path)

    print(f"Mesh: {relative_mesh_path}")
    print(f"  Original (v0): {len(v0.faces)} faces, {len(v0.vertices)} vertices")
    print(f"  Decimated:     {len(dec.faces)} faces, {len(dec.vertices)} vertices")
    print(f"  Ratio:         {len(dec.faces)/len(v0.faces)*100:.1f}%")

    name = os.path.basename(relative_mesh_path)
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(121, projection="3d")
    plot_mesh(ax1, v0, f"Original (v0)\n{len(v0.faces)}F / {len(v0.vertices)}V", "#4A90D9")

    ax2 = fig.add_subplot(122, projection="3d")
    plot_mesh(ax2, dec, f"Decimated ({len(dec.faces)/len(v0.faces)*100:.0f}%)\n{len(dec.faces)}F / {len(dec.vertices)}V", "#E8734A")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1])
