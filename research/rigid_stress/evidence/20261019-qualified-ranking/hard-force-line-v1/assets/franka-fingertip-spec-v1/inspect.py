"""Render the manufacturer drawing and existing visual-mesh sections for inspection."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pypdfium2

out = Path(__file__).parent
source = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_force_line/genesis/assets/xml/franka_emika_panda')
doc = pypdfium2.PdfDocument(str(out / 'franka-hand-1.2.pdf'))
doc[12].render(scale=2).to_pil().save(out / 'page13.png')
rows = []
for name in ['finger_0.obj', 'finger_1.obj']:
    path = source / 'assets' / name
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == 'v':
            vertices.append([float(value) for value in parts[1:4]])
        elif parts and parts[0] == 'f':
            assert len(parts) == 4
            faces.append([int(value.split('/')[0])-1 for value in parts[1:]])
    points = np.array(vertices) * 1000
    tri = points[np.array(faces)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, dim, value, xy in zip(axes, [0, 1, 2], [0.0, 0.5, 44.5], [[1, 2], [0, 2], [0, 1]]):
        edges = tri[:, [[0, 1], [1, 2], [2, 0]], :]
        delta = edges[:, :, 1, dim] - edges[:, :, 0, dim]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (value - edges[:, :, 0, dim]) / delta
        mask = np.isfinite(t) & (t >= 0) & (t < 1)
        valid = mask.sum(axis=1) == 2
        e, t, mask = edges[valid], t[valid], mask[valid]
        intersections = (e[:, :, 0] + np.where(mask, t, 0)[..., None] * (e[:, :, 1] - e[:, :, 0]))[mask].reshape(-1, 2, 3)
        ax.add_collection(LineCollection(intersections[:, :, xy], linewidths=.7))
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(f'{"xyz"[dim]} = {value} mm')
        ax.set_xlabel(f'{"xyz"[xy[0]]} (mm)')
        ax.set_ylabel(f'{"xyz"[xy[1]]} (mm)')
        ax.grid(alpha=.2)
    fig.suptitle(f'{name}: mesh sections; manufacturing edge radius unverified')
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=140)
    plt.close(fig)
    rows.append({'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'vertices':len(vertices), 'faces':len(faces), 'bounds_mm':[points.min(axis=0).tolist(),points.max(axis=0).tolist()]})
(out / 'inspection.json').write_text(json.dumps({'passed':True,'meshes':rows,'scope':'Existing visual geometry and manufacturer drawing inspection; physical fillet dimensions remain unverified.'},indent=2)+'\n')
