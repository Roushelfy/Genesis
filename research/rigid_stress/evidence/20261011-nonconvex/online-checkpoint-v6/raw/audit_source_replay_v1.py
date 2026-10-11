"""Verify live pre-integration geometry, source frames and supporting normals independently in FP64."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(__file__).parent
capture = root / 'source-replay-v2/results'
out = root / 'source-replay-audit-v1'
out.mkdir(exist_ok=False)
with np.load(capture / 'before-recovery-710.npz') as saved:
    geometry = {name: saved[name] for name in saved.files}
with np.load(root / 'online-square-fan-v1/results/failure/snapshot-710.npz') as saved:
    failure = {name: saved[name] for name in saved.files}
descriptions = json.loads((capture / 'geometry-description.json').read_text())
env, slot = 5, 0
frame_rotation = Rotation.from_quat(failure['frame_quaternion'][env, [1, 2, 3, 0]])
frame_position = failure['frame_position'][env]
geom = int(failure['geom'][env, slot])
target = frame_rotation.apply(failure['witness'][env, slot]) + frame_position
matches = []
for contact in range(int(geometry['contact_count'][env])):
    ga, gb = int(geometry['contact_geom_a'][env, contact]), int(geometry['contact_geom_b'][env, contact])
    if geom in (ga, gb):
        side = 'a' if ga == geom else 'b'
        delta = np.linalg.norm(geometry[f'witness_{side}'][env, contact]-target)
        matches.append((float(delta), contact, side))
matches.sort()
assert matches and matches[0][0] < 1e-12, matches
delta, contact, egg_side = matches[0]
normal = geometry['contact_normal'][env, contact]
record = {'scope': __doc__, 'tick': 710, 'env': env, 'raw_slot': slot,
    'collider_index': contact, 'reference_witness_world_difference_m': delta,
    'world_normal': normal.tolist(), 'producer_method': int(geometry['witness_method'][env, contact]),
    'geometries': []}
with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as saved:
    reference_vertices = saved['vertices']
for side in ('a', 'b'):
    g = int(geometry[f'contact_geom_{side}'][env, contact])
    start, end = int(geometry['geom_vertex_start'][g]), int(geometry['geom_vertex_end'][g])
    rotation = Rotation.from_quat(geometry['geom_quaternion'][env, g, [1, 2, 3, 0]])
    position = geometry['geom_position'][env, g]
    world = rotation.apply(geometry['vertices_local'][start:end]) + position
    witness = geometry[f'witness_{side}'][env, contact]
    dots = world @ normal
    dot = witness @ normal
    low, high = float(dot-dots.min()), float(dots.max()-dot)
    item = {'side': side, 'geom': g, 'link': int(geometry['geom_link'][g]),
        'vertices': len(world), 'low_support_gap_m': low, 'high_support_gap_m': high,
        'min_support_gap_m': min(low, high)}
    if side == egg_side:
        authored = frame_rotation.inv().apply(world-frame_position)
        distances = np.linalg.norm(authored[:, None]-reference_vertices[None], axis=2)
        item['reference_collision_vertex_max_difference_m'] = float(distances.min(axis=1).max())
        assert item['reference_collision_vertex_max_difference_m'] < 1e-13
        triangles = np.asarray(descriptions[g]['faces'], dtype=int)
        points = world[triangles]
        facet_normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
        facet_normals /= np.linalg.norm(facet_normals, axis=1)[:, None]
        item['nearest_true_facet_axis_normal_difference'] = float(np.minimum(
            np.linalg.norm(facet_normals-normal, axis=1), np.linalg.norm(facet_normals+normal, axis=1)).min())
        local_normal = frame_rotation.inv().apply(normal)
        if local_normal @ failure['normal'][env, slot] < 0:
            local_normal = -local_normal
        item['saved_reference_normal_difference'] = float(np.linalg.norm(local_normal-failure['normal'][env, slot]))
        assert item['saved_reference_normal_difference'] < 1e-13
    record['geometries'].append(item)
(out / 'audit.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
