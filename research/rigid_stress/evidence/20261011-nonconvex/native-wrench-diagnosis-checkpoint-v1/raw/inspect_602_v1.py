import json
from pathlib import Path
import numpy as np
root=Path(__file__).parent
with np.load(root/'warp-v7-online-b1024-v1/b1024-seed623201/results/failure/snapshot-602.npz') as saved:
    data={key:saved[key] for key in saved.files}
ids=np.flatnonzero(data['valid'][66])
print(json.dumps({key:data[key][66,ids].tolist() for key in ('position','force','normal','witness','anchor','anchor_face','geom','pair','method','radius','friction')},indent=2),flush=True)
with np.load(root/'exact-602-socp-v1/component-1.npz') as saved:
    ray=saved['ray']
with np.load(root/'exact-602-replay-v1/results/fixture.npz') as saved:
    fixture={key:saved[key] for key in saved.files}
with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as saved:
    vertices,faces,tets=(saved[key] for key in ('vertices','surface_triangles','tetrahedra'))
records=json.loads((root/'exact-602-continuous-v1/summary.json').read_text())['records']
values=[]
for record in records:
    source,face=record['source'],record['face']
    triangle=vertices[faces[face]]
    normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
    normal/=np.linalg.norm(normal)
    owner=tets[np.flatnonzero(np.isin(tets,faces[face]).sum(axis=1)==3)[0]]
    opposite=owner[~np.isin(owner,faces[face])][0]
    if (vertices[opposite]-triangle[0])@normal<0:
        normal=-normal
    value=ray[:3]+np.cross(ray[3:],(triangle-fixture['center'][1])/fixture['radius'][1])
    compression=value@normal
    polar=compression+fixture['friction'][source]*np.linalg.norm(value-compression[:,None]*normal,axis=1)
    values.append({'source':source,'face':face,'full_triangle_polar_max':float(polar.max())})
print(json.dumps({'full_triangle_max':max(v['full_triangle_polar_max'] for v in values),'records':values},indent=2),flush=True)
