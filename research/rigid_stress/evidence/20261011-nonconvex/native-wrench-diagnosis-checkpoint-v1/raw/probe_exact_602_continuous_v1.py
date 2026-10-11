"""Conservative continuous Euclidean disk/triangle upper bounds for the sampled separating ray."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent
output = root / 'exact-602-continuous-v1'
output.mkdir(parents=True, exist_ok=False)
with np.load(root / 'exact-602-socp-v1/component-1.npz') as saved:
    ray, target = saved['ray'], saved['target']
with np.load(root / 'exact-602-replay-v1/results/fixture.npz') as saved:
    fixture = {key: saved[key].copy() for key in saved.files}
with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as saved:
    vertices, tets, faces = (saved[key] for key in ('vertices','tetrahedra','surface_triangles'))
triangles = vertices[faces]
raw_normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
raw_normals /= np.linalg.norm(raw_normals,axis=1)[:,None]
owners = np.array([np.flatnonzero(np.isin(tets,face).sum(axis=1)==3)[0] for face in faces])
opposite = np.array([tets[owner][~np.isin(tets[owner],face)][0] for face,owner in zip(faces,owners)])
outward = raw_normals * np.where(np.einsum('fi,fi->f',raw_normals,vertices[opposite]-triangles[:,0])>0,-1,1)[:,None]
anchor, target_radius = fixture['center'][1], fixture['radius'][1]
records = []

for source, (center, radius, mu) in enumerate(zip(fixture['center'],fixture['radius'],fixture['friction'])):
    for face, (triangle, normal) in enumerate(zip(triangles,-outward)):
        distance = (center-triangle[0])@normal
        if abs(distance) >= radius:
            continue
        circle_center = center-distance*normal
        circle_radius = np.sqrt(radius*radius-distance*distance)
        e0 = triangle[1]-triangle[0]
        e0 /= np.linalg.norm(e0)
        e1 = np.cross(normal,e0)
        coordinates = np.column_stack(((triangle-circle_center)@e0,(triangle-circle_center)@e1))
        bary_dual = np.linalg.pinv(np.column_stack((coordinates[1]-coordinates[0],coordinates[2]-coordinates[0])))
        points = [v for v in coordinates if np.linalg.norm(v)<=circle_radius]
        angles = [0.,2*np.pi]
        for first, second in zip(coordinates,np.roll(coordinates,-1,axis=0)):
            direction = second-first
            a,b,c = direction@direction,2*first@direction,first@first-circle_radius**2
            discriminant = b*b-4*a*c
            if discriminant>=0:
                for fraction in ((-b-np.sqrt(discriminant))/(2*a),(-b+np.sqrt(discriminant))/(2*a)):
                    if 0<=fraction<=1:
                        point = first+fraction*direction
                        points.append(point)
                        angles.append(np.arctan2(point[1],point[0])%(2*np.pi))
        def polar(local):
            xyz = circle_center + local[:,0,None]*e0 + local[:,1,None]*e1
            value = ray[:3]+np.cross(ray[3:],(xyz-anchor)/target_radius)
            compression = value@normal
            return compression+mu*np.linalg.norm(value-compression[:,None]*normal,axis=1), xyz
        maximum = -np.inf
        witness = np.full(3,np.nan)
        if points:
            values,xyz=polar(np.asarray(points))
            maximum=float(values.max())
            witness=xyz[values.argmax()]
        angles=np.unique(angles)
        arc_upper=-np.inf
        for first,second in zip(angles[:-1],angles[1:]):
            middle=(first+second)/2
            point=circle_radius*np.array([np.cos(middle),np.sin(middle)])
            uv=bary_dual@(point-coordinates[0])
            if (uv>=0).all() and uv.sum()<=1:
                count=max(1,int(np.ceil((second-first)*16384/(2*np.pi))))
                theta=np.linspace(first,second,count+1)
                local=circle_radius*np.column_stack((np.cos(theta),np.sin(theta)))
                values,xyz=polar(local)
                lipschitz=(1+mu)*np.linalg.norm(ray[3:])*circle_radius/target_radius
                bound=float(values.max()+lipschitz*(second-first)/(2*count))
                arc_upper=max(arc_upper,bound)
                if values.max()>maximum:
                    maximum=float(values.max())
                    witness=xyz[values.argmax()]
        if np.isfinite(maximum):
            record={'source':source,'face':face,'polar_observed_max':maximum,
                    'polar_upper_bound':max(maximum,arc_upper),'witness_position_m':witness.tolist()}
            records.append(record)
summary={'certificate_margin':float(target@ray),
         'conservative_polar_upper_bound':max(r['polar_upper_bound'] for r in records),
         'note':'Upper domain includes every boundary triangle intersecting either Euclidean disk, without intrinsic path restrictions. A negative bound proves continuous intrinsic incompatibility; a positive bound does not establish feasibility.',
         'records':records}
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
