"""Export the production ray and independently bound its continuous material-boundary support."""
import json
import sys
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.compatibility import create_wrench_search, kernel_search_wrench_certificate, kernel_certify_continuous_wrench
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import create_footprints, create_surface_fan_tractions, kernel_footprint_ports, kernel_surface_fan_profiles
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, kernel_fit_tractions
from genesis.utils.misc import qd_to_numpy

root = Path(__file__).parent
backend = sys.argv[1]
output = root / f'native-wrench-certificate-audit-v1/{backend}/results'
output.mkdir(parents=True, exist_ok=False)
gs.init(backend={'cpu': gs.cpu, 'cuda': gs.cuda}[backend], precision='64', logging_level='warning')
with np.load('tests/rigid/assets/nonconvex_mesh64_tick602.npz') as saved:
    centers, points, forces, radii, mu = (saved[key] for key in ('anchor','position','force','radius','friction'))
    anchor_faces = saved['anchor_face']
model = StressModel(gs.options.RigidStressOptions(mesh='examples/rigid/assets/hollow_egg/level1/elastic.npz', method='pcg'))
surface = StressSurface(10, model.info)
paths = StressSurfacePaths(model.info, 8)
contacts, sources = create_contacts(2,1,float(radii[0])), create_sources(2,1)
contacts.valid.fill(True)
contacts.center.from_numpy(centers[:,None])
contacts.position.from_numpy(points[:,None])
contacts.force.from_numpy(forces[:,None])
contacts.radius.from_numpy(radii[:,None])
contacts.friction.from_numpy(mu[:,None])
contacts.anchor_face.from_numpy(anchor_faces[:,None])
sources.status.fill(0)
sources.geom.fill(24)
sources.pair.fill(22)
sources.free_moment.fill(0)
moment = np.cross(points-centers,forces)
sources.anchor_moment.from_numpy(moment[:,None])
footprints = create_footprints(paths.info.positions.shape[0],2,1)
tractions = create_surface_fan_tractions(surface.info,2,1,shared_support=True)
info, search = create_traction_info(surface.info), create_wrench_search(2,1,8)
epsilon = float(np.finfo(gs.np_float).eps)
kernel_footprint_ports(contacts,sources,footprints,surface.info,paths.info)
kernel_surface_fan_profiles(contacts,footprints,tractions,model.info,surface.info,paths.info,info)
kernel_bind_traction_support(contacts,sources,footprints,tractions)
kernel_fit_tractions(contacts,sources,tractions,info,epsilon,9e-9,128)
kernel_search_wrench_certificate(contacts,sources,tractions,search,info,epsilon,1e-8)
kernel_certify_continuous_wrench(contacts,sources,tractions,search,surface.info,epsilon,8,16384,1e-8)
statuses = qd_to_numpy(tractions.status,transpose=True,copy=True)
assert statuses.tolist() == [[0,6]]
ray = qd_to_numpy(tractions.certificate,transpose=True,copy=True)[0,1]
scale = max(np.linalg.norm(forces[1]),np.linalg.norm(moment[1])/radii[1])
target = np.concatenate((forces[1],moment[1]/radii[1]))/scale
np.savez_compressed(output/'native.npz',ray=ray,target=target,centers=centers,forces=forces,points=points,
                    radius=radii,friction=mu,status=statuses,
                    native_bound=qd_to_numpy(search.bound,transpose=True,copy=True),
                    native_nodes=qd_to_numpy(search.visited,transpose=True,copy=True),
                    native_evaluations=qd_to_numpy(search.evaluations,transpose=True,copy=True))

with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as saved:
    vertices,tets,faces = (saved[key] for key in ('vertices','tetrahedra','surface_triangles'))
triangles = vertices[faces]
raw_normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
raw_normals /= np.linalg.norm(raw_normals,axis=1)[:,None]
owners = np.array([np.flatnonzero(np.isin(tets,face).sum(axis=1)==3)[0] for face in faces])
opposite = np.array([tets[owner][~np.isin(tets[owner],face)][0] for face,owner in zip(faces,owners)])
outward = raw_normals*np.where(np.einsum('fi,fi->f',raw_normals,vertices[opposite]-triangles[:,0])>0,-1,1)[:,None]
anchor,target_radius = centers[1],radii[1]
records = []
for source,(center,radius,friction) in enumerate(zip(centers,radii,mu)):
    for face,(triangle,normal) in enumerate(zip(triangles,-outward)):
        distance = (center-triangle[0])@normal
        if abs(distance)>=radius:
            continue
        circle_center = center-distance*normal
        circle_radius = np.sqrt(radius*radius-distance*distance)
        e0 = triangle[1]-triangle[0]
        e0 /= np.linalg.norm(e0)
        e1 = np.cross(normal,e0)
        coordinates = np.column_stack(((triangle-circle_center)@e0,(triangle-circle_center)@e1))
        bary_dual = np.linalg.pinv(np.column_stack((coordinates[1]-coordinates[0],coordinates[2]-coordinates[0])))
        candidates = [v for v in coordinates if np.linalg.norm(v)<=circle_radius]
        angles = [0.,2*np.pi]
        for first,second in zip(coordinates,np.roll(coordinates,-1,axis=0)):
            direction = second-first
            a,b,c = direction@direction,2*first@direction,first@first-circle_radius**2
            discriminant = b*b-4*a*c
            if discriminant>=0:
                for fraction in ((-b-np.sqrt(discriminant))/(2*a),(-b+np.sqrt(discriminant))/(2*a)):
                    if 0<=fraction<=1:
                        point = first+fraction*direction
                        candidates.append(point)
                        angles.append(np.arctan2(point[1],point[0])%(2*np.pi))
        maximum = -np.inf
        if candidates:
            local = np.array(candidates)
            xyz = circle_center+local[:,0,None]*e0+local[:,1,None]*e1
            value = ray[:3]+np.cross(ray[3:],(xyz-anchor)/target_radius)
            compression = value@normal
            maximum = float((compression+friction*np.linalg.norm(value-compression[:,None]*normal,axis=1)).max())
        angles = np.unique(angles)
        for first,second in zip(angles[:-1],angles[1:]):
            middle = (first+second)/2
            point = circle_radius*np.array([np.cos(middle),np.sin(middle)])
            uv = bary_dual@(point-coordinates[0])
            if (uv>=0).all() and uv.sum()<=1:
                count = max(1,int(np.ceil((second-first)*16384/(2*np.pi))))
                theta = np.linspace(first,second,count+1)
                local = circle_radius*np.column_stack((np.cos(theta),np.sin(theta)))
                xyz = circle_center+local[:,0,None]*e0+local[:,1,None]*e1
                value = ray[:3]+np.cross(ray[3:],(xyz-anchor)/target_radius)
                compression = value@normal
                values = compression+friction*np.linalg.norm(value-compression[:,None]*normal,axis=1)
                lipschitz = (1+friction)*np.linalg.norm(ray[3:])*circle_radius/target_radius
                maximum = max(maximum,float(values.max()+lipschitz*(second-first)/(2*count)))
        if np.isfinite(maximum):
            records.append({'source':source,'face':face,'polar_upper_bound':maximum})
summary = {'backend':backend,'status':statuses.tolist(),'ray':ray.tolist(),
           'certificate_margin':float(target@ray),'cpu_conservative_polar_upper_bound':max(record['polar_upper_bound'] for record in records),
           'native_bound':qd_to_numpy(search.bound,transpose=True,copy=True).tolist(),
           'native_nodes':qd_to_numpy(search.visited,transpose=True,copy=True).tolist(),
           'native_evaluations':qd_to_numpy(search.evaluations,transpose=True,copy=True).tolist(),'records':records,
           'scope':'Independent CPU triangle/disk clipping and analytic arc bounds audit the production ray; no physical integration convergence claim.'}
assert summary['cpu_conservative_polar_upper_bound'] < -1e-4
assert summary['certificate_margin'] > 1e-3
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
