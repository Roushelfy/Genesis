"""Short live primitive/decomposition/SDF source coverage with independent full equilibrium fields."""

import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import quadrants as qd
import trimesh
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--shape', choices=('cube', 'u', 'cavity'), required=True)
parser.add_argument('--producer', choices=('primitive', 'decomposition', 'sdf'), required=True)
parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
source_files = [*Path('genesis/engine/solvers/rigid/stress').glob('*.py'), Path('genesis/options/rigid_stress.py'), Path(__file__)]
source_before = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
gs.init(backend={'cuda':gs.cuda, 'cpu':gs.cpu}[args.backend], precision='64', logging_level='warning')
shape = (1, 1, 1) if args.shape == 'cube' else (3, 3, 1) if args.shape == 'u' else (3, 3, 3)
cells = np.indices(shape).reshape((3, -1)).T
if args.shape == 'u':
    cells = cells[(cells[:, 0] != 1) | (cells[:, 1] == 0)]
elif args.shape == 'cavity':
    cells = cells[(cells != 1).any(axis=1)]
steps = np.eye(3, dtype=np.int32)[np.array(list(permutations(range(3))))]
offsets = np.concatenate((np.zeros((6, 1, 3), dtype=np.int32), steps.cumsum(axis=1)), axis=1)
coords = cells[:, None, None, :] + offsets
grid_shape = tuple(size + 1 for size in shape)
tetrahedra = np.ravel_multi_index(np.moveaxis(coords, -1, 0), grid_shape).reshape((-1, 4))
used, inverse = np.unique(tetrahedra, return_inverse=True)
vertices = np.indices(grid_shape).reshape((3, -1)).T[used] * 0.01
if args.shape == 'cube':
    vertices -= 0.005
tetrahedra = inverse.reshape((-1, 4))
local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
surface, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
surface = surface[counts == 1]
material_faces = vertices[surface]
normal = np.cross(material_faces[:, 1]-material_faces[:, 0], material_faces[:, 2]-material_faces[:, 0])
owners = np.array([np.flatnonzero(np.isin(tetrahedra, triangle).sum(axis=1)==3)[0] for triangle in surface])
opposite = np.array([tetrahedra[i][~np.isin(tetrahedra[i], triangle)][0] for i, triangle in zip(owners, surface)])
flip = (normal * (vertices[opposite]-material_faces[:, 0])).sum(axis=1)>0
collision_faces = surface.copy()
collision_faces[flip] = collision_faces[flip, ::-1]
surface[::2] = surface[::2, ::-1]
elastic, collision = args.output/'elastic.npz', args.output/'collision.obj'
np.savez(elastic, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
trimesh.Trimesh(vertices, collision_faces, process=False).export(collision)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
    rigid_options=gs.options.RigidOptions(use_gjk_collision=True, enable_contact_patch=True),
    show_viewer=False,
)
material = gs.materials.Rigid(friction=0.7, sdf_cell_size=0.0005, sdf_min_res=16, sdf_max_res=96)
if args.producer == 'primitive':
    assert args.shape == 'cube'
    target = scene.add_entity(gs.morphs.Box(size=(0.01, 0.01, 0.01)), material=material)
elif args.producer == 'decomposition':
    assert args.shape == 'u'
    mjcf = ET.Element('mujoco', model='u_decomposition')
    world = ET.SubElement(mjcf, 'worldbody')
    link = ET.SubElement(world, 'body', name='u_material')
    ET.SubElement(link, 'joint', type='free', name='root')
    # Exact authored disjoint-volume partition of the connected U material.
    for center, size in [((0.005,0.015,0.005),(0.005,0.015,0.005)),
                         ((0.025,0.015,0.005),(0.005,0.015,0.005)),
                         ((0.015,0.005,0.005),(0.005,0.005,0.005))]:
        ET.SubElement(link, 'geom', type='box', pos=' '.join(map(str, center)), size=' '.join(map(str, size)))
    target = scene.add_entity(gs.morphs.MJCF(file=ET.tostring(mjcf, encoding='unicode')), material=material)
else:
    assert args.shape in ('u', 'cavity')
    target = scene.add_entity(gs.morphs.Mesh(file=collision, convexify=False, decimate=False), material=material)
contact_position = np.array((0.011, 0.0221, 0.005) if args.shape == 'u' else
                            (0.011, 0.0153, 0.0147) if args.shape == 'cavity' else (0.0003, -0.0003, 0.006))
counterpart = scene.add_entity(gs.morphs.Box(size=(0.004, 0.004, 0.004), pos=contact_position, fixed=True),
                               material=gs.materials.Rigid(friction=0.2, rho=1000))
target.base_link.configure_stress_recovery(gs.options.RigidStressOptions(
    mesh=elastic, contact_model='intrinsic', intrinsic_support='pair_union',
    contact_radius=0.006, source_projection_limit=0.001, surface_edge_subdivisions=2,
    output_mode='full', method='inverse',
))
scene.build(n_envs=3)
entry = scene.rigid_solver.stress_recovery.links[0]
oracle = P2Shell(vertices, tetrahedra, surface, entry.options.young, entry.options.poisson,
                 entry.options.density, 2, factor_backend='none')
free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(entry.model.info.pins, transpose=True))
factor = splu(oracle.k[free][:,free].tocsc())
records = []
complete = 0
rotations = Rotation.from_euler('xyz', [[0,0,0],[15,-20,25],[-12,18,-9]],degrees=True)
xyzw = rotations.as_quat()
quaternions = xyzw[:,[3,0,1,2]]
translations = np.array([[0.1,0.1,0.1],[-0.05,0.05,0.05],[0.0,0.1,0.2]])
try:
    for tick in range(16):
        if tick == 8:
            scene.reset(envs_idx=[0])
        friction = (0.3, 0.7, 1.2)[tick // 6]
        for geom in target.base_link.geoms:
            geom.set_friction(friction)
        local = np.tile(contact_position, (3,1))
        local[:,1] += np.array([1,-1,0.5]) * (tick%4)*0.0001
        if tick in (4,5,12):
            local[0] += [0.1,0.1,0.1]
        counterpart.set_pos(rotations.apply(local)+translations)
        counterpart.set_quat(quaternions)
        target.set_pos(translations)
        target.set_quat(quaternions)
        target.set_dofs_velocity(np.column_stack((rotations.apply(np.tile([0,-0.003,0.001],(3,1))),np.zeros((3,3)))))
        scene.step()
        scene.rigid_solver.check_errno()
        active = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)
        force = qd_to_numpy(entry.contacts.force, transpose=True, copy=True)
        error_f = qd_to_numpy(entry.contacts.force_error, transpose=True, copy=True)
        error_m = qd_to_numpy(entry.contacts.moment_error, transpose=True, copy=True)
        radius = qd_to_numpy(entry.contacts.radius, transpose=True, copy=True)
        point = qd_to_numpy(entry.contacts.position, transpose=True, copy=True)
        nodal = qd_to_numpy(entry.state.force, transpose=True, copy=True)
        magnitude = np.linalg.norm(force,axis=-1)
        loaded = active & (magnitude>0)
        assert loaded[1:].any(axis=1).all()
        assert (error_f[loaded] <= 1e-8*magnitude[loaded]).all()
        assert (error_m[loaded] <= 1e-8*magnitude[loaded]*radius[loaded]).all()
        # Nodal wrench independently checks signed P2 scatter, including the original points.
        np.testing.assert_allclose(nodal.sum(axis=1), (force*active[...,None]).sum(axis=1), rtol=0, atol=1e-8*magnitude.sum(axis=1).max())
        # A moment compared at the reference origin includes p cross delta-F.
        # Propagate the independently accepted component errors into this frame;
        # the original-point per-component 1e-8 checks above stay unchanged.
        moment_actual = np.cross(oracle.xyz[None],nodal).sum(axis=1)
        moment_expected = np.cross(point,np.where(active[...,None],force,0)).sum(axis=1)
        moment_bound = (error_m+np.linalg.norm(point,axis=-1)*error_f).sum(axis=1)
        roundoff = 64*np.finfo(gs.np_float).eps * (
            (np.linalg.norm(point,axis=-1)*magnitude).sum(axis=1)+
            (np.linalg.norm(oracle.xyz,axis=-1)[None]*np.linalg.norm(nodal,axis=-1)).sum(axis=1))
        assert (np.linalg.norm(moment_actual-moment_expected,axis=1) <= moment_bound+roundoff).all()
        omega = qd_to_numpy(entry.omega, transpose=True, copy=True)
        centrifugal = -np.cross(omega[:,None],np.cross(omega[:,None],oracle.xyz[None]-oracle.com))
        raw = np.moveaxis(nodal,0,-1).reshape(oracle.ndof,3)
        raw += oracle.m @ np.moveaxis(centrifugal,0,-1).reshape(oracle.ndof,3)
        rhs = oracle.balance(raw)
        reference = np.zeros_like(rhs)
        reference[free] = factor.solve(rhs[free])
        native = qd_to_numpy(entry.state.displacement, transpose=True, copy=True).reshape(3,oracle.ndof).T
        residual = np.linalg.norm(oracle.k@native-rhs,axis=0)
        budget = np.maximum(1e-11,1e-8*np.linalg.norm(rhs,axis=0))
        assert (residual<=budget).all()
        tensor = qd_to_numpy(entry.state.stress_tensor,transpose=True,copy=True)
        vm = qd_to_numpy(entry.state.von_mises,transpose=True,copy=True)[...,0]
        peaks=[]
        for i_b in range(3):
            expected_tensor, expected_vm = stress_field(oracle.glambda,oracle.elements,reference[:,i_b],entry.options.young,entry.options.poisson)
            np.testing.assert_allclose(tensor[i_b],expected_tensor,rtol=1e-4,atol=1e-3)
            np.testing.assert_allclose(vm[i_b],expected_vm,rtol=1e-4,atol=1e-3)
            peaks.append(expected_vm.max())
        np.testing.assert_allclose(qd_to_numpy(entry.state.peak,transpose=True),peaks,rtol=1e-4,atol=1e-3)
        record={'tick':tick,'active_contacts':active.sum(axis=1).tolist(),
                'force_error_max':float((error_f[loaded]/magnitude[loaded]).max(initial=0)),
                'moment_error_max':float((error_m[loaded]/(magnitude[loaded]*radius[loaded])).max(initial=0)),
                'complete_residual_max_N':float(residual.max()),
                'friction_coefficients':np.unique(qd_to_numpy(entry.contacts.friction,transpose=True)[active]).tolist(),
                'methods':np.unique(qd_to_numpy(entry.intrinsic.raw_sources.method,transpose=True)[active]).tolist(),
                'peaks_Pa':peaks}
        records.append(record)
        print(json.dumps(record),flush=True)
        complete+=1
finally:
    arrays={'contacts_valid':entry.contacts.valid, 'contacts_position':entry.contacts.position,
            'contacts_force':entry.contacts.force, 'contacts_status':entry.contacts.status,
            'source_witness':entry.intrinsic.raw_sources.witness, 'source_status':entry.intrinsic.raw_sources.status,
            'source_method':entry.intrinsic.raw_sources.method, 'source_geom':entry.intrinsic.raw_sources.geom,
            'source_pair':entry.intrinsic.raw_sources.pair, 'source_anchor':entry.contacts.center}
    np.savez_compressed(args.output/'last-native-state.npz',**{name:qd_to_numpy(value,transpose=True,copy=True) for name,value in arrays.items()})
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    assert source_before==after
    (args.output/'summary.json').write_text(json.dumps({'arguments':vars(args)|{'output':str(args.output)},
        'complete_steps':complete,'records':records,'source_sha256':source_before,'source_sha256_after':after,
        'scope':'Intrusive actual-collider live recovery and CPU full-field verification; no performance or independent load-fit claim.'},indent=2)+'\n')
