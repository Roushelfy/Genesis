"""Capture a live producer frame and replay its failed geom pair before rigid integration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.collider import gjk
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils import array_class as ac
from genesis.utils.misc import qd_to_numpy

candidate_path = Path(__file__).with_name('gjk_patch_normal_candidate_v1.py')
spec = importlib.util.spec_from_file_location('genesis.engine.solvers.rigid.collider.gjk_replay_candidate', candidate_path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@qd.kernel
def replay(i_ga: int, i_gb: int, env: int, dyn_state: ac.DynState,
           collider_state: ac.ColliderState, scratch: ac.GJKState,
           dyn_info: ac.DynInfo, rigid_info: ac.RigidInfo, collider_info: ac.ColliderInfo,
           rigid_config: qd.template(), collider_config: qd.template(),
           gjk_config: qd.template(), use_candidate: qd.template()):
    if qd.static(use_candidate):
        candidate.func_gjk_contact(i_ga, i_gb, 0,
            dyn_state.geoms.pos[i_ga, env], dyn_state.geoms.quat[i_ga, env],
            dyn_state.geoms.pos[i_gb, env], dyn_state.geoms.quat[i_gb, env],
            dyn_state, collider_state, scratch, dyn_info, rigid_info, collider_info,
            rigid_config, collider_config, gjk_config)
    else:
        gjk.func_gjk_contact(i_ga, i_gb, 0,
            dyn_state.geoms.pos[i_ga, env], dyn_state.geoms.quat[i_ga, env],
            dyn_state.geoms.pos[i_gb, env], dyn_state.geoms.quat[i_gb, env],
            dyn_state, collider_state, scratch, dyn_info, rigid_info, collider_info,
            rigid_config, collider_config, gjk_config)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
gs.init(backend=gs.cuda, precision='64')
workload = FrankaEgg(32, seed=623101, varied=True, output_mode='full',
    contact_model='intrinsic', contact_block_size=4, egg_geometry='mesh64',
    use_gjk_collision=True, enable_contact_patch=True)
solver = workload.scene.rigid_solver
collider = solver.collider
scratch = ac.get_gjk_state(1, solver.rigid_config, collider.gjk._gjk_info, False)
original_recover = RigidStressRecovery.recover
tick = -1
report = {'scope': __doc__, 'tick': 710, 'env': 5}


def capture(self, i_substep):
    if tick == 710:
        arrays = {
            'geom_position': solver.dyn_state.geoms.pos, 'geom_quaternion': solver.dyn_state.geoms.quat,
            'geom_vertex_start': solver.dyn_info.geoms.vert_start, 'geom_vertex_end': solver.dyn_info.geoms.vert_end,
            'geom_link': solver.dyn_info.geoms.link_idx, 'geom_type': solver.dyn_info.geoms.type,
            'vertices_local': solver.dyn_info.verts.init_pos,
            'contact_count': collider.collider_state.n_contacts, 'contact_order': collider.collider_state.contact_sort_idx,
            'contact_geom_a': collider.collider_state.contact_data.geom_a,
            'contact_geom_b': collider.collider_state.contact_data.geom_b,
            'contact_position': collider.collider_state.contact_data.pos,
            'contact_normal': collider.collider_state.contact_data.normal,
            'contact_force': collider.collider_state.contact_data.force,
            'contact_friction': collider.collider_state.contact_data.friction,
            'witness_a': collider.collider_state.contact_data.witness.point_a,
            'witness_b': collider.collider_state.contact_data.witness.point_b,
            'witness_method': collider.collider_state.contact_data.witness.method,
        }
        values = {name: qd_to_numpy(value, transpose=True) for name, value in arrays.items()}
        np.savez_compressed(args.output / 'before-recovery-710.npz', **values)
        (args.output / 'geometry-description.json').write_text(json.dumps([
            {'idx': g.idx, 'type': int(g.type), 'vertex_start': g.vert_start,
             'faces': np.asarray(g.get_trimesh().faces).tolist()} for g in solver.geoms])+'\n')
        with np.load(Path(__file__).parent / 'online-square-fan-v1/results/failure/snapshot-710.npz') as old:
            geom = int(old['geom'][5, 0])
            witness = old['witness'][5, 0]
        match = []
        for i in range(int(values['contact_count'][5])):
            ga, gb = int(values['contact_geom_a'][5, i]), int(values['contact_geom_b'][5, i])
            point = values['witness_a'][5, i] if ga == geom else values['witness_b'][5, i]
            if geom in (ga, gb) and np.linalg.norm(point-witness) < 1e-12:
                match.append((i, ga, gb))
        report['matched_contacts'] = match
        assert len(match) == 1, match
        _, ga, gb = match[0]
        for use_candidate in (False, True):
            replay(ga, gb, 5, solver.dyn_state, collider.collider_state, scratch,
                solver.dyn_info, solver.rigid_info, collider.collider_info,
                solver.rigid_config, collider.collider_config, collider.gjk.gjk_config, use_candidate)
            qd.sync()
            state = {name: qd_to_numpy(getattr(scratch, name), transpose=True) for name in
                ('normal', 'contact_pos', 'n_contacts', 'n_witness', 'multi_contact_flag', 'nearest_face', 'distance')}
            state.update({f'witness_{name}': qd_to_numpy(getattr(scratch.witness, name), transpose=True)
                for name in ('point_obj1', 'point_obj2')})
            for owner_name, names in (('polytope', ('nverts', 'nfaces', 'nfaces_map')),
                ('polytope_faces', ('normal', 'verts_idx', 'dist2')),
                ('polytope_verts', ('id1', 'id2', 'obj1', 'obj2', 'mink'))):
                owner = getattr(scratch, owner_name)
                state.update({f'{owner_name}_{name}': qd_to_numpy(getattr(owner, name), transpose=True) for name in names})
            label = 'patch_witness' if use_candidate else 'epa_override'
            np.savez_compressed(args.output / f'replay-{label}.npz', **state)
            report[label] = {name: state[name].tolist() for name in
                ('normal', 'n_contacts', 'n_witness', 'multi_contact_flag', 'nearest_face', 'distance')}
    return original_recover(self, i_substep)


try:
    with patch.object(RigidStressRecovery, 'recover', capture):
        for tick in range(711):
            workload.step()
            solver.check_errno()
except Exception as exc:
    report['exception'] = str(exc)
finally:
    report['completed_steps'] = tick
    report['candidate_sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
